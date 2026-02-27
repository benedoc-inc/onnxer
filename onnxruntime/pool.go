package onnxruntime

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"time"
)

// SessionPool manages a pool of inference sessions for safe concurrent use.
// Each goroutine borrows a session, runs inference, and returns it automatically.
//
// Example:
//
//	pool, err := onnxruntime.NewSessionPool(runtime, env, modelBytes, 8, nil)
//	if err != nil {
//	    log.Fatal(err)
//	}
//	defer pool.Close()
//
//	// Safe to call from many goroutines:
//	outputs, err := pool.Run(ctx, map[string]*Value{"input": tensor})
type SessionPool struct {
	sessions chan *Session
	runtime  *Runtime
	closed   atomic.Bool
	hooks    []Hook
	inflight sync.WaitGroup // tracks in-flight Run calls

	// cached from first session (all sessions share the same model)
	inputNames  []string
	outputNames []string

	// prepacked weights shared across all sessions
	prepackedWeights     *PrepackedWeightsContainer
	ownsPrepackedWeights bool // true if pool created the container and should close it

	// metrics
	totalRuns    atomic.Int64
	totalErrors  atomic.Int64
	totalLatency atomic.Int64 // nanoseconds
}

// PoolConfig configures session pool behavior.
type PoolConfig struct {
	// SessionOptions applied to every session in the pool.
	SessionOptions *SessionOptions

	// Hooks are called around every Run invocation.
	Hooks []Hook

	// SharePrepackedWeights enables sharing of pre-packed kernel weights across
	// all sessions in the pool. This significantly reduces memory usage because
	// the packed weight buffers are allocated once and shared rather than
	// duplicated per session. Recommended for pools with 2+ sessions.
	SharePrepackedWeights bool
}

// NewSessionPool creates a pool of n sessions from the given model data.
// All sessions share the same Runtime and Env but are independent for concurrent use.
func NewSessionPool(runtime *Runtime, env *Env, modelData []byte, n int, config *PoolConfig) (*SessionPool, error) {
	if n <= 0 {
		return nil, fmt.Errorf("pool size must be positive, got %d", n)
	}
	if len(modelData) == 0 {
		return nil, fmt.Errorf("model data cannot be empty")
	}

	var opts *SessionOptions
	var hooks []Hook
	var shareWeights bool
	if config != nil {
		opts = config.SessionOptions
		hooks = config.Hooks
		shareWeights = config.SharePrepackedWeights
	}

	pool := &SessionPool{
		sessions: make(chan *Session, n),
		runtime:  runtime,
		hooks:    hooks,
	}

	if shareWeights {
		container, err := runtime.NewPrepackedWeightsContainer()
		if err != nil {
			return nil, fmt.Errorf("failed to create prepacked weights container: %w", err)
		}
		pool.prepackedWeights = container
		pool.ownsPrepackedWeights = true
	}

	for i := 0; i < n; i++ {
		session, err := runtime.newSessionFromBytes(env, modelData, opts, pool.prepackedWeights)
		if err != nil {
			pool.Close()
			return nil, fmt.Errorf("failed to create session %d: %w", i, err)
		}
		if i == 0 {
			pool.inputNames = session.InputNames()
			pool.outputNames = session.OutputNames()
		}
		pool.sessions <- session
	}

	return pool, nil
}

// NewSessionPoolFromFile creates a pool of n sessions from a model file path.
func NewSessionPoolFromFile(runtime *Runtime, env *Env, modelPath string, n int, config *PoolConfig) (*SessionPool, error) {
	if n <= 0 {
		return nil, fmt.Errorf("pool size must be positive, got %d", n)
	}

	var opts *SessionOptions
	var hooks []Hook
	var shareWeights bool
	if config != nil {
		opts = config.SessionOptions
		hooks = config.Hooks
		shareWeights = config.SharePrepackedWeights
	}

	pool := &SessionPool{
		sessions: make(chan *Session, n),
		runtime:  runtime,
		hooks:    hooks,
	}

	if shareWeights {
		container, err := runtime.NewPrepackedWeightsContainer()
		if err != nil {
			return nil, fmt.Errorf("failed to create prepacked weights container: %w", err)
		}
		pool.prepackedWeights = container
		pool.ownsPrepackedWeights = true
	}

	for i := 0; i < n; i++ {
		session, err := runtime.newSessionFromFile(env, modelPath, opts, pool.prepackedWeights)
		if err != nil {
			pool.Close()
			return nil, fmt.Errorf("failed to create session %d: %w", i, err)
		}
		if i == 0 {
			pool.inputNames = session.InputNames()
			pool.outputNames = session.OutputNames()
		}
		pool.sessions <- session
	}

	return pool, nil
}

// Run borrows a session from the pool, executes inference, and returns the session.
// It blocks until a session is available or ctx is cancelled.
// This is safe to call from multiple goroutines concurrently.
func (p *SessionPool) Run(ctx context.Context, inputs map[string]*Value, opts ...RunOption) (map[string]*Value, error) {
	if p.closed.Load() {
		return nil, fmt.Errorf("session pool is closed")
	}

	// Fast path: short-circuit if context is already cancelled
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	// Track in-flight run so Close() waits for us
	p.inflight.Add(1)
	defer p.inflight.Done()

	// Borrow a session
	var session *Session
	select {
	case session = <-p.sessions:
	case <-ctx.Done():
		return nil, ctx.Err()
	}

	// Always return the session to the pool
	defer func() {
		if !p.closed.Load() {
			p.sessions <- session
		} else {
			session.Close()
		}
	}()

	// Run hooks
	info := &RunInfo{
		InputNames: keys(inputs),
	}
	for _, h := range p.hooks {
		h.BeforeRun(info)
	}

	start := time.Now()
	outputs, err := session.Run(ctx, inputs, opts...)
	elapsed := time.Since(start)

	info.Duration = elapsed
	info.Error = err
	if outputs != nil {
		info.OutputNames = keys(outputs)
	}

	p.totalRuns.Add(1)
	p.totalLatency.Add(int64(elapsed))
	if err != nil {
		p.totalErrors.Add(1)
	}

	for _, h := range p.hooks {
		h.AfterRun(info)
	}

	return outputs, err
}

// Size returns the total number of sessions in the pool.
func (p *SessionPool) Size() int {
	return cap(p.sessions)
}

// Available returns the number of idle sessions currently available.
func (p *SessionPool) Available() int {
	return len(p.sessions)
}

// Stats returns pool usage statistics.
func (p *SessionPool) Stats() PoolStats {
	return PoolStats{
		TotalRuns:         p.totalRuns.Load(),
		TotalErrors:       p.totalErrors.Load(),
		TotalLatency:      time.Duration(p.totalLatency.Load()),
		PoolSize:          cap(p.sessions),
		AvailableSessions: len(p.sessions),
	}
}

// PoolStats contains pool usage statistics.
type PoolStats struct {
	TotalRuns         int64
	TotalErrors       int64
	TotalLatency      time.Duration
	PoolSize          int
	AvailableSessions int
}

// AvgLatency returns the average inference latency, or 0 if no runs have completed.
func (s PoolStats) AvgLatency() time.Duration {
	if s.TotalRuns == 0 {
		return 0
	}
	return s.TotalLatency / time.Duration(s.TotalRuns)
}

// Warmup pre-runs inference on every session in the pool to warm JIT caches,
// trigger graph optimizations, and allocate internal buffers. This reduces
// latency variance on the first real requests.
//
// The inputs map should contain representative data matching your model's
// input schema. Each session runs inference once and the outputs are discarded.
func (p *SessionPool) Warmup(ctx context.Context, inputs map[string]*Value) error {
	if p.closed.Load() {
		return fmt.Errorf("session pool is closed")
	}

	size := cap(p.sessions)
	for i := 0; i < size; i++ {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		outputs, err := p.Run(ctx, inputs)
		if err != nil {
			return fmt.Errorf("warmup run %d/%d failed: %w", i+1, size, err)
		}
		for _, v := range outputs {
			v.Close()
		}
	}

	return nil
}

// HealthCheck verifies the pool can execute inference by running a single
// inference with the provided inputs. Returns nil if healthy.
//
// Use this in readiness probes (e.g., Kubernetes /healthz endpoints).
func (p *SessionPool) HealthCheck(ctx context.Context, inputs map[string]*Value) error {
	if p.closed.Load() {
		return fmt.Errorf("session pool is closed")
	}

	outputs, err := p.Run(ctx, inputs)
	if err != nil {
		return fmt.Errorf("health check failed: %w", err)
	}
	for _, v := range outputs {
		v.Close()
	}
	return nil
}

// ResetStats resets all pool usage statistics to zero.
func (p *SessionPool) ResetStats() {
	p.totalRuns.Store(0)
	p.totalErrors.Store(0)
	p.totalLatency.Store(0)
}

// Close waits for in-flight runs to complete, then drains the pool and closes all sessions.
func (p *SessionPool) Close() {
	if !p.closed.CompareAndSwap(false, true) {
		return
	}

	// Wait for in-flight runs to finish and return their sessions
	p.inflight.Wait()

	// Drain and close all sessions
	close(p.sessions)
	for session := range p.sessions {
		session.Close()
	}

	// Release shared prepacked weights after all sessions are closed
	if p.ownsPrepackedWeights && p.prepackedWeights != nil {
		p.prepackedWeights.Close()
		p.prepackedWeights = nil
	}
}

// InputNames returns the model's input names.
// This is safe to call concurrently — names are cached at pool creation time.
func (p *SessionPool) InputNames() []string {
	return p.inputNames
}

// OutputNames returns the model's output names.
// This is safe to call concurrently — names are cached at pool creation time.
func (p *SessionPool) OutputNames() []string {
	return p.outputNames
}

func keys[V any](m map[string]V) []string {
	result := make([]string, 0, len(m))
	for k := range m {
		result = append(result, k)
	}
	return result
}
