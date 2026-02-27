package onnxruntime

import (
	"context"
	"fmt"
	"io"
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
	if config != nil {
		opts = config.SessionOptions
		hooks = config.Hooks
	}

	pool := &SessionPool{
		sessions: make(chan *Session, n),
		runtime:  runtime,
		hooks:    hooks,
	}

	for i := 0; i < n; i++ {
		reader := newBytesReader(modelData)
		session, err := runtime.NewSessionFromReader(env, reader, opts)
		if err != nil {
			pool.Close()
			return nil, fmt.Errorf("failed to create session %d: %w", i, err)
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
	if config != nil {
		opts = config.SessionOptions
		hooks = config.Hooks
	}

	pool := &SessionPool{
		sessions: make(chan *Session, n),
		runtime:  runtime,
		hooks:    hooks,
	}

	for i := 0; i < n; i++ {
		session, err := runtime.NewSession(env, modelPath, opts)
		if err != nil {
			pool.Close()
			return nil, fmt.Errorf("failed to create session %d: %w", i, err)
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

// Close drains the pool and closes all sessions.
func (p *SessionPool) Close() {
	if !p.closed.CompareAndSwap(false, true) {
		return
	}

	// Drain and close all sessions
	close(p.sessions)
	for session := range p.sessions {
		session.Close()
	}
}

// InputNames returns the model's input names (from the first session).
// Returns nil if the pool is closed.
func (p *SessionPool) InputNames() []string {
	if p.closed.Load() {
		return nil
	}

	// Peek at a session without removing it
	session := <-p.sessions
	names := session.InputNames()
	p.sessions <- session
	return names
}

// OutputNames returns the model's output names (from the first session).
// Returns nil if the pool is closed.
func (p *SessionPool) OutputNames() []string {
	if p.closed.Load() {
		return nil
	}

	session := <-p.sessions
	names := session.OutputNames()
	p.sessions <- session
	return names
}

// bytesReader implements io.Reader over a byte slice.
// We avoid importing bytes to keep dependencies minimal.
type bytesReader struct {
	data []byte
	pos  int
}

func newBytesReader(data []byte) io.Reader {
	return &bytesReader{data: data}
}

func (r *bytesReader) Read(p []byte) (n int, err error) {
	if r.pos >= len(r.data) {
		return 0, io.EOF
	}
	n = copy(p, r.data[r.pos:])
	r.pos += n
	return n, nil
}

func keys[V any](m map[string]V) []string {
	result := make([]string, 0, len(m))
	for k := range m {
		result = append(result, k)
	}
	return result
}
