package onnxruntime

import (
	"fmt"
	"runtime"

	"github.com/benedoc-inc/onnxer/onnxruntime/internal/api"
)

// ThreadingOptions configures global thread pools that can be shared across
// all sessions created from the same environment. This is more efficient than
// per-session thread pools when running many concurrent sessions, as it avoids
// creating redundant OS threads.
//
// Usage:
//
//	threadOpts, _ := runtime.NewThreadingOptions()
//	defer threadOpts.Close()
//	threadOpts.SetIntraOpNumThreads(4)
//	threadOpts.SetInterOpNumThreads(2)
//
//	env, _ := runtime.NewEnvWithGlobalThreadPools("app", ort.LoggingLevelWarning, threadOpts)
//	defer env.Close()
//
//	// Sessions must disable per-session threads to use the global pool:
//	session, _ := runtime.NewSessionFromReader(env, reader, &ort.SessionOptions{
//	    DisablePerSessionThreads: true,
//	})
type ThreadingOptions struct {
	ptr     api.OrtThreadingOptions
	runtime *Runtime
}

// NewThreadingOptions creates new threading options for configuring global thread pools.
func (r *Runtime) NewThreadingOptions() (*ThreadingOptions, error) {
	var ptr api.OrtThreadingOptions
	status := r.apiFuncs.CreateThreadingOptions(&ptr)
	if err := r.statusError(status); err != nil {
		return nil, fmt.Errorf("failed to create threading options: %w", err)
	}

	t := &ThreadingOptions{
		ptr:     ptr,
		runtime: r,
	}
	runtime.AddCleanup(t, func(_ struct{}) { t.Close() }, struct{}{})
	return t, nil
}

// SetIntraOpNumThreads sets the number of threads for parallelism within
// individual operators (e.g., matrix multiply, convolution).
func (t *ThreadingOptions) SetIntraOpNumThreads(n int) error {
	status := t.runtime.apiFuncs.SetGlobalIntraOpNumThreads(t.ptr, int32(n))
	if err := t.runtime.statusError(status); err != nil {
		return fmt.Errorf("failed to set global intra-op threads: %w", err)
	}
	return nil
}

// SetInterOpNumThreads sets the number of threads for parallelism across
// independent operators in the graph.
func (t *ThreadingOptions) SetInterOpNumThreads(n int) error {
	status := t.runtime.apiFuncs.SetGlobalInterOpNumThreads(t.ptr, int32(n))
	if err := t.runtime.statusError(status); err != nil {
		return fmt.Errorf("failed to set global inter-op threads: %w", err)
	}
	return nil
}

// SetSpinControl controls whether threads spin-wait instead of blocking.
// Spinning reduces latency at the cost of higher CPU usage when idle.
// When false, threads block (lower CPU usage, slightly higher latency).
func (t *ThreadingOptions) SetSpinControl(allow bool) error {
	val := int32(0)
	if allow {
		val = 1
	}
	status := t.runtime.apiFuncs.SetGlobalSpinControl(t.ptr, val)
	if err := t.runtime.statusError(status); err != nil {
		return fmt.Errorf("failed to set global spin control: %w", err)
	}
	return nil
}

// Close releases the threading options.
// It is safe to call Close multiple times.
func (t *ThreadingOptions) Close() {
	if t.ptr != 0 && t.runtime != nil && t.runtime.apiFuncs != nil {
		t.runtime.apiFuncs.ReleaseThreadingOptions(t.ptr)
		t.ptr = 0
	}
}

// NewEnvWithGlobalThreadPools creates an environment that uses global thread pools
// configured by the given ThreadingOptions. All sessions created from this environment
// should set DisablePerSessionThreads: true in their SessionOptions to use the
// shared thread pool instead of creating per-session threads.
func (r *Runtime) NewEnvWithGlobalThreadPools(logID string, logLevel LoggingLevel, threadingOpts *ThreadingOptions) (*Env, error) {
	logIDBytes := append([]byte(logID), 0)
	var envPtr api.OrtEnv

	status := r.apiFuncs.CreateEnvWithGlobalThreadPools(logLevel, &logIDBytes[0], threadingOpts.ptr, &envPtr)
	if err := r.statusError(status); err != nil {
		return nil, fmt.Errorf("failed to create environment with global thread pools: %w", err)
	}

	env := &Env{
		ptr:     envPtr,
		runtime: r,
	}
	runtime.AddCleanup(env, func(_ struct{}) { env.Close() }, struct{}{})
	return env, nil
}
