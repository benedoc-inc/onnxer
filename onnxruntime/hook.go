package onnxruntime

import "time"

// Hook provides callbacks around inference execution for observability.
// Implement this interface to add metrics, logging, or tracing.
//
// Example:
//
//	type metricsHook struct {
//	    histogram prometheus.Histogram
//	}
//
//	func (h *metricsHook) BeforeRun(info *RunInfo) {}
//	func (h *metricsHook) AfterRun(info *RunInfo) {
//	    h.histogram.Observe(info.Duration.Seconds())
//	    if info.Error != nil {
//	        errorCounter.Inc()
//	    }
//	}
type Hook interface {
	// BeforeRun is called before inference starts.
	BeforeRun(info *RunInfo)

	// AfterRun is called after inference completes (or fails).
	// Duration, Error, and OutputNames are populated.
	AfterRun(info *RunInfo)
}

// RunInfo contains information about an inference execution.
// Fields are progressively populated: InputNames is set before Run,
// Duration/Error/OutputNames are set after.
type RunInfo struct {
	InputNames  []string
	OutputNames []string
	Duration    time.Duration
	Error       error
}

// HookFunc adapts a simple function into a Hook.
// The function is called as AfterRun; BeforeRun is a no-op.
//
// Example:
//
//	pool, _ := NewSessionPool(runtime, env, modelData, 4, &PoolConfig{
//	    Hooks: []Hook{
//	        HookFunc(func(info *RunInfo) {
//	            log.Printf("inference took %v", info.Duration)
//	        }),
//	    },
//	})
type hookFunc struct {
	fn func(*RunInfo)
}

func (h *hookFunc) BeforeRun(_ *RunInfo)   {}
func (h *hookFunc) AfterRun(info *RunInfo) { h.fn(info) }

// AfterRunHook creates a Hook that calls fn after every inference.
// This is a convenience for the common case where you only need AfterRun.
func AfterRunHook(fn func(*RunInfo)) Hook {
	return &hookFunc{fn: fn}
}
