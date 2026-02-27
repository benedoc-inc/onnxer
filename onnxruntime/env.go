package onnxruntime

import (
	"fmt"
	"runtime"

	"github.com/benedoc-inc/onnxer/onnxruntime/internal/api"
)

// Env represents an ONNX Runtime environment that manages global state
// and configuration for all inference sessions.
type Env struct {
	ptr     api.OrtEnv
	runtime *Runtime
}

// NewEnv creates a new ONNX Runtime environment with the specified logging level and identifier.
// The logLevel parameter controls logging verbosity, and logID is used to tag log messages.
func (r *Runtime) NewEnv(logID string, logLevel LoggingLevel) (*Env, error) {
	logIDBytes := append([]byte(logID), 0)
	var envPtr api.OrtEnv

	status := r.apiFuncs.CreateEnv(logLevel, &logIDBytes[0], &envPtr)
	if err := r.statusError(status); err != nil {
		return nil, fmt.Errorf("failed to create environment: %w", err)
	}

	env := &Env{
		ptr:     envPtr,
		runtime: r,
	}
	runtime.AddCleanup(env, func(_ struct{}) { env.Close() }, struct{}{})
	return env, nil
}

// EnableTelemetry enables telemetry event collection for this environment.
// Telemetry helps the ONNX Runtime team understand usage patterns.
// It is enabled by default; call DisableTelemetry to opt out.
func (e *Env) EnableTelemetry() error {
	status := e.runtime.apiFuncs.EnableTelemetryEvents(e.ptr)
	if err := e.runtime.statusError(status); err != nil {
		return fmt.Errorf("failed to enable telemetry: %w", err)
	}
	return nil
}

// DisableTelemetry disables telemetry event collection for this environment.
func (e *Env) DisableTelemetry() error {
	status := e.runtime.apiFuncs.DisableTelemetryEvents(e.ptr)
	if err := e.runtime.statusError(status); err != nil {
		return fmt.Errorf("failed to disable telemetry: %w", err)
	}
	return nil
}

// Close releases the environment and frees associated resources.
func (e *Env) Close() {
	if e.ptr != 0 && e.runtime != nil && e.runtime.apiFuncs != nil {
		e.runtime.apiFuncs.ReleaseEnv(e.ptr)
		e.ptr = 0
	}
}
