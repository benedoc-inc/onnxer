package onnxruntime

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"os"
)

// Model wraps Runtime + Env + Session into a single object for simple use cases.
// It provides a one-call path from model data to running inference.
//
// For advanced use (multiple models, custom environments, session pooling),
// use Runtime, Env, and Session directly.
//
// Example:
//
//	model, err := onnxruntime.LoadModelFromFile("model.onnx", nil)
//	if err != nil {
//	    log.Fatal(err)
//	}
//	defer model.Close()
//
//	outputs, err := model.Run(ctx, map[string]*Value{"input": tensor})
type Model struct {
	runtime *Runtime
	env     *Env
	session *Session
}

// ModelConfig configures model loading.
type ModelConfig struct {
	// LibraryPath overrides the ONNX Runtime shared library path.
	// If empty, searches standard system paths.
	LibraryPath string

	// APIVersion specifies which ORT API version to use (default: 23).
	APIVersion uint32

	// SessionOptions configures the inference session.
	SessionOptions *SessionOptions

	// LogLevel sets the ORT logging level (default: LoggingLevelWarning).
	LogLevel *LoggingLevel
}

func (c *ModelConfig) apiVersion() uint32 {
	if c != nil && c.APIVersion != 0 {
		return c.APIVersion
	}
	return 23
}

func (c *ModelConfig) libraryPath() string {
	if c != nil {
		return c.LibraryPath
	}
	return ""
}

func (c *ModelConfig) logLevel() LoggingLevel {
	if c != nil && c.LogLevel != nil {
		return *c.LogLevel
	}
	return LoggingLevelWarning
}

func (c *ModelConfig) sessionOptions() *SessionOptions {
	if c != nil {
		return c.SessionOptions
	}
	return nil
}

// LoadModel creates a Model from model data read from an io.Reader.
func LoadModel(modelReader io.Reader, config *ModelConfig) (*Model, error) {
	rt, err := NewRuntime(config.libraryPath(), config.apiVersion())
	if err != nil {
		return nil, fmt.Errorf("failed to create runtime: %w", err)
	}

	env, err := rt.NewEnv("onnxer", config.logLevel())
	if err != nil {
		rt.Close()
		return nil, fmt.Errorf("failed to create environment: %w", err)
	}

	session, err := rt.NewSessionFromReader(env, modelReader, config.sessionOptions())
	if err != nil {
		env.Close()
		rt.Close()
		return nil, fmt.Errorf("failed to create session: %w", err)
	}

	return &Model{
		runtime: rt,
		env:     env,
		session: session,
	}, nil
}

// LoadModelFromFile creates a Model from a model file path.
func LoadModelFromFile(path string, config *ModelConfig) (*Model, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open model file: %w", err)
	}
	defer f.Close()

	return LoadModel(f, config)
}

// LoadModelFromBytes creates a Model from model data in memory.
func LoadModelFromBytes(data []byte, config *ModelConfig) (*Model, error) {
	return LoadModel(bytes.NewReader(data), config)
}

// Run executes inference with the model.
func (m *Model) Run(ctx context.Context, inputs map[string]*Value, opts ...RunOption) (map[string]*Value, error) {
	return m.session.Run(ctx, inputs, opts...)
}

// Session returns the underlying Session for advanced operations
// like GetModelMetadata, GetInputInfo, or IoBinding.
func (m *Model) Session() *Session {
	return m.session
}

// Runtime returns the underlying Runtime.
func (m *Model) Runtime() *Runtime {
	return m.runtime
}

// InputNames returns the model's input names.
func (m *Model) InputNames() []string {
	return m.session.InputNames()
}

// OutputNames returns the model's output names.
func (m *Model) OutputNames() []string {
	return m.session.OutputNames()
}

// Close releases all resources (session, environment, and runtime).
// It is safe to call Close multiple times.
func (m *Model) Close() {
	if m.session != nil {
		m.session.Close()
		m.session = nil
	}
	if m.env != nil {
		m.env.Close()
		m.env = nil
	}
	if m.runtime != nil {
		m.runtime.Close()
		m.runtime = nil
	}
}
