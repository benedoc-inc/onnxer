package onnxruntime

import (
	"bytes"
	"os"
	"testing"
)

func newSessionWithOptions(t *testing.T, runtime *Runtime, opts *SessionOptions) *Session {
	t.Helper()

	env, err := runtime.NewEnv("test", LoggingLevelWarning)
	if err != nil {
		t.Fatalf("Failed to create environment: %v", err)
	}
	t.Cleanup(func() { env.Close() })

	modelData, err := os.ReadFile(testModelPath())
	if err != nil {
		t.Fatalf("Failed to read model file: %v", err)
	}

	session, err := runtime.NewSessionFromReader(env, bytes.NewReader(modelData), opts)
	if err != nil {
		t.Fatalf("Failed to create session with options: %v", err)
	}
	t.Cleanup(func() { session.Close() })

	return session
}

func TestSessionOptionsGraphOptimization(t *testing.T) {
	runtime := newTestRuntime(t)

	levels := []GraphOptimizationLevel{
		GraphOptimizationDisabled,
		GraphOptimizationBasic,
		GraphOptimizationExtended,
		GraphOptimizationAll,
	}

	for _, level := range levels {
		session := newSessionWithOptions(t, runtime, &SessionOptions{
			GraphOptimization: level,
		})
		// Verify session works
		runInference(t, runtime, session)
	}
}

func TestSessionOptionsInterOpNumThreads(t *testing.T) {
	runtime := newTestRuntime(t)

	session := newSessionWithOptions(t, runtime, &SessionOptions{
		InterOpNumThreads: 2,
	})
	runInference(t, runtime, session)
}

func TestSessionOptionsExecutionMode(t *testing.T) {
	runtime := newTestRuntime(t)

	for _, mode := range []ExecutionMode{ExecutionModeSequential, ExecutionModeParallel} {
		session := newSessionWithOptions(t, runtime, &SessionOptions{
			ExecutionMode: mode,
		})
		runInference(t, runtime, session)
	}
}

func TestSessionOptionsCpuMemArena(t *testing.T) {
	runtime := newTestRuntime(t)

	enabled := true
	disabled := false

	for _, arena := range []*bool{&enabled, &disabled, nil} {
		session := newSessionWithOptions(t, runtime, &SessionOptions{
			CpuMemArena: arena,
		})
		runInference(t, runtime, session)
	}
}

func TestSessionOptionsMemPattern(t *testing.T) {
	runtime := newTestRuntime(t)

	enabled := true
	disabled := false

	for _, pattern := range []*bool{&enabled, &disabled, nil} {
		session := newSessionWithOptions(t, runtime, &SessionOptions{
			MemPattern: pattern,
		})
		runInference(t, runtime, session)
	}
}

func TestSessionOptionsLogSeverityLevel(t *testing.T) {
	runtime := newTestRuntime(t)

	level := LoggingLevelError
	session := newSessionWithOptions(t, runtime, &SessionOptions{
		LogSeverityLevel: &level,
	})
	runInference(t, runtime, session)
}

func TestSessionOptionsConfigEntries(t *testing.T) {
	runtime := newTestRuntime(t)

	session := newSessionWithOptions(t, runtime, &SessionOptions{
		ConfigEntries: map[string]string{
			"session.use_env_allocators": "0",
		},
	})
	runInference(t, runtime, session)
}

func TestSessionOptionsCombined(t *testing.T) {
	runtime := newTestRuntime(t)

	enabled := true
	level := LoggingLevelWarning

	session := newSessionWithOptions(t, runtime, &SessionOptions{
		IntraOpNumThreads: 2,
		InterOpNumThreads: 1,
		GraphOptimization: GraphOptimizationAll,
		ExecutionMode:     ExecutionModeSequential,
		CpuMemArena:       &enabled,
		MemPattern:        &enabled,
		LogSeverityLevel:  &level,
	})
	runInference(t, runtime, session)
}

func TestSessionOptionsDeterministicCompute(t *testing.T) {
	runtime := newTestRuntime(t)

	enabled := true
	session := newSessionWithOptions(t, runtime, &SessionOptions{
		DeterministicCompute: &enabled,
	})
	runInference(t, runtime, session)
}

func TestSessionOptionsFreeDimensionOverrides(t *testing.T) {
	runtime := newTestRuntime(t)

	// Our test model may have dynamic batch dim — try overriding it
	session := newSessionWithOptions(t, runtime, &SessionOptions{
		FreeDimensionOverrides: map[string]int64{
			"batch_size": 1,
		},
	})
	runInference(t, runtime, session)
}

func TestWithRunTag(t *testing.T) {
	runtime := newTestRuntime(t)
	session := newTestSession(t, runtime)

	inputData := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	tensor, err := NewTensorValue(runtime, inputData, []int64{1, 10})
	if err != nil {
		t.Fatalf("Failed to create tensor: %v", err)
	}
	defer tensor.Close()

	outputs, err := session.Run(t.Context(), map[string]*Value{
		"input": tensor,
	}, WithRunTag("test-inference-001"))
	if err != nil {
		t.Fatalf("Failed to run with run tag: %v", err)
	}
	for _, v := range outputs {
		v.Close()
	}
}

func TestHasValue(t *testing.T) {
	runtime := newTestRuntime(t)

	tensor, err := NewTensorValue(runtime, []float32{1, 2, 3}, []int64{3})
	if err != nil {
		t.Fatalf("Failed to create tensor: %v", err)
	}
	defer tensor.Close()

	has, err := tensor.HasValue()
	if err != nil {
		t.Fatalf("HasValue failed: %v", err)
	}
	if !has {
		t.Error("Expected tensor to have a value")
	}
}

// runInference runs a basic inference to verify the session works
func runInference(t *testing.T, runtime *Runtime, session *Session) {
	t.Helper()

	inputData := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	tensor, err := NewTensorValue(runtime, inputData, []int64{1, 10})
	if err != nil {
		t.Fatalf("Failed to create tensor: %v", err)
	}
	defer tensor.Close()

	outputs, err := session.Run(t.Context(), map[string]*Value{
		"input": tensor,
	})
	if err != nil {
		t.Fatalf("Failed to run inference: %v", err)
	}
	for _, v := range outputs {
		v.Close()
	}
}
