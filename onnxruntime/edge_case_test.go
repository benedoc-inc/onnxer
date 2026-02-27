package onnxruntime

import (
	"context"
	"sync"
	"testing"
)

func TestSessionRunAfterClose(t *testing.T) {
	runtime := newTestRuntime(t)
	session := newTestSession(t, runtime)

	session.Close()

	_, err := session.Run(context.Background(), map[string]*Value{})
	if err != ErrSessionClosed {
		t.Errorf("Expected ErrSessionClosed, got %v", err)
	}
}

func TestSessionDoubleClose(t *testing.T) {
	runtime := newTestRuntime(t)
	session := newTestSession(t, runtime)

	session.Close()
	session.Close() // should not panic
}

func TestEnvDoubleClose(t *testing.T) {
	runtime := newTestRuntime(t)

	env, err := runtime.NewEnv("test", LoggingLevelWarning)
	if err != nil {
		t.Fatalf("Failed to create env: %v", err)
	}

	env.Close()
	env.Close() // should not panic
}

func TestIoBindingDoubleClose(t *testing.T) {
	runtime := newTestRuntime(t)
	session := newTestSession(t, runtime)

	binding, err := session.NewIoBinding()
	if err != nil {
		t.Fatalf("Failed to create IO binding: %v", err)
	}

	binding.Close()
	binding.Close() // should not panic
}

func TestValueDoubleClose(t *testing.T) {
	runtime := newTestRuntime(t)

	tensor, err := NewTensorValue(runtime, []float32{1, 2, 3}, []int64{3})
	if err != nil {
		t.Fatalf("Failed to create tensor: %v", err)
	}

	tensor.Close()
	tensor.Close() // should not panic
}

func TestNewTensorValueEmptyData(t *testing.T) {
	runtime := newTestRuntime(t)

	_, err := NewTensorValue(runtime, []float32{}, []int64{0})
	if err == nil {
		t.Error("Expected error for empty data, got nil")
	}
}

func TestGetTensorDataTypeMismatch(t *testing.T) {
	runtime := newTestRuntime(t)

	tensor, err := NewTensorValue(runtime, []float32{1.0, 2.0}, []int64{2})
	if err != nil {
		t.Fatalf("Failed to create tensor: %v", err)
	}
	defer tensor.Close()

	// Try to read float32 tensor as int32 — should error
	_, _, err = GetTensorData[int32](tensor)
	if err == nil {
		t.Error("Expected type mismatch error, got nil")
	}
}

func TestGetTensorDataUnsafe(t *testing.T) {
	runtime := newTestRuntime(t)

	data := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
	tensor, err := NewTensorValue(runtime, data, []int64{2, 3})
	if err != nil {
		t.Fatalf("Failed to create tensor: %v", err)
	}
	defer tensor.Close()

	result, shape, err := GetTensorDataUnsafe[float32](tensor)
	if err != nil {
		t.Fatalf("Failed to get tensor data: %v", err)
	}

	if len(result) != 6 {
		t.Fatalf("Expected 6 elements, got %d", len(result))
	}

	if shape[0] != 2 || shape[1] != 3 {
		t.Errorf("Expected shape [2 3], got %v", shape)
	}

	for i, v := range data {
		if result[i] != v {
			t.Errorf("Element %d: expected %f, got %f", i, v, result[i])
		}
	}
}

func TestGetTensorDataUnsafeTypeMismatch(t *testing.T) {
	runtime := newTestRuntime(t)

	tensor, err := NewTensorValue(runtime, []float32{1.0}, []int64{1})
	if err != nil {
		t.Fatalf("Failed to create tensor: %v", err)
	}
	defer tensor.Close()

	_, _, err = GetTensorDataUnsafe[int32](tensor)
	if err == nil {
		t.Error("Expected type mismatch error, got nil")
	}
}

func TestConcurrentSessionRun(t *testing.T) {
	runtime := newTestRuntime(t)

	// Create separate sessions for concurrent use
	sessions := make([]*Session, 4)
	for i := range sessions {
		sessions[i] = newTestSession(t, runtime)
	}

	inputData := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	inputShape := []int64{1, 10}

	var wg sync.WaitGroup
	errors := make([]error, len(sessions))

	for i, session := range sessions {
		wg.Add(1)
		go func(idx int, s *Session) {
			defer wg.Done()

			for j := 0; j < 10; j++ {
				tensor, err := NewTensorValue(runtime, inputData, inputShape)
				if err != nil {
					errors[idx] = err
					return
				}

				outputs, err := s.Run(context.Background(), map[string]*Value{"input": tensor})
				tensor.Close()
				if err != nil {
					errors[idx] = err
					return
				}
				for _, v := range outputs {
					v.Close()
				}
			}
		}(i, session)
	}

	wg.Wait()

	for i, err := range errors {
		if err != nil {
			t.Errorf("Session %d errored: %v", i, err)
		}
	}
}

func TestProviderFallback(t *testing.T) {
	runtime := newTestRuntime(t)

	env, err := runtime.NewEnv("test", LoggingLevelWarning)
	if err != nil {
		t.Fatalf("Failed to create env: %v", err)
	}
	defer env.Close()

	modelFile := mustOpenModel(t)
	defer modelFile.Close()

	// Request a non-existent provider — should fall back to CPU
	session, provider, err := runtime.NewSessionWithProviderFallback(env, modelFile, nil,
		ExecutionProvider{Name: "NonExistentProvider"},
	)
	if err != nil {
		t.Fatalf("Failed to create session with fallback: %v", err)
	}
	defer session.Close()

	if provider != "CPUExecutionProvider" {
		t.Errorf("Expected CPUExecutionProvider fallback, got %s", provider)
	}
}

func TestStringMethods(t *testing.T) {
	tests := []struct {
		name     string
		value    string
		expected string
	}{
		{"GraphOptimizationDisabled", GraphOptimizationDisabled.String(), "Disabled"},
		{"GraphOptimizationBasic", GraphOptimizationBasic.String(), "Basic"},
		{"GraphOptimizationExtended", GraphOptimizationExtended.String(), "Extended"},
		{"GraphOptimizationAll", GraphOptimizationAll.String(), "All"},
		{"ExecutionModeSequential", ExecutionModeSequential.String(), "Sequential"},
		{"ExecutionModeParallel", ExecutionModeParallel.String(), "Parallel"},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if tc.value != tc.expected {
				t.Errorf("Expected %q, got %q", tc.expected, tc.value)
			}
		})
	}
}

func TestErrorCodeName(t *testing.T) {
	err := &RuntimeError{Code: ErrorCodeInvalidArgument, Message: "test error"}
	expected := "onnxruntime error (InvalidArgument): test error"
	if err.Error() != expected {
		t.Errorf("Expected %q, got %q", expected, err.Error())
	}
}
