package onnxruntime

import (
	"context"
	"testing"
)

func TestSessionRunWithCancellableContext(t *testing.T) {
	runtime := newTestRuntime(t)
	session := newTestSession(t, runtime)

	inputData := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	tensor, err := NewTensorValue(runtime, inputData, []int64{1, 10})
	if err != nil {
		t.Fatalf("Failed to create tensor: %v", err)
	}
	defer tensor.Close()

	// Run with a cancellable context that is NOT cancelled — should succeed
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	outputs, err := session.Run(ctx, map[string]*Value{
		"input": tensor,
	})
	if err != nil {
		t.Fatalf("Failed to run with cancellable context: %v", err)
	}
	for _, v := range outputs {
		v.Close()
	}
}

func TestSessionRunWithPreCancelledContext(t *testing.T) {
	runtime := newTestRuntime(t)
	session := newTestSession(t, runtime)

	inputData := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	tensor, err := NewTensorValue(runtime, inputData, []int64{1, 10})
	if err != nil {
		t.Fatalf("Failed to create tensor: %v", err)
	}
	defer tensor.Close()

	// Cancel before running — the model may be too fast to actually be cancelled,
	// but the run options machinery should not panic or leak
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	outputs, err := session.Run(ctx, map[string]*Value{
		"input": tensor,
	})
	if err != nil {
		// Expected: ORT may return an error due to termination
		t.Logf("Pre-cancelled run returned error (expected): %v", err)
		return
	}

	// If the model ran too fast to be cancelled, that's fine too
	for _, v := range outputs {
		v.Close()
	}
}

func TestSessionRunWithBackgroundContext(t *testing.T) {
	runtime := newTestRuntime(t)
	session := newTestSession(t, runtime)

	inputData := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	tensor, err := NewTensorValue(runtime, inputData, []int64{1, 10})
	if err != nil {
		t.Fatalf("Failed to create tensor: %v", err)
	}
	defer tensor.Close()

	// Background context (no Done channel) — should use nil run options
	outputs, err := session.Run(context.Background(), map[string]*Value{
		"input": tensor,
	})
	if err != nil {
		t.Fatalf("Failed to run with background context: %v", err)
	}
	for _, v := range outputs {
		v.Close()
	}
}
