package onnxruntime

import (
	"context"
	"os"
	"testing"
)

func TestPrepackedWeightsContainer(t *testing.T) {
	runtime := newTestRuntime(t)

	container, err := runtime.NewPrepackedWeightsContainer()
	if err != nil {
		t.Fatalf("Failed to create prepacked weights container: %v", err)
	}
	defer container.Close()

	// Double close should not panic
	container.Close()
	container.Close()
}

func TestSessionPoolWithPrepackedWeights(t *testing.T) {
	runtime := newTestRuntime(t)

	env, err := runtime.NewEnv("test", LoggingLevelWarning)
	if err != nil {
		t.Fatalf("Failed to create env: %v", err)
	}
	defer env.Close()

	modelData, err := os.ReadFile(testModelPath())
	if err != nil {
		t.Fatalf("Failed to read model: %v", err)
	}

	pool, err := NewSessionPool(runtime, env, modelData, 4, &PoolConfig{
		SharePrepackedWeights: true,
	})
	if err != nil {
		t.Fatalf("Failed to create pool with prepacked weights: %v", err)
	}
	defer pool.Close()

	if pool.Size() != 4 {
		t.Errorf("Expected pool size 4, got %d", pool.Size())
	}

	// Run inference to verify sessions work correctly
	tensor, err := NewTensorValue(runtime, []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, []int64{1, 10})
	if err != nil {
		t.Fatalf("Failed to create tensor: %v", err)
	}
	defer tensor.Close()

	outputs, err := pool.Run(context.Background(), map[string]*Value{"input": tensor})
	if err != nil {
		t.Fatalf("Run with prepacked weights failed: %v", err)
	}
	for _, v := range outputs {
		v.Close()
	}

	stats := pool.Stats()
	if stats.TotalRuns != 1 {
		t.Errorf("Expected 1 run, got %d", stats.TotalRuns)
	}
}

func TestSessionPoolFromFileWithPrepackedWeights(t *testing.T) {
	runtime := newTestRuntime(t)

	env, err := runtime.NewEnv("test", LoggingLevelWarning)
	if err != nil {
		t.Fatalf("Failed to create env: %v", err)
	}
	defer env.Close()

	pool, err := NewSessionPoolFromFile(runtime, env, testModelPath(), 3, &PoolConfig{
		SharePrepackedWeights: true,
	})
	if err != nil {
		t.Fatalf("Failed to create file pool with prepacked weights: %v", err)
	}
	defer pool.Close()

	tensor, err := NewTensorValue(runtime, []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, []int64{1, 10})
	if err != nil {
		t.Fatalf("Failed to create tensor: %v", err)
	}
	defer tensor.Close()

	outputs, err := pool.Run(context.Background(), map[string]*Value{"input": tensor})
	if err != nil {
		t.Fatalf("Run with prepacked weights (file) failed: %v", err)
	}
	for _, v := range outputs {
		v.Close()
	}
}
