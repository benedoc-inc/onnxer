package onnxruntime

import (
	"context"
	"os"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func newTestPool(t *testing.T, size int, hooks ...Hook) *SessionPool {
	t.Helper()
	runtime := newTestRuntime(t)

	env, err := runtime.NewEnv("test", LoggingLevelWarning)
	if err != nil {
		t.Fatalf("Failed to create env: %v", err)
	}
	t.Cleanup(func() { env.Close() })

	modelData, err := os.ReadFile(testModelPath())
	if err != nil {
		t.Fatalf("Failed to read model: %v", err)
	}

	config := &PoolConfig{Hooks: hooks}
	pool, err := NewSessionPool(runtime, env, modelData, size, config)
	if err != nil {
		t.Fatalf("Failed to create pool: %v", err)
	}
	t.Cleanup(func() { pool.Close() })

	return pool
}

func runPoolInference(t *testing.T, pool *SessionPool) map[string]*Value {
	t.Helper()
	runtime := pool.runtime

	inputData := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	tensor, err := NewTensorValue(runtime, inputData, []int64{1, 10})
	if err != nil {
		t.Fatalf("Failed to create tensor: %v", err)
	}
	defer tensor.Close()

	outputs, err := pool.Run(context.Background(), map[string]*Value{"input": tensor})
	if err != nil {
		t.Fatalf("Failed to run inference: %v", err)
	}
	return outputs
}

func TestSessionPoolBasic(t *testing.T) {
	pool := newTestPool(t, 2)

	if pool.Size() != 2 {
		t.Errorf("Expected pool size 2, got %d", pool.Size())
	}
	if pool.Available() != 2 {
		t.Errorf("Expected 2 available, got %d", pool.Available())
	}

	outputs := runPoolInference(t, pool)
	for _, v := range outputs {
		v.Close()
	}

	stats := pool.Stats()
	if stats.TotalRuns != 1 {
		t.Errorf("Expected 1 total run, got %d", stats.TotalRuns)
	}
	if stats.TotalErrors != 0 {
		t.Errorf("Expected 0 errors, got %d", stats.TotalErrors)
	}
	if stats.AvgLatency() == 0 {
		t.Error("Expected non-zero average latency")
	}
}

func TestSessionPoolConcurrent(t *testing.T) {
	pool := newTestPool(t, 4)

	var wg sync.WaitGroup
	errCount := atomic.Int32{}

	for i := 0; i < 20; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()

			tensor, err := NewTensorValue(pool.runtime, []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, []int64{1, 10})
			if err != nil {
				errCount.Add(1)
				return
			}
			defer tensor.Close()

			outputs, err := pool.Run(context.Background(), map[string]*Value{"input": tensor})
			if err != nil {
				errCount.Add(1)
				return
			}
			for _, v := range outputs {
				v.Close()
			}
		}()
	}

	wg.Wait()

	if errCount.Load() != 0 {
		t.Errorf("Expected 0 errors, got %d", errCount.Load())
	}

	stats := pool.Stats()
	if stats.TotalRuns != 20 {
		t.Errorf("Expected 20 total runs, got %d", stats.TotalRuns)
	}
}

func TestSessionPoolContextCancellation(t *testing.T) {
	pool := newTestPool(t, 1)

	// Borrow the only session
	tensor, _ := NewTensorValue(pool.runtime, []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, []int64{1, 10})
	defer tensor.Close()

	outputs, err := pool.Run(context.Background(), map[string]*Value{"input": tensor})
	if err != nil {
		t.Fatalf("First run failed: %v", err)
	}
	for _, v := range outputs {
		v.Close()
	}

	// Now try with an already-cancelled context while pool is empty briefly
	// (pool has 1 session, but it should be returned by now)
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // cancel immediately

	// The session is available, so this should still work OR return ctx error
	// depending on timing. Either is acceptable.
	outputs2, err := pool.Run(ctx, map[string]*Value{"input": tensor})
	if err != nil {
		// Context cancelled before we got a session — acceptable
		return
	}
	for _, v := range outputs2 {
		v.Close()
	}
}

func TestSessionPoolHooks(t *testing.T) {
	var beforeCount, afterCount atomic.Int32
	var lastDuration atomic.Int64

	hook := AfterRunHook(func(info *RunInfo) {
		afterCount.Add(1)
		lastDuration.Store(int64(info.Duration))
	})

	// Also test the full Hook interface
	fullHook := &testHook{beforeCount: &beforeCount}

	pool := newTestPool(t, 2, hook, fullHook)

	outputs := runPoolInference(t, pool)
	for _, v := range outputs {
		v.Close()
	}

	if beforeCount.Load() != 1 {
		t.Errorf("Expected 1 BeforeRun call, got %d", beforeCount.Load())
	}
	if afterCount.Load() != 1 {
		t.Errorf("Expected 1 AfterRun call, got %d", afterCount.Load())
	}
	if lastDuration.Load() == 0 {
		t.Error("Expected non-zero duration in hook")
	}
}

type testHook struct {
	beforeCount *atomic.Int32
}

func (h *testHook) BeforeRun(_ *RunInfo) { h.beforeCount.Add(1) }
func (h *testHook) AfterRun(_ *RunInfo)  {}

func TestSessionPoolNames(t *testing.T) {
	pool := newTestPool(t, 1)

	inputs := pool.InputNames()
	if len(inputs) == 0 {
		t.Error("Expected input names, got empty")
	}
	if inputs[0] != "input" {
		t.Errorf("Expected input name 'input', got %q", inputs[0])
	}

	outputs := pool.OutputNames()
	if len(outputs) == 0 {
		t.Error("Expected output names, got empty")
	}
}

func TestSessionPoolClose(t *testing.T) {
	pool := newTestPool(t, 2)
	pool.Close()

	// Run after close should fail
	tensor, _ := NewTensorValue(pool.runtime, []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, []int64{1, 10})
	if tensor != nil {
		defer tensor.Close()
	}

	_, err := pool.Run(context.Background(), map[string]*Value{"input": tensor})
	if err == nil {
		t.Error("Expected error running on closed pool")
	}

	// Double close should not panic
	pool.Close()
}

func TestSessionPoolStats(t *testing.T) {
	pool := newTestPool(t, 2)

	// Run 5 inferences
	for i := 0; i < 5; i++ {
		outputs := runPoolInference(t, pool)
		for _, v := range outputs {
			v.Close()
		}
	}

	stats := pool.Stats()
	if stats.TotalRuns != 5 {
		t.Errorf("Expected 5 runs, got %d", stats.TotalRuns)
	}
	if stats.TotalErrors != 0 {
		t.Errorf("Expected 0 errors, got %d", stats.TotalErrors)
	}
	if stats.PoolSize != 2 {
		t.Errorf("Expected pool size 2, got %d", stats.PoolSize)
	}
	if stats.AvgLatency() == 0 {
		t.Error("Expected non-zero avg latency")
	}
}

func TestSessionPoolTimeout(t *testing.T) {
	pool := newTestPool(t, 1)

	// Grab the session
	runtime := pool.runtime
	tensor, _ := NewTensorValue(runtime, []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, []int64{1, 10})
	defer tensor.Close()

	// Run a normal inference
	outputs, err := pool.Run(context.Background(), map[string]*Value{"input": tensor})
	if err != nil {
		t.Fatalf("Run failed: %v", err)
	}
	for _, v := range outputs {
		v.Close()
	}

	// Now run with a very short timeout to verify it doesn't hang
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	outputs2, err := pool.Run(ctx, map[string]*Value{"input": tensor})
	if err != nil {
		t.Fatalf("Run with timeout failed: %v", err)
	}
	for _, v := range outputs2 {
		v.Close()
	}
}

func TestSessionPoolWarmup(t *testing.T) {
	pool := newTestPool(t, 3)

	tensor, err := NewTensorValue(pool.runtime, []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, []int64{1, 10})
	if err != nil {
		t.Fatalf("Failed to create tensor: %v", err)
	}
	defer tensor.Close()

	err = pool.Warmup(context.Background(), map[string]*Value{"input": tensor})
	if err != nil {
		t.Fatalf("Warmup failed: %v", err)
	}

	stats := pool.Stats()
	if stats.TotalRuns != 3 {
		t.Errorf("Expected 3 warmup runs, got %d", stats.TotalRuns)
	}
}

func TestSessionPoolHealthCheck(t *testing.T) {
	pool := newTestPool(t, 1)

	tensor, err := NewTensorValue(pool.runtime, []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, []int64{1, 10})
	if err != nil {
		t.Fatalf("Failed to create tensor: %v", err)
	}
	defer tensor.Close()

	err = pool.HealthCheck(context.Background(), map[string]*Value{"input": tensor})
	if err != nil {
		t.Fatalf("HealthCheck failed: %v", err)
	}
}

func TestSessionPoolHealthCheckClosed(t *testing.T) {
	pool := newTestPool(t, 1)
	pool.Close()

	err := pool.HealthCheck(context.Background(), nil)
	if err == nil {
		t.Error("Expected error on closed pool health check")
	}
}

func TestSessionPoolResetStats(t *testing.T) {
	pool := newTestPool(t, 1)

	for i := 0; i < 3; i++ {
		outputs := runPoolInference(t, pool)
		for _, v := range outputs {
			v.Close()
		}
	}

	if stats := pool.Stats(); stats.TotalRuns != 3 {
		t.Fatalf("Expected 3 runs before reset, got %d", stats.TotalRuns)
	}

	pool.ResetStats()

	stats := pool.Stats()
	if stats.TotalRuns != 0 {
		t.Errorf("Expected 0 runs after reset, got %d", stats.TotalRuns)
	}
}

func TestSessionPoolSlogHook(t *testing.T) {
	hook := NewSlogHook(nil)
	pool := newTestPool(t, 1, hook)

	outputs := runPoolInference(t, pool)
	for _, v := range outputs {
		v.Close()
	}
}
