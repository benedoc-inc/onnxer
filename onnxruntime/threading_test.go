package onnxruntime

import (
	"bytes"
	"context"
	"os"
	"testing"
)

func TestThreadingOptions(t *testing.T) {
	runtime := newTestRuntime(t)

	opts, err := runtime.NewThreadingOptions()
	if err != nil {
		t.Fatalf("Failed to create threading options: %v", err)
	}
	defer opts.Close()

	if err := opts.SetIntraOpNumThreads(4); err != nil {
		t.Fatalf("Failed to set intra-op threads: %v", err)
	}
	if err := opts.SetInterOpNumThreads(2); err != nil {
		t.Fatalf("Failed to set inter-op threads: %v", err)
	}
	if err := opts.SetSpinControl(false); err != nil {
		t.Fatalf("Failed to set spin control: %v", err)
	}

	// Double close should not panic
	opts.Close()
	opts.Close()
}

func TestEnvWithGlobalThreadPools(t *testing.T) {
	runtime := newTestRuntime(t)

	threadOpts, err := runtime.NewThreadingOptions()
	if err != nil {
		t.Fatalf("Failed to create threading options: %v", err)
	}
	defer threadOpts.Close()

	if err := threadOpts.SetIntraOpNumThreads(2); err != nil {
		t.Fatalf("Failed to set intra-op threads: %v", err)
	}
	if err := threadOpts.SetInterOpNumThreads(1); err != nil {
		t.Fatalf("Failed to set inter-op threads: %v", err)
	}

	env, err := runtime.NewEnvWithGlobalThreadPools("test-global-threads", LoggingLevelWarning, threadOpts)
	if err != nil {
		t.Fatalf("Failed to create env with global thread pools: %v", err)
	}
	defer env.Close()

	// Create a session that uses the global thread pool
	modelData, err := os.ReadFile(testModelPath())
	if err != nil {
		t.Fatalf("Failed to read model: %v", err)
	}

	session, err := runtime.NewSessionFromReader(env, bytes.NewReader(modelData), &SessionOptions{
		DisablePerSessionThreads: true,
		GraphOptimization:        GraphOptimizationAll,
	})
	if err != nil {
		t.Fatalf("Failed to create session with global threads: %v", err)
	}
	defer session.Close()

	// Run inference to verify it works
	tensor, err := NewTensorValue(runtime, []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, []int64{1, 10})
	if err != nil {
		t.Fatalf("Failed to create tensor: %v", err)
	}
	defer tensor.Close()

	outputs, err := session.Run(context.Background(), map[string]*Value{"input": tensor})
	if err != nil {
		t.Fatalf("Run with global thread pool failed: %v", err)
	}
	for _, v := range outputs {
		v.Close()
	}
}

func TestSessionOptionsDisablePerSessionThreads(t *testing.T) {
	runtime := newTestRuntime(t)

	session := newSessionWithOptions(t, runtime, &SessionOptions{
		DisablePerSessionThreads: true,
	})
	runInference(t, runtime, session)
}
