package onnxruntime

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestProfilingEnabled(t *testing.T) {
	runtime := newTestRuntime(t)

	env, err := runtime.NewEnv("test", LoggingLevelWarning)
	if err != nil {
		t.Fatalf("Failed to create env: %v", err)
	}
	defer env.Close()

	profileDir := t.TempDir()
	profilePrefix := filepath.Join(profileDir, "ort_profile")

	f := mustOpenModel(t)
	defer f.Close()

	session, err := runtime.NewSessionFromReader(env, f, &SessionOptions{
		ProfilingOutputPath: profilePrefix,
	})
	if err != nil {
		t.Fatalf("Failed to create session with profiling: %v", err)
	}
	defer session.Close()

	// Run inference to generate profiling data
	tensor, err := NewTensorValue(runtime, []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, []int64{1, 10})
	if err != nil {
		t.Fatalf("Failed to create tensor: %v", err)
	}
	defer tensor.Close()

	outputs, err := session.Run(context.Background(), map[string]*Value{"input": tensor})
	if err != nil {
		t.Fatalf("Failed to run inference: %v", err)
	}
	for _, v := range outputs {
		v.Close()
	}

	// End profiling and check output
	profilePath, err := session.EndProfiling()
	if err != nil {
		t.Fatalf("Failed to end profiling: %v", err)
	}

	if profilePath == "" {
		t.Error("Expected non-empty profile path")
	}

	// Verify the file exists
	if _, err := os.Stat(profilePath); os.IsNotExist(err) {
		t.Errorf("Profile file does not exist: %s", profilePath)
	}
}

func TestProfilingStartTimeNs(t *testing.T) {
	runtime := newTestRuntime(t)

	env, err := runtime.NewEnv("test", LoggingLevelWarning)
	if err != nil {
		t.Fatalf("Failed to create env: %v", err)
	}
	defer env.Close()

	profileDir := t.TempDir()
	profilePrefix := filepath.Join(profileDir, "ort_profile")

	f := mustOpenModel(t)
	defer f.Close()

	session, err := runtime.NewSessionFromReader(env, f, &SessionOptions{
		ProfilingOutputPath: profilePrefix,
	})
	if err != nil {
		t.Fatalf("Failed to create session: %v", err)
	}
	defer session.Close()

	startTime, err := session.ProfilingStartTimeNs()
	if err != nil {
		t.Fatalf("Failed to get profiling start time: %v", err)
	}

	if startTime == 0 {
		t.Error("Expected non-zero profiling start time")
	}
}

func TestOptimizedModelFilePath(t *testing.T) {
	runtime := newTestRuntime(t)

	env, err := runtime.NewEnv("test", LoggingLevelWarning)
	if err != nil {
		t.Fatalf("Failed to create env: %v", err)
	}
	defer env.Close()

	optimizedPath := filepath.Join(t.TempDir(), "optimized_model.onnx")

	f := mustOpenModel(t)
	defer f.Close()

	session, err := runtime.NewSessionFromReader(env, f, &SessionOptions{
		GraphOptimization:      GraphOptimizationAll,
		OptimizedModelFilePath: optimizedPath,
	})
	if err != nil {
		t.Fatalf("Failed to create session: %v", err)
	}
	defer session.Close()

	// Run inference to trigger optimization
	tensor, err := NewTensorValue(runtime, []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, []int64{1, 10})
	if err != nil {
		t.Fatalf("Failed to create tensor: %v", err)
	}
	defer tensor.Close()

	outputs, err := session.Run(context.Background(), map[string]*Value{"input": tensor})
	if err != nil {
		t.Fatalf("Failed to run inference: %v", err)
	}
	for _, v := range outputs {
		v.Close()
	}

	// Verify the optimized model was written
	info, err := os.Stat(optimizedPath)
	if os.IsNotExist(err) {
		t.Errorf("Optimized model file was not created: %s", optimizedPath)
	} else if err != nil {
		t.Fatalf("Failed to stat optimized model: %v", err)
	} else if info.Size() == 0 {
		t.Error("Optimized model file is empty")
	}
}

func TestBuildInfo(t *testing.T) {
	runtime := newTestRuntime(t)

	info := runtime.GetBuildInfo()
	if info == "" {
		t.Error("Expected non-empty build info")
	}

	// Build info should contain something identifiable
	if !strings.Contains(info, "ORT") && !strings.Contains(info, "onnx") && !strings.Contains(info, "Build") && len(info) < 5 {
		t.Errorf("Build info looks unexpected: %q", info)
	}
}

func TestTelemetry(t *testing.T) {
	runtime := newTestRuntime(t)

	env, err := runtime.NewEnv("test", LoggingLevelWarning)
	if err != nil {
		t.Fatalf("Failed to create env: %v", err)
	}
	defer env.Close()

	// Should not error
	if err := env.DisableTelemetry(); err != nil {
		t.Fatalf("Failed to disable telemetry: %v", err)
	}

	if err := env.EnableTelemetry(); err != nil {
		t.Fatalf("Failed to enable telemetry: %v", err)
	}
}
