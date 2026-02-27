package onnxruntime

import (
	"context"
	"os"
	"testing"
)

func TestLoadModelFromFile(t *testing.T) {
	// Skip if no ORT library
	_ = newTestRuntime(t)

	model, err := LoadModelFromFile(testModelPath(), &ModelConfig{
		LibraryPath: libraryPath,
	})
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}
	defer model.Close()

	if len(model.InputNames()) == 0 {
		t.Error("Expected input names")
	}
	if len(model.OutputNames()) == 0 {
		t.Error("Expected output names")
	}

	// Run inference
	tensor, err := NewTensorValue(model.Runtime(), []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, []int64{1, 10})
	if err != nil {
		t.Fatalf("Failed to create tensor: %v", err)
	}
	defer tensor.Close()

	outputs, err := model.Run(context.Background(), map[string]*Value{"input": tensor})
	if err != nil {
		t.Fatalf("Failed to run inference: %v", err)
	}
	for _, v := range outputs {
		v.Close()
	}
}

func TestLoadModelFromBytes(t *testing.T) {
	_ = newTestRuntime(t)

	modelData, err := os.ReadFile(testModelPath())
	if err != nil {
		t.Fatalf("Failed to read model: %v", err)
	}

	model, err := LoadModelFromBytes(modelData, &ModelConfig{
		LibraryPath: libraryPath,
	})
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}
	defer model.Close()

	tensor, err := NewTensorValue(model.Runtime(), []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, []int64{1, 10})
	if err != nil {
		t.Fatalf("Failed to create tensor: %v", err)
	}
	defer tensor.Close()

	outputs, err := model.Run(context.Background(), map[string]*Value{"input": tensor})
	if err != nil {
		t.Fatalf("Failed to run inference: %v", err)
	}
	for _, v := range outputs {
		v.Close()
	}
}

func TestLoadModelWithOptions(t *testing.T) {
	_ = newTestRuntime(t)

	logLevel := LoggingLevelError
	model, err := LoadModelFromFile(testModelPath(), &ModelConfig{
		LibraryPath: libraryPath,
		SessionOptions: &SessionOptions{
			IntraOpNumThreads: 2,
			GraphOptimization: GraphOptimizationAll,
		},
		LogLevel: &logLevel,
	})
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}
	defer model.Close()

	tensor, err := NewTensorValue(model.Runtime(), []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, []int64{1, 10})
	if err != nil {
		t.Fatalf("Failed to create tensor: %v", err)
	}
	defer tensor.Close()

	outputs, err := model.Run(context.Background(), map[string]*Value{"input": tensor})
	if err != nil {
		t.Fatalf("Failed to run inference: %v", err)
	}
	for _, v := range outputs {
		v.Close()
	}
}

func TestModelSession(t *testing.T) {
	_ = newTestRuntime(t)

	model, err := LoadModelFromFile(testModelPath(), &ModelConfig{
		LibraryPath: libraryPath,
	})
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}
	defer model.Close()

	// Access underlying session for advanced ops
	session := model.Session()
	metadata, err := session.GetModelMetadata()
	if err != nil {
		t.Fatalf("Failed to get metadata: %v", err)
	}

	if metadata.ProducerName == "" {
		t.Error("Expected non-empty producer name")
	}
}

func TestModelDoubleClose(t *testing.T) {
	_ = newTestRuntime(t)

	model, err := LoadModelFromFile(testModelPath(), &ModelConfig{
		LibraryPath: libraryPath,
	})
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}

	model.Close()
	model.Close() // should not panic
}
