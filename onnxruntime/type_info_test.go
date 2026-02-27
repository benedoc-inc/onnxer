package onnxruntime

import (
	"slices"
	"testing"
)

func TestGetInputInfo(t *testing.T) {
	runtime := newTestRuntime(t)
	session := newTestSession(t, runtime)

	infos, err := session.GetInputInfo()
	if err != nil {
		t.Fatalf("Failed to get input info: %v", err)
	}

	// Test model has 1 input named "input"
	if len(infos) != 1 {
		t.Fatalf("Expected 1 input, got %d", len(infos))
	}

	info := infos[0]
	if info.Name != "input" {
		t.Errorf("Expected input name 'input', got %q", info.Name)
	}
	if info.Type != ONNXTypeTensor {
		t.Errorf("Expected tensor type, got %d", info.Type)
	}
	if info.TensorInfo == nil {
		t.Fatal("TensorInfo should not be nil for tensor input")
	}
	if info.TensorInfo.ElementType != ONNXTensorElementDataTypeFloat {
		t.Errorf("Expected float32 element type, got %d", info.TensorInfo.ElementType)
	}

	// Shape should be [batch_size, 10] — batch_size may be -1 (dynamic)
	shape := info.TensorInfo.Shape
	if len(shape) != 2 {
		t.Fatalf("Expected 2D shape, got %v", shape)
	}
	if shape[1] != 10 {
		t.Errorf("Expected second dim = 10, got %d", shape[1])
	}
}

func TestGetOutputInfo(t *testing.T) {
	runtime := newTestRuntime(t)
	session := newTestSession(t, runtime)

	infos, err := session.GetOutputInfo()
	if err != nil {
		t.Fatalf("Failed to get output info: %v", err)
	}

	// Test model has 1 output named "logits"
	if len(infos) != 1 {
		t.Fatalf("Expected 1 output, got %d", len(infos))
	}

	info := infos[0]
	if info.Name != "logits" {
		t.Errorf("Expected output name 'logits', got %q", info.Name)
	}
	if info.Type != ONNXTypeTensor {
		t.Errorf("Expected tensor type, got %d", info.Type)
	}
	if info.TensorInfo == nil {
		t.Fatal("TensorInfo should not be nil for tensor output")
	}
	if info.TensorInfo.ElementType != ONNXTensorElementDataTypeFloat {
		t.Errorf("Expected float32 element type, got %d", info.TensorInfo.ElementType)
	}

	// Shape should be [batch_size, 3]
	shape := info.TensorInfo.Shape
	if len(shape) != 2 {
		t.Fatalf("Expected 2D shape, got %v", shape)
	}
	if shape[1] != 3 {
		t.Errorf("Expected second dim = 3, got %d", shape[1])
	}
}

func TestGetInputOutputInfoConsistency(t *testing.T) {
	runtime := newTestRuntime(t)
	session := newTestSession(t, runtime)

	inputInfos, err := session.GetInputInfo()
	if err != nil {
		t.Fatalf("Failed to get input info: %v", err)
	}

	outputInfos, err := session.GetOutputInfo()
	if err != nil {
		t.Fatalf("Failed to get output info: %v", err)
	}

	// Input names should match session.InputNames()
	inputNames := make([]string, len(inputInfos))
	for i, info := range inputInfos {
		inputNames[i] = info.Name
	}
	if !slices.Equal(inputNames, session.InputNames()) {
		t.Errorf("Input names mismatch: info=%v, session=%v", inputNames, session.InputNames())
	}

	// Output names should match session.OutputNames()
	outputNames := make([]string, len(outputInfos))
	for i, info := range outputInfos {
		outputNames[i] = info.Name
	}
	if !slices.Equal(outputNames, session.OutputNames()) {
		t.Errorf("Output names mismatch: info=%v, session=%v", outputNames, session.OutputNames())
	}
}

func TestSymbolicDimensions(t *testing.T) {
	runtime := newTestRuntime(t)
	session := newTestSession(t, runtime)

	infos, err := session.GetInputInfo()
	if err != nil {
		t.Fatalf("Failed to get input info: %v", err)
	}

	info := infos[0]
	if info.TensorInfo == nil {
		t.Fatal("TensorInfo should not be nil")
	}

	// SymbolicDimNames should be populated (same length as Shape)
	if len(info.TensorInfo.SymbolicDimNames) != len(info.TensorInfo.Shape) {
		t.Errorf("SymbolicDimNames length %d != Shape length %d",
			len(info.TensorInfo.SymbolicDimNames), len(info.TensorInfo.Shape))
	}
}

func TestGetInputInfoClosedSession(t *testing.T) {
	runtime := newTestRuntime(t)
	session := newTestSession(t, runtime)
	session.Close()

	_, err := session.GetInputInfo()
	if err == nil {
		t.Error("Expected error for closed session")
	}
}
