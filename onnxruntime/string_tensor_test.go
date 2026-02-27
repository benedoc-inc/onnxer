package onnxruntime

import (
	"slices"
	"testing"
)

func TestNewStringTensorValue(t *testing.T) {
	runtime := newTestRuntime(t)

	data := []string{"hello", "world", "foo"}
	shape := []int64{3}

	tensor, err := runtime.NewStringTensorValue(data, shape)
	if err != nil {
		t.Fatalf("Failed to create string tensor: %v", err)
	}
	defer tensor.Close()

	// Verify it's a tensor
	isTensor, err := tensor.IsTensor()
	if err != nil {
		t.Fatalf("Failed to check IsTensor: %v", err)
	}
	if !isTensor {
		t.Error("Expected IsTensor() to return true")
	}

	// Verify element type
	elemType, err := tensor.GetTensorElementType()
	if err != nil {
		t.Fatalf("Failed to get element type: %v", err)
	}
	if elemType != ONNXTensorElementDataTypeString {
		t.Errorf("Expected string type (%d), got %d", ONNXTensorElementDataTypeString, elemType)
	}

	// Verify shape
	gotShape, err := tensor.GetTensorShape()
	if err != nil {
		t.Fatalf("Failed to get shape: %v", err)
	}
	if !slices.Equal(gotShape, shape) {
		t.Errorf("Shape mismatch: expected %v, got %v", shape, gotShape)
	}
}

func TestGetStringTensorData(t *testing.T) {
	runtime := newTestRuntime(t)

	data := []string{"hello", "world", "foo bar baz"}
	shape := []int64{3}

	tensor, err := runtime.NewStringTensorValue(data, shape)
	if err != nil {
		t.Fatalf("Failed to create string tensor: %v", err)
	}
	defer tensor.Close()

	gotData, gotShape, err := GetStringTensorData(tensor)
	if err != nil {
		t.Fatalf("Failed to get string tensor data: %v", err)
	}

	if !slices.Equal(gotData, data) {
		t.Errorf("Data mismatch: expected %v, got %v", data, gotData)
	}
	if !slices.Equal(gotShape, shape) {
		t.Errorf("Shape mismatch: expected %v, got %v", shape, gotShape)
	}
}

func TestGetStringTensorDataMultiDim(t *testing.T) {
	runtime := newTestRuntime(t)

	data := []string{"a", "b", "c", "d", "e", "f"}
	shape := []int64{2, 3}

	tensor, err := runtime.NewStringTensorValue(data, shape)
	if err != nil {
		t.Fatalf("Failed to create string tensor: %v", err)
	}
	defer tensor.Close()

	gotData, gotShape, err := GetStringTensorData(tensor)
	if err != nil {
		t.Fatalf("Failed to get string tensor data: %v", err)
	}

	if !slices.Equal(gotData, data) {
		t.Errorf("Data mismatch: expected %v, got %v", data, gotData)
	}
	if !slices.Equal(gotShape, shape) {
		t.Errorf("Shape mismatch: expected %v, got %v", shape, gotShape)
	}
}

func TestStringTensorElement(t *testing.T) {
	runtime := newTestRuntime(t)

	data := []string{"hello", "world", "test"}
	shape := []int64{3}

	tensor, err := runtime.NewStringTensorValue(data, shape)
	if err != nil {
		t.Fatalf("Failed to create string tensor: %v", err)
	}
	defer tensor.Close()

	// Read each element by index
	for i, expected := range data {
		got, err := tensor.GetStringTensorElement(i)
		if err != nil {
			t.Fatalf("Failed to get element %d: %v", i, err)
		}
		if got != expected {
			t.Errorf("Element %d: expected %q, got %q", i, expected, got)
		}
	}

	// Overwrite element 1
	err = tensor.SetStringTensorElement(1, "replaced")
	if err != nil {
		t.Fatalf("Failed to set element: %v", err)
	}

	got, err := tensor.GetStringTensorElement(1)
	if err != nil {
		t.Fatalf("Failed to get element after set: %v", err)
	}
	if got != "replaced" {
		t.Errorf("Expected 'replaced', got %q", got)
	}
}

func TestStringTensorEmptyStrings(t *testing.T) {
	runtime := newTestRuntime(t)

	data := []string{"", "notempty", ""}
	shape := []int64{3}

	tensor, err := runtime.NewStringTensorValue(data, shape)
	if err != nil {
		t.Fatalf("Failed to create string tensor: %v", err)
	}
	defer tensor.Close()

	gotData, _, err := GetStringTensorData(tensor)
	if err != nil {
		t.Fatalf("Failed to get string tensor data: %v", err)
	}

	if !slices.Equal(gotData, data) {
		t.Errorf("Data mismatch: expected %v, got %v", data, gotData)
	}
}

func TestNewStringTensorValueEmpty(t *testing.T) {
	runtime := newTestRuntime(t)

	_, err := runtime.NewStringTensorValue([]string{}, []int64{0})
	if err == nil {
		t.Error("Expected error for empty string data")
	}
}

func TestGetStringTensorDataTypeMismatch(t *testing.T) {
	runtime := newTestRuntime(t)

	// Create a float32 tensor and try to read it as string
	tensor, err := NewTensorValue(runtime, []float32{1, 2, 3}, []int64{3})
	if err != nil {
		t.Fatalf("Failed to create tensor: %v", err)
	}
	defer tensor.Close()

	_, _, err = GetStringTensorData(tensor)
	if err == nil {
		t.Error("Expected error when reading float32 tensor as string")
	}
}
