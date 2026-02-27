package onnxruntime

import (
	"testing"
)

func TestNewIoBinding(t *testing.T) {
	runtime := newTestRuntime(t)
	session := newTestSession(t, runtime)

	binding, err := session.NewIoBinding()
	if err != nil {
		t.Fatalf("Failed to create IO binding: %v", err)
	}
	defer binding.Close()

	if binding.ptr == 0 {
		t.Error("Binding pointer should not be 0")
	}
}

func TestIoBindingRunWithPreallocatedOutput(t *testing.T) {
	runtime := newTestRuntime(t)
	session := newTestSession(t, runtime)

	// Create input tensor
	inputData := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	inputTensor, err := NewTensorValue(runtime, inputData, []int64{1, 10})
	if err != nil {
		t.Fatalf("Failed to create input tensor: %v", err)
	}
	defer inputTensor.Close()

	// Create output tensor (model outputs [1, 3])
	outputData := make([]float32, 3)
	outputTensor, err := NewTensorValue(runtime, outputData, []int64{1, 3})
	if err != nil {
		t.Fatalf("Failed to create output tensor: %v", err)
	}
	defer outputTensor.Close()

	binding, err := session.NewIoBinding()
	if err != nil {
		t.Fatalf("Failed to create IO binding: %v", err)
	}
	defer binding.Close()

	err = binding.BindInput("input", inputTensor)
	if err != nil {
		t.Fatalf("Failed to bind input: %v", err)
	}

	err = binding.BindOutput("logits", outputTensor)
	if err != nil {
		t.Fatalf("Failed to bind output: %v", err)
	}

	err = binding.Run(t.Context())
	if err != nil {
		t.Fatalf("Failed to run with binding: %v", err)
	}

	// Read data from pre-allocated output tensor
	data, shape, err := GetTensorData[float32](outputTensor)
	if err != nil {
		t.Fatalf("Failed to get output data: %v", err)
	}

	if len(data) != 3 {
		t.Errorf("Expected 3 output elements, got %d", len(data))
	}
	if len(shape) != 2 || shape[0] != 1 || shape[1] != 3 {
		t.Errorf("Expected shape [1 3], got %v", shape)
	}
}

func TestIoBindingRunWithDeviceOutput(t *testing.T) {
	runtime := newTestRuntime(t)
	session := newTestSession(t, runtime)

	inputData := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	inputTensor, err := NewTensorValue(runtime, inputData, []int64{1, 10})
	if err != nil {
		t.Fatalf("Failed to create input tensor: %v", err)
	}
	defer inputTensor.Close()

	memInfo, err := runtime.NewCPUMemoryInfo()
	if err != nil {
		t.Fatalf("Failed to create CPU memory info: %v", err)
	}
	defer memInfo.Close()

	binding, err := session.NewIoBinding()
	if err != nil {
		t.Fatalf("Failed to create IO binding: %v", err)
	}
	defer binding.Close()

	err = binding.BindInput("input", inputTensor)
	if err != nil {
		t.Fatalf("Failed to bind input: %v", err)
	}

	err = binding.BindOutputToDevice("logits", memInfo)
	if err != nil {
		t.Fatalf("Failed to bind output to device: %v", err)
	}

	err = binding.Run(t.Context())
	if err != nil {
		t.Fatalf("Failed to run with binding: %v", err)
	}

	// Get output values from the binding
	outputs, err := binding.GetOutputValues()
	if err != nil {
		t.Fatalf("Failed to get output values: %v", err)
	}

	output, ok := outputs["logits"]
	if !ok {
		t.Fatal("Expected 'logits' in output map")
	}
	defer output.Close()

	data, _, err := GetTensorData[float32](output)
	if err != nil {
		t.Fatalf("Failed to get tensor data: %v", err)
	}
	if len(data) != 3 {
		t.Errorf("Expected 3 output elements, got %d", len(data))
	}
}

func TestIoBindingClear(t *testing.T) {
	runtime := newTestRuntime(t)
	session := newTestSession(t, runtime)

	binding, err := session.NewIoBinding()
	if err != nil {
		t.Fatalf("Failed to create IO binding: %v", err)
	}
	defer binding.Close()

	// ClearInputs and ClearOutputs should not panic even with no bindings
	binding.ClearInputs()
	binding.ClearOutputs()
}

func TestNewCPUMemoryInfo(t *testing.T) {
	runtime := newTestRuntime(t)

	memInfo, err := runtime.NewCPUMemoryInfo()
	if err != nil {
		t.Fatalf("Failed to create CPU memory info: %v", err)
	}
	defer memInfo.Close()

	if memInfo.ptr == 0 {
		t.Error("MemoryInfo pointer should not be 0")
	}

	// Double close should not panic
	memInfo.Close()
}

func TestIoBindingClosedSession(t *testing.T) {
	runtime := newTestRuntime(t)
	session := newTestSession(t, runtime)
	session.Close()

	_, err := session.NewIoBinding()
	if err == nil {
		t.Error("Expected error for closed session")
	}
}
