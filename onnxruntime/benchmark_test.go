package onnxruntime

import (
	"bytes"
	"context"
	"os"
	"testing"
)

func BenchmarkSessionRun(b *testing.B) {
	runtime, err := NewRuntime(libraryPath, 23)
	if err != nil {
		b.Skipf("Skipping: ONNX Runtime library not available: %v", err)
	}
	defer runtime.Close()

	env, err := runtime.NewEnv("test", LoggingLevelWarning)
	if err != nil {
		b.Fatalf("Failed to create environment: %v", err)
	}
	defer env.Close()

	modelData, err := os.ReadFile(testModelPath())
	if err != nil {
		b.Fatalf("Failed to read model file: %v", err)
	}

	session, err := runtime.NewSessionFromReader(env, bytes.NewReader(modelData), nil)
	if err != nil {
		b.Fatalf("Failed to create session: %v", err)
	}
	defer session.Close()

	inputData := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0}
	inputShape := []int64{1, 10}

	inputTensor, err := NewTensorValue(runtime, inputData, inputShape)
	if err != nil {
		b.Fatalf("Failed to create input tensor: %v", err)
	}
	defer inputTensor.Close()

	inputs := map[string]*Value{
		"input": inputTensor,
	}

	for b.Loop() {
		outputs, err := session.Run(b.Context(), inputs)
		if err != nil {
			b.Fatalf("Failed to run inference: %v", err)
		}
		for _, output := range outputs {
			output.Close()
		}
	}
}

func BenchmarkTensorCreation(b *testing.B) {
	runtime, err := NewRuntime(libraryPath, 23)
	if err != nil {
		b.Skipf("Skipping: ONNX Runtime library not available: %v", err)
	}
	defer runtime.Close()

	data := make([]float32, 1000)
	for i := range data {
		data[i] = float32(i)
	}
	shape := []int64{10, 100}

	for b.Loop() {
		tensor, err := NewTensorValue(runtime, data, shape)
		if err != nil {
			b.Fatalf("Failed to create tensor: %v", err)
		}
		tensor.Close()
	}
}

func BenchmarkTensorCreationLarge(b *testing.B) {
	runtime, err := NewRuntime(libraryPath, 23)
	if err != nil {
		b.Skipf("Skipping: ONNX Runtime library not available: %v", err)
	}
	defer runtime.Close()

	data := make([]float32, 224*224*3)
	for i := range data {
		data[i] = float32(i)
	}
	shape := []int64{1, 3, 224, 224}

	for b.Loop() {
		tensor, err := NewTensorValue(runtime, data, shape)
		if err != nil {
			b.Fatalf("Failed to create tensor: %v", err)
		}
		tensor.Close()
	}
}

func BenchmarkGetTensorData(b *testing.B) {
	runtime, err := NewRuntime(libraryPath, 23)
	if err != nil {
		b.Skipf("Skipping: ONNX Runtime library not available: %v", err)
	}
	defer runtime.Close()

	data := make([]float32, 1000)
	for i := range data {
		data[i] = float32(i)
	}

	tensor, err := NewTensorValue(runtime, data, []int64{10, 100})
	if err != nil {
		b.Fatalf("Failed to create tensor: %v", err)
	}
	defer tensor.Close()

	for b.Loop() {
		_, _, err := GetTensorData[float32](tensor)
		if err != nil {
			b.Fatalf("Failed to get tensor data: %v", err)
		}
	}
}

func BenchmarkStringTensorCreation(b *testing.B) {
	runtime, err := NewRuntime(libraryPath, 23)
	if err != nil {
		b.Skipf("Skipping: ONNX Runtime library not available: %v", err)
	}
	defer runtime.Close()

	data := []string{
		"The quick brown fox jumps over the lazy dog",
		"ONNX Runtime is a high-performance inference engine",
		"Go is a statically typed compiled language",
		"Machine learning inference at the edge",
	}
	shape := []int64{4}

	for b.Loop() {
		tensor, err := runtime.NewStringTensorValue(data, shape)
		if err != nil {
			b.Fatalf("Failed to create string tensor: %v", err)
		}
		tensor.Close()
	}
}

func BenchmarkSessionPool(b *testing.B) {
	runtime, err := NewRuntime(libraryPath, 23)
	if err != nil {
		b.Skipf("Skipping: ONNX Runtime library not available: %v", err)
	}
	defer runtime.Close()

	env, err := runtime.NewEnv("bench", LoggingLevelWarning)
	if err != nil {
		b.Fatalf("Failed to create env: %v", err)
	}
	defer env.Close()

	modelData, err := os.ReadFile(testModelPath())
	if err != nil {
		b.Fatalf("Failed to read model: %v", err)
	}

	pool, err := NewSessionPool(runtime, env, modelData, 4, nil)
	if err != nil {
		b.Fatalf("Failed to create pool: %v", err)
	}
	defer pool.Close()

	inputTensor, err := NewTensorValue(runtime, []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, []int64{1, 10})
	if err != nil {
		b.Fatalf("Failed to create tensor: %v", err)
	}
	defer inputTensor.Close()

	inputs := map[string]*Value{"input": inputTensor}

	for b.Loop() {
		outputs, err := pool.Run(context.Background(), inputs)
		if err != nil {
			b.Fatalf("Failed to run: %v", err)
		}
		for _, v := range outputs {
			v.Close()
		}
	}
}

func BenchmarkSessionPoolConcurrent(b *testing.B) {
	runtime, err := NewRuntime(libraryPath, 23)
	if err != nil {
		b.Skipf("Skipping: ONNX Runtime library not available: %v", err)
	}
	defer runtime.Close()

	env, err := runtime.NewEnv("bench", LoggingLevelWarning)
	if err != nil {
		b.Fatalf("Failed to create env: %v", err)
	}
	defer env.Close()

	modelData, err := os.ReadFile(testModelPath())
	if err != nil {
		b.Fatalf("Failed to read model: %v", err)
	}

	pool, err := NewSessionPool(runtime, env, modelData, 4, nil)
	if err != nil {
		b.Fatalf("Failed to create pool: %v", err)
	}
	defer pool.Close()

	b.RunParallel(func(pb *testing.PB) {
		tensor, err := NewTensorValue(runtime, []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, []int64{1, 10})
		if err != nil {
			b.Fatalf("Failed to create tensor: %v", err)
		}
		defer tensor.Close()

		inputs := map[string]*Value{"input": tensor}

		for pb.Next() {
			outputs, err := pool.Run(context.Background(), inputs)
			if err != nil {
				b.Fatalf("Failed to run: %v", err)
			}
			for _, v := range outputs {
				v.Close()
			}
		}
	})
}

func BenchmarkGetTensorDataUnsafe(b *testing.B) {
	runtime, err := NewRuntime(libraryPath, 23)
	if err != nil {
		b.Skipf("Skipping: ONNX Runtime library not available: %v", err)
	}
	defer runtime.Close()

	data := make([]float32, 1000)
	for i := range data {
		data[i] = float32(i)
	}

	tensor, err := NewTensorValue(runtime, data, []int64{10, 100})
	if err != nil {
		b.Fatalf("Failed to create tensor: %v", err)
	}
	defer tensor.Close()

	for b.Loop() {
		_, _, err := GetTensorDataUnsafe[float32](tensor)
		if err != nil {
			b.Fatalf("Failed to get tensor data: %v", err)
		}
	}
}

func BenchmarkModelLoad(b *testing.B) {
	runtime, err := NewRuntime(libraryPath, 23)
	if err != nil {
		b.Skipf("Skipping: ONNX Runtime library not available: %v", err)
	}
	defer runtime.Close()

	modelData, err := os.ReadFile(testModelPath())
	if err != nil {
		b.Fatalf("Failed to read model: %v", err)
	}

	env, err := runtime.NewEnv("bench", LoggingLevelWarning)
	if err != nil {
		b.Fatalf("Failed to create env: %v", err)
	}
	defer env.Close()

	for b.Loop() {
		session, err := runtime.NewSessionFromReader(env, bytes.NewReader(modelData), nil)
		if err != nil {
			b.Fatalf("Failed to create session: %v", err)
		}
		session.Close()
	}
}

func BenchmarkConcurrentSessionRun(b *testing.B) {
	runtime, err := NewRuntime(libraryPath, 23)
	if err != nil {
		b.Skipf("Skipping: ONNX Runtime library not available: %v", err)
	}
	defer runtime.Close()

	env, err := runtime.NewEnv("bench", LoggingLevelWarning)
	if err != nil {
		b.Fatalf("Failed to create env: %v", err)
	}
	defer env.Close()

	modelData, err := os.ReadFile(testModelPath())
	if err != nil {
		b.Fatalf("Failed to read model: %v", err)
	}

	// Create one session per goroutine via pool for thread safety
	const numSessions = 4
	pool, err := NewSessionPool(runtime, env, modelData, numSessions, nil)
	if err != nil {
		b.Fatalf("Failed to create pool: %v", err)
	}
	defer pool.Close()

	b.RunParallel(func(pb *testing.PB) {
		tensor, err := NewTensorValue(runtime, []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, []int64{1, 10})
		if err != nil {
			b.Fatalf("Failed to create tensor: %v", err)
		}
		defer tensor.Close()

		inputs := map[string]*Value{"input": tensor}

		for pb.Next() {
			outputs, err := pool.Run(context.Background(), inputs)
			if err != nil {
				b.Fatalf("Failed to run: %v", err)
			}
			for _, v := range outputs {
				v.Close()
			}
		}
	})
}

func BenchmarkFloat16Conversion(b *testing.B) {
	for b.Loop() {
		f16 := NewFloat16(3.14)
		_ = f16.Float32()
	}
}

func BenchmarkBFloat16Conversion(b *testing.B) {
	for b.Loop() {
		bf16 := NewBFloat16(3.14)
		_ = bf16.Float32()
	}
}
