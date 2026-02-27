package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log"
	"os"
	"time"

	ort "github.com/benedoc-inc/onnxer/onnxruntime"
)

var (
	modelPath  = flag.String("f", "", "path to ONNX model file (required)")
	iterations = flag.Int("iterations", 100, "number of inference runs")
)

func run(ctx context.Context) error {
	libraryPath := os.Getenv("ONNXRUNTIME_LIB_PATH")
	if libraryPath == "" {
		return errors.New("ONNXRUNTIME_LIB_PATH environment variable not set")
	}

	runtime, err := ort.NewRuntime(libraryPath, 23)
	if err != nil {
		return fmt.Errorf("failed to create runtime: %w", err)
	}
	defer runtime.Close()

	fmt.Printf("ONNX Runtime %s\n", runtime.GetVersionString())

	env, err := runtime.NewEnv("io-binding-example", ort.LoggingLevelWarning)
	if err != nil {
		return fmt.Errorf("failed to create environment: %w", err)
	}
	defer env.Close()

	modelFile, err := os.Open(*modelPath)
	if err != nil {
		return fmt.Errorf("failed to open model: %w", err)
	}
	defer modelFile.Close()

	session, err := runtime.NewSessionFromReader(env, modelFile, &ort.SessionOptions{
		IntraOpNumThreads: 4,
		GraphOptimization: ort.GraphOptimizationAll,
	})
	if err != nil {
		return fmt.Errorf("failed to create session: %w", err)
	}
	defer session.Close()

	fmt.Printf("Model inputs:  %v\n", session.InputNames())
	fmt.Printf("Model outputs: %v\n", session.OutputNames())

	// Create input tensor
	inputInfo, err := session.GetInputInfo()
	if err != nil {
		return fmt.Errorf("failed to get input info: %w", err)
	}

	var inputTensor *ort.Value
	var inputName string
	for _, info := range inputInfo {
		if info.TensorInfo == nil {
			continue
		}
		inputName = info.Name
		shape := info.TensorInfo.Shape
		for j, d := range shape {
			if d < 0 {
				shape[j] = 1
			}
		}
		size := int64(1)
		for _, d := range shape {
			size *= d
		}
		data := make([]float32, size)
		for i := range data {
			data[i] = float32(i+1) * 0.1
		}
		inputTensor, err = ort.NewTensorValue(runtime, data, shape)
		if err != nil {
			return fmt.Errorf("failed to create input tensor: %w", err)
		}
		defer inputTensor.Close()
		break // just use first input for demo
	}

	// --- Standard Run (for comparison) ---
	fmt.Printf("\n--- Standard Run (%d iterations) ---\n", *iterations)
	start := time.Now()
	for i := 0; i < *iterations; i++ {
		outputs, err := session.Run(ctx, map[string]*ort.Value{inputName: inputTensor})
		if err != nil {
			return fmt.Errorf("standard run failed: %w", err)
		}
		for _, v := range outputs {
			v.Close()
		}
	}
	standardElapsed := time.Since(start)
	fmt.Printf("Total: %v (avg: %v/run)\n", standardElapsed, standardElapsed/time.Duration(*iterations))

	// --- IO Binding Run ---
	fmt.Printf("\n--- IO Binding Run (%d iterations) ---\n", *iterations)

	// Create IO binding
	binding, err := session.NewIoBinding()
	if err != nil {
		return fmt.Errorf("failed to create IO binding: %w", err)
	}
	defer binding.Close()

	// Bind input
	if err := binding.BindInput(inputName, inputTensor); err != nil {
		return fmt.Errorf("failed to bind input: %w", err)
	}

	// Bind outputs to CPU (ORT allocates the output buffer)
	cpuMem, err := runtime.NewCPUMemoryInfo()
	if err != nil {
		return fmt.Errorf("failed to create CPU memory info: %w", err)
	}
	defer cpuMem.Close()

	for _, name := range session.OutputNames() {
		if err := binding.BindOutputToDevice(name, cpuMem); err != nil {
			return fmt.Errorf("failed to bind output %q: %w", name, err)
		}
	}

	fmt.Println("Inputs and outputs bound successfully")

	start = time.Now()
	for i := 0; i < *iterations; i++ {
		if err := binding.Run(ctx); err != nil {
			return fmt.Errorf("IO binding run failed: %w", err)
		}
	}
	bindingElapsed := time.Since(start)
	fmt.Printf("Total: %v (avg: %v/run)\n", bindingElapsed, bindingElapsed/time.Duration(*iterations))

	// Get final outputs
	outputs, err := binding.GetOutputValues()
	if err != nil {
		return fmt.Errorf("failed to get output values: %w", err)
	}

	for name, output := range outputs {
		shape, _ := output.GetTensorShape()
		data, _, _ := ort.GetTensorData[float32](output)
		preview := data
		if len(preview) > 5 {
			preview = preview[:5]
		}
		fmt.Printf("  %s: shape=%v first_values=%v\n", name, shape, preview)
		output.Close()
	}

	// Compare
	fmt.Printf("\n--- Comparison ---\n")
	fmt.Printf("Standard: %v/run\n", standardElapsed/time.Duration(*iterations))
	fmt.Printf("Binding:  %v/run\n", bindingElapsed/time.Duration(*iterations))
	if bindingElapsed < standardElapsed {
		speedup := float64(standardElapsed) / float64(bindingElapsed)
		fmt.Printf("IO Binding is %.1fx faster\n", speedup)
	} else {
		fmt.Println("IO Binding overhead visible on CPU (benefits show on GPU)")
	}

	return nil
}

func main() {
	flag.Parse()

	if *modelPath == "" {
		fmt.Fprintf(os.Stderr, "Usage: %s -f <model_path> [-iterations 100]\n", os.Args[0])
		flag.PrintDefaults()
		os.Exit(1)
	}

	if err := run(context.Background()); err != nil {
		log.Fatalf("Error: %v", err)
	}
}
