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
	modelPath   = flag.String("f", "", "path to ONNX model file (required)")
	cancelAfter = flag.Duration("cancel-after", 0, "cancel inference after this duration (e.g., 100ms)")
)

func run(ctx context.Context, modelPath string, cancelAfter time.Duration) error {
	libraryPath := os.Getenv("ONNXRUNTIME_LIB_PATH")
	if libraryPath == "" {
		return errors.New("ONNXRUNTIME_LIB_PATH environment variable not set")
	}

	runtime, err := ort.NewRuntime(libraryPath, 23)
	if err != nil {
		return fmt.Errorf("failed to create runtime: %w", err)
	}
	defer runtime.Close()

	env, err := runtime.NewEnv("cancellation-example", ort.LoggingLevelWarning)
	if err != nil {
		return fmt.Errorf("failed to create environment: %w", err)
	}
	defer env.Close()

	modelFile, err := os.Open(modelPath)
	if err != nil {
		return fmt.Errorf("failed to open model file: %w", err)
	}
	defer modelFile.Close()

	session, err := runtime.NewSessionFromReader(env, modelFile, &ort.SessionOptions{
		IntraOpNumThreads: 1,
	})
	if err != nil {
		return fmt.Errorf("failed to create session: %w", err)
	}
	defer session.Close()

	inputNames := session.InputNames()
	outputNames := session.OutputNames()
	fmt.Printf("Input names:  %v\n", inputNames)
	fmt.Printf("Output names: %v\n", outputNames)

	// Get input info to create appropriately-shaped dummy data
	inputInfo, err := session.GetInputInfo()
	if err != nil {
		return fmt.Errorf("failed to get input info: %w", err)
	}

	// Create dummy input tensors matching the model's expected shapes
	inputs := make(map[string]*ort.Value)
	for _, info := range inputInfo {
		if info.TensorInfo == nil {
			continue
		}
		shape := info.TensorInfo.Shape
		// Replace dynamic dimensions (-1) with 1
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
		tensor, err := ort.NewTensorValue(runtime, data, shape)
		if err != nil {
			return fmt.Errorf("failed to create input tensor %q: %w", info.Name, err)
		}
		defer tensor.Close()
		inputs[info.Name] = tensor
	}

	// Run with cancellation
	runCtx := ctx
	if cancelAfter > 0 {
		var cancel context.CancelFunc
		runCtx, cancel = context.WithTimeout(ctx, cancelAfter)
		defer cancel()
		fmt.Printf("\nRunning inference with %v timeout...\n", cancelAfter)
	} else {
		fmt.Println("\nRunning inference (no timeout)...")
	}

	start := time.Now()
	outputs, err := session.Run(runCtx, inputs)
	elapsed := time.Since(start)

	if err != nil {
		fmt.Printf("Inference failed after %v: %v\n", elapsed, err)
		if runCtx.Err() != nil {
			fmt.Println("(cancelled via context)")
		}
		return nil // not an error for this demo
	}

	fmt.Printf("Inference completed in %v\n", elapsed)

	for name, output := range outputs {
		shape, _ := output.GetTensorShape()
		fmt.Printf("  %s: shape %v\n", name, shape)
		output.Close()
	}

	// Also demonstrate manual cancellation with context.WithCancel
	fmt.Println("\n--- Manual cancellation demo ---")
	manualCtx, manualCancel := context.WithCancel(ctx)

	// Cancel immediately to demonstrate the mechanism
	manualCancel()

	start = time.Now()
	_, err = session.Run(manualCtx, inputs)
	elapsed = time.Since(start)

	if err != nil {
		fmt.Printf("Pre-cancelled run failed after %v: %v\n", elapsed, err)
		fmt.Println("Context cancellation is working correctly!")
	} else {
		fmt.Printf("Run completed in %v (model was too fast to cancel)\n", elapsed)
		for _, output := range outputs {
			output.Close()
		}
	}

	return nil
}

func main() {
	flag.Parse()

	if *modelPath == "" {
		fmt.Fprintf(os.Stderr, "Error: -f flag is required\n\n")
		fmt.Fprintf(os.Stderr, "Usage: %s -f <model_path> [-cancel-after 100ms]\n", os.Args[0])
		flag.PrintDefaults()
		os.Exit(1)
	}

	ctx := context.Background()
	if err := run(ctx, *modelPath, *cancelAfter); err != nil {
		log.Fatalf("Error: %v", err)
	}
}
