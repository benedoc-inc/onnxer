package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log"
	"os"

	ort "github.com/benedoc-inc/onnxer/onnxruntime"
)

var (
	modelPath   = flag.String("f", "", "path to ONNX model file (required)")
	adapterPath = flag.String("adapter", "", "path to LoRA adapter file (.onnx_adapter)")
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

	env, err := runtime.NewEnv("lora-example", ort.LoggingLevelWarning)
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

	// Create dummy inputs
	inputInfo, err := session.GetInputInfo()
	if err != nil {
		return fmt.Errorf("failed to get input info: %w", err)
	}

	inputs := make(map[string]*ort.Value)
	for _, info := range inputInfo {
		if info.TensorInfo == nil {
			continue
		}
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
		tensor, err := ort.NewTensorValue(runtime, data, shape)
		if err != nil {
			return fmt.Errorf("failed to create input tensor: %w", err)
		}
		defer tensor.Close()
		inputs[info.Name] = tensor
	}

	// Run baseline inference (no adapter)
	fmt.Println("\n--- Baseline (no LoRA adapter) ---")
	outputs, err := session.Run(ctx, inputs)
	if err != nil {
		return fmt.Errorf("baseline inference failed: %w", err)
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

	// If adapter path provided, load and run with it
	if *adapterPath != "" {
		fmt.Printf("\n--- Loading LoRA adapter: %s ---\n", *adapterPath)

		adapter, err := runtime.LoadLoraAdapterFromFile(*adapterPath)
		if err != nil {
			return fmt.Errorf("failed to load LoRA adapter: %w", err)
		}
		defer adapter.Close()

		fmt.Println("Adapter loaded successfully")

		// Run with adapter
		outputs, err := session.Run(ctx, inputs, ort.WithLoraAdapters(adapter))
		if err != nil {
			return fmt.Errorf("inference with adapter failed: %w", err)
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

		fmt.Println("\nLoRA adapter hot-swap demonstrated successfully!")
		fmt.Println("The same session can switch between different adapters per-run")
		fmt.Println("without reloading the base model.")
	} else {
		fmt.Println("\nNo adapter provided. Use -adapter <path> to test LoRA hot-swap.")
		fmt.Println("LoRA adapters are .onnx_adapter files that modify base model weights.")
	}

	return nil
}

func main() {
	flag.Parse()

	if *modelPath == "" {
		fmt.Fprintf(os.Stderr, "Usage: %s -f <model_path> [-adapter <adapter_path>]\n", os.Args[0])
		flag.PrintDefaults()
		os.Exit(1)
	}

	if err := run(context.Background()); err != nil {
		log.Fatalf("Error: %v", err)
	}
}
