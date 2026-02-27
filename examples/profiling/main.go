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
	modelPath = flag.String("f", "", "path to ONNX model file (required)")
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

	env, err := runtime.NewEnv("profiling-example", ort.LoggingLevelWarning)
	if err != nil {
		return fmt.Errorf("failed to create environment: %w", err)
	}
	defer env.Close()

	// Create a temp directory for profiling output
	tmpDir, err := os.MkdirTemp("", "ort-profile-*")
	if err != nil {
		return fmt.Errorf("failed to create temp dir: %w", err)
	}
	defer os.RemoveAll(tmpDir)

	profilePrefix := tmpDir + "/profile"

	// Enable profiling via session options
	modelFile, err := os.Open(*modelPath)
	if err != nil {
		return fmt.Errorf("failed to open model: %w", err)
	}
	defer modelFile.Close()

	session, err := runtime.NewSessionFromReader(env, modelFile, &ort.SessionOptions{
		IntraOpNumThreads:   4,
		GraphOptimization:   ort.GraphOptimizationAll,
		ProfilingOutputPath: profilePrefix,
	})
	if err != nil {
		return fmt.Errorf("failed to create session: %w", err)
	}
	defer session.Close()

	fmt.Printf("Profiling enabled (output prefix: %s)\n", profilePrefix)
	fmt.Printf("Model inputs:  %v\n", session.InputNames())
	fmt.Printf("Model outputs: %v\n", session.OutputNames())

	// Get profiling start time
	startNs, err := session.ProfilingStartTimeNs()
	if err != nil {
		return fmt.Errorf("failed to get profiling start time: %w", err)
	}
	fmt.Printf("Profiling start time: %d ns\n\n", startNs)

	// Run inference multiple times to collect meaningful profiling data
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
		tensor, err := ort.NewTensorValue(runtime, data, shape)
		if err != nil {
			return fmt.Errorf("failed to create input tensor: %w", err)
		}
		defer tensor.Close()
		inputs[info.Name] = tensor
	}

	const numRuns = 10
	fmt.Printf("Running %d inferences...\n", numRuns)
	start := time.Now()

	for i := 0; i < numRuns; i++ {
		outputs, err := session.Run(ctx, inputs)
		if err != nil {
			return fmt.Errorf("inference %d failed: %w", i+1, err)
		}
		for _, v := range outputs {
			v.Close()
		}
	}

	elapsed := time.Since(start)
	fmt.Printf("Completed %d runs in %v (avg: %v/run)\n\n", numRuns, elapsed, elapsed/numRuns)

	// End profiling and get the output file
	profilePath, err := session.EndProfiling()
	if err != nil {
		return fmt.Errorf("failed to end profiling: %w", err)
	}

	fmt.Printf("Profiling data written to: %s\n", profilePath)

	// Read and display the profiling file size
	stat, err := os.Stat(profilePath)
	if err != nil {
		return fmt.Errorf("failed to stat profile file: %w", err)
	}
	fmt.Printf("Profile size: %d bytes\n", stat.Size())

	// Read first 500 bytes to show format
	data, err := os.ReadFile(profilePath)
	if err != nil {
		return fmt.Errorf("failed to read profile: %w", err)
	}
	preview := string(data)
	if len(preview) > 500 {
		preview = preview[:500] + "..."
	}
	fmt.Printf("\nProfile preview (JSON):\n%s\n", preview)

	return nil
}

func main() {
	flag.Parse()

	if *modelPath == "" {
		fmt.Fprintf(os.Stderr, "Usage: %s -f <model_path>\n", os.Args[0])
		flag.PrintDefaults()
		os.Exit(1)
	}

	if err := run(context.Background()); err != nil {
		log.Fatalf("Error: %v", err)
	}
}
