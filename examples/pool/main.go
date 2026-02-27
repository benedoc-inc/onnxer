package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log"
	"log/slog"
	"os"
	"sync"
	"sync/atomic"
	"time"

	ort "github.com/benedoc-inc/onnxer/onnxruntime"
)

var (
	modelPath   = flag.String("f", "", "path to ONNX model file (required)")
	poolSize    = flag.Int("pool-size", 4, "number of sessions in the pool")
	concurrency = flag.Int("concurrency", 8, "number of concurrent inference goroutines")
	iterations  = flag.Int("iterations", 100, "total inference runs")
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
	fmt.Printf("Build: %s\n\n", runtime.GetBuildInfo())

	env, err := runtime.NewEnv("pool-example", ort.LoggingLevelWarning)
	if err != nil {
		return fmt.Errorf("failed to create environment: %w", err)
	}
	defer env.Close()

	modelData, err := os.ReadFile(*modelPath)
	if err != nil {
		return fmt.Errorf("failed to read model: %w", err)
	}

	// Create pool with slog hook for structured logging
	pool, err := ort.NewSessionPool(runtime, env, modelData, *poolSize, &ort.PoolConfig{
		SessionOptions: &ort.SessionOptions{
			IntraOpNumThreads: 2,
			GraphOptimization: ort.GraphOptimizationAll,
		},
		Hooks: []ort.Hook{
			// Log every inference to stderr with structured fields
			ort.NewSlogHook(slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{
				Level: slog.LevelWarn, // only log errors, not every inference
			}))),
			// Custom metrics hook
			ort.AfterRunHook(func(info *ort.RunInfo) {
				if info.Error != nil {
					fmt.Fprintf(os.Stderr, "[HOOK] inference error: %v\n", info.Error)
				}
			}),
		},
	})
	if err != nil {
		return fmt.Errorf("failed to create pool: %w", err)
	}
	defer pool.Close()

	fmt.Printf("Pool created: %d sessions\n", pool.Size())
	fmt.Printf("Model inputs:  %v\n", pool.InputNames())
	fmt.Printf("Model outputs: %v\n", pool.OutputNames())

	// Create dummy input for warm-up and inference
	inputData := make([]float32, 10)
	for i := range inputData {
		inputData[i] = float32(i + 1)
	}
	warmupTensor, err := ort.NewTensorValue(runtime, inputData, []int64{1, 10})
	if err != nil {
		return fmt.Errorf("failed to create tensor: %w", err)
	}
	defer warmupTensor.Close()

	inputs := map[string]*ort.Value{"input": warmupTensor}

	// Warm up all sessions
	fmt.Println("\nWarming up pool...")
	warmupStart := time.Now()
	if err := pool.Warmup(ctx, inputs); err != nil {
		return fmt.Errorf("warmup failed: %w", err)
	}
	fmt.Printf("Warmup completed in %v (%d sessions)\n", time.Since(warmupStart), pool.Size())

	// Health check
	if err := pool.HealthCheck(ctx, inputs); err != nil {
		return fmt.Errorf("health check failed: %w", err)
	}
	fmt.Println("Health check: OK")

	// Reset stats after warmup so they reflect real workload
	pool.ResetStats()

	// Run concurrent inference
	fmt.Printf("\nRunning %d inferences with %d goroutines...\n", *iterations, *concurrency)

	var wg sync.WaitGroup
	var errCount atomic.Int32
	sem := make(chan struct{}, *concurrency)
	start := time.Now()

	for i := 0; i < *iterations; i++ {
		wg.Add(1)
		sem <- struct{}{} // limit concurrency
		go func() {
			defer wg.Done()
			defer func() { <-sem }()

			tensor, err := ort.NewTensorValue(runtime, inputData, []int64{1, 10})
			if err != nil {
				errCount.Add(1)
				return
			}
			defer tensor.Close()

			outputs, err := pool.Run(ctx, map[string]*ort.Value{"input": tensor})
			if err != nil {
				errCount.Add(1)
				return
			}
			for _, v := range outputs {
				v.Close()
			}
		}()
	}

	wg.Wait()
	elapsed := time.Since(start)

	// Print results
	stats := pool.Stats()
	fmt.Printf("\n--- Results ---\n")
	fmt.Printf("Total time:     %v\n", elapsed)
	fmt.Printf("Total runs:     %d\n", stats.TotalRuns)
	fmt.Printf("Total errors:   %d (goroutine errors: %d)\n", stats.TotalErrors, errCount.Load())
	fmt.Printf("Avg latency:    %v\n", stats.AvgLatency())
	fmt.Printf("Throughput:     %.0f inferences/sec\n", float64(stats.TotalRuns)/elapsed.Seconds())
	fmt.Printf("Pool size:      %d\n", stats.PoolSize)
	fmt.Printf("Available now:  %d\n", stats.AvailableSessions)

	return nil
}

func main() {
	flag.Parse()

	if *modelPath == "" {
		fmt.Fprintf(os.Stderr, "Usage: %s -f <model_path> [-pool-size 4] [-concurrency 8] [-iterations 100]\n", os.Args[0])
		flag.PrintDefaults()
		os.Exit(1)
	}

	ctx := context.Background()
	if err := run(ctx); err != nil {
		log.Fatalf("Error: %v", err)
	}
}
