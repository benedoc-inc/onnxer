package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log"
	"os"
	"sync"
	"sync/atomic"
	"time"

	ort "github.com/benedoc-inc/onnxer/onnxruntime"
)

var (
	modelPath    = flag.String("f", "", "path to ONNX model file (required)")
	poolSize     = flag.Int("pool-size", 4, "number of sessions in the pool")
	concurrency  = flag.Int("concurrency", 8, "number of concurrent inference goroutines")
	iterations   = flag.Int("iterations", 100, "total inference runs")
	intraThreads = flag.Int("intra-threads", 4, "global intra-op thread count")
	interThreads = flag.Int("inter-threads", 2, "global inter-op thread count")
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

	// --- Configure global thread pools ---
	// Instead of each session creating its own threads, all sessions share
	// a single thread pool. This saves memory and avoids thread over-subscription
	// when running many concurrent sessions.
	threadOpts, err := runtime.NewThreadingOptions()
	if err != nil {
		return fmt.Errorf("failed to create threading options: %w", err)
	}
	defer threadOpts.Close()

	if err := threadOpts.SetIntraOpNumThreads(*intraThreads); err != nil {
		return fmt.Errorf("failed to set intra-op threads: %w", err)
	}
	if err := threadOpts.SetInterOpNumThreads(*interThreads); err != nil {
		return fmt.Errorf("failed to set inter-op threads: %w", err)
	}
	// Disable spin-wait to reduce idle CPU usage in server workloads
	if err := threadOpts.SetSpinControl(false); err != nil {
		return fmt.Errorf("failed to set spin control: %w", err)
	}

	fmt.Printf("Global thread pool: intra=%d, inter=%d, spin=off\n",
		*intraThreads, *interThreads)

	// Create environment with global thread pools
	env, err := runtime.NewEnvWithGlobalThreadPools("global-threads-example", ort.LoggingLevelWarning, threadOpts)
	if err != nil {
		return fmt.Errorf("failed to create environment: %w", err)
	}
	defer env.Close()

	modelData, err := os.ReadFile(*modelPath)
	if err != nil {
		return fmt.Errorf("failed to read model: %w", err)
	}

	// Create pool — sessions disable per-session threads and use global pool.
	// Also enable prepacked weights sharing to save memory.
	pool, err := ort.NewSessionPool(runtime, env, modelData, *poolSize, &ort.PoolConfig{
		SessionOptions: &ort.SessionOptions{
			DisablePerSessionThreads: true,
			GraphOptimization:        ort.GraphOptimizationAll,
		},
		SharePrepackedWeights: true,
	})
	if err != nil {
		return fmt.Errorf("failed to create pool: %w", err)
	}
	defer pool.Close()

	fmt.Printf("Pool created: %d sessions (shared threads + shared weights)\n", pool.Size())
	fmt.Printf("Model inputs:  %v\n", pool.InputNames())
	fmt.Printf("Model outputs: %v\n", pool.OutputNames())

	// Create input tensor
	inputData := make([]float32, 10)
	for i := range inputData {
		inputData[i] = float32(i + 1)
	}

	// Warmup
	warmupTensor, err := ort.NewTensorValue(runtime, inputData, []int64{1, 10})
	if err != nil {
		return fmt.Errorf("failed to create tensor: %w", err)
	}
	defer warmupTensor.Close()

	fmt.Println("\nWarming up...")
	if err := pool.Warmup(ctx, map[string]*ort.Value{"input": warmupTensor}); err != nil {
		return fmt.Errorf("warmup failed: %w", err)
	}
	pool.ResetStats()

	// Run concurrent inference
	fmt.Printf("Running %d inferences with %d goroutines...\n", *iterations, *concurrency)

	var wg sync.WaitGroup
	var errCount atomic.Int32
	sem := make(chan struct{}, *concurrency)
	start := time.Now()

	for i := 0; i < *iterations; i++ {
		wg.Add(1)
		sem <- struct{}{}
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

	stats := pool.Stats()
	fmt.Printf("\n--- Results ---\n")
	fmt.Printf("Total time:     %v\n", elapsed)
	fmt.Printf("Total runs:     %d\n", stats.TotalRuns)
	fmt.Printf("Total errors:   %d\n", stats.TotalErrors)
	fmt.Printf("Avg latency:    %v\n", stats.AvgLatency())
	fmt.Printf("Throughput:     %.0f inferences/sec\n", float64(stats.TotalRuns)/elapsed.Seconds())
	fmt.Printf("Pool size:      %d\n", stats.PoolSize)

	return nil
}

func main() {
	flag.Parse()

	if *modelPath == "" {
		fmt.Fprintf(os.Stderr, "Usage: %s -f <model_path> [-pool-size 4] [-intra-threads 4] [-inter-threads 2]\n", os.Args[0])
		flag.PrintDefaults()
		os.Exit(1)
	}

	if err := run(context.Background()); err != nil {
		log.Fatalf("Error: %v", err)
	}
}
