# onnxer
[![Go Reference](https://pkg.go.dev/badge/github.com/benedoc-inc/onnxer.svg)](https://pkg.go.dev/github.com/benedoc-inc/onnxer)

Pure Go bindings for [ONNX Runtime](https://github.com/microsoft/onnxruntime) using [ebitengine/purego](https://github.com/ebitengine/purego).

This library provides a pure Go interface to ONNX Runtime without requiring cgo, enabling cross-platform machine learning inference in Go applications.

## Why onnxer?

- **Pure Go** — no CGO required. Cross-compiles everywhere Go does.
- **GenAI support** — text generation and multimodal inference via ONNX Runtime GenAI.
- **Multi-version API** — supports ORT 1.23.x and 1.24.x simultaneously.
- **Generics tensor API** — type-safe `NewTensorValue[T]` / `GetTensorData[T]` with compile-time checks.
- **Context cancellation** — `context.Context` wired through to ORT RunOptions for real cancellation.
- **Session pooling** — goroutine-safe `SessionPool` with built-in metrics and observability hooks.
- **Profiling** — built-in ORT profiling support for diagnosing per-operator latency.
- **LoRA adapters** — hot-swap fine-tuned adapters per inference run without reloading the base model.
- **Comprehensive** — string tensors, IO binding, model metadata, type introspection, Float16/BFloat16, sequence/map outputs.

## Feature Comparison

| Feature | onnxer | onnxruntime_go |
|---------|--------|----------------|
| Pure Go (no CGO) | Yes | No |
| GenAI support | Yes | No |
| Multi-version API (v23+v24) | Yes | No |
| Generics tensor API | Yes | No |
| String tensors | Yes | Yes |
| Session options (graph opt, threading, memory) | Yes | Yes |
| Model metadata | Yes | Yes |
| Context cancellation (wired to ORT) | Yes | No |
| Session pooling with metrics | Yes | No |
| Inference hooks (observability) | Yes | No |
| IO binding | Yes | Yes |
| Type introspection | Yes | Yes |
| Sequence/Map outputs | Yes | Yes |
| Float16/BFloat16 | Yes | Yes |
| Profiling (per-operator timing) | Yes | No |
| LoRA adapter hot-swap | Yes | No |
| Optimized model caching | Yes | No |
| Zero-copy tensor access | Yes | No |
| Symbolic dimension introspection | Yes | No |
| Dynamic dimension overrides | Yes | Yes |
| Deterministic compute mode | Yes | No |
| Run tagging (log correlation) | Yes | No |
| IO binding synchronization | Yes | No |
| Race-tested concurrent pool | Yes | No |

## Supported Versions

| Library | Supported Version |
|---------|-------------------|
| ONNX Runtime | 1.23.x, 1.24.x |
| ONNX Runtime GenAI | 0.11.x |

## Prerequisites

You need to have the ONNX Runtime shared library installed on your system:

- **macOS**: `libonnxruntime.dylib`
- **Linux**: `libonnxruntime.so`
- **Windows**: `onnxruntime.dll`

Download the appropriate library from the [ONNX Runtime releases](https://github.com/microsoft/onnxruntime/releases).

The library will be automatically discovered if placed in standard system locations:

- **macOS**: `/usr/local/lib`, `/opt/homebrew/lib`, `/usr/lib`
- **Linux**: `/usr/local/lib`, `/usr/lib`, `/lib`
- **Windows**: Standard DLL search paths

Alternatively, you can specify a custom path when creating the runtime.

## Installation

```bash
go get github.com/benedoc-inc/onnxer
```

## Quick Start

```go
package main

import (
	"context"
	"fmt"
	"os"

	ort "github.com/benedoc-inc/onnxer/onnxruntime"
)

func main() {
	runtime, _ := ort.NewRuntime("", 23)
	defer runtime.Close()

	env, _ := runtime.NewEnv("example", ort.LoggingLevelWarning)
	defer env.Close()

	f, _ := os.Open("model.onnx")
	defer f.Close()

	session, _ := runtime.NewSessionFromReader(env, f, &ort.SessionOptions{
		IntraOpNumThreads: 4,
		GraphOptimization: ort.GraphOptimizationAll,
	})
	defer session.Close()

	input, _ := ort.NewTensorValue(runtime, []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, []int64{1, 10})
	defer input.Close()

	outputs, _ := session.Run(context.Background(), map[string]*ort.Value{
		session.InputNames()[0]: input,
	})

	data, shape, _ := ort.GetTensorData[float32](outputs[session.OutputNames()[0]])
	fmt.Printf("Output shape: %v, data: %v\n", shape, data)
}
```

## One-Line Model Loading

For simple use cases, `Model` wraps Runtime + Env + Session into a single object:

```go
model, _ := ort.LoadModelFromFile("model.onnx", &ort.ModelConfig{
    SessionOptions: &ort.SessionOptions{
        IntraOpNumThreads: 4,
        GraphOptimization: ort.GraphOptimizationAll,
    },
})
defer model.Close()

outputs, _ := model.Run(ctx, map[string]*ort.Value{"input": tensor})
```

## Session Pooling

`SessionPool` manages multiple sessions for safe concurrent inference from many goroutines:

```go
pool, _ := ort.NewSessionPool(runtime, env, modelBytes, 8, &ort.PoolConfig{
    Hooks: []ort.Hook{
        ort.AfterRunHook(func(info *ort.RunInfo) {
            log.Printf("inference took %v", info.Duration)
        }),
    },
})
defer pool.Close()

// Safe to call from many goroutines concurrently:
outputs, _ := pool.Run(ctx, map[string]*ort.Value{"input": tensor})

// Built-in metrics:
stats := pool.Stats()
fmt.Printf("runs=%d avg=%v errors=%d\n", stats.TotalRuns, stats.AvgLatency(), stats.TotalErrors)
```

## Examples

See the [`examples/`](./examples/) directory for complete usage examples:

- [**resnet**](./examples/resnet/) — Image classification
- [**roberta-sentiment**](./examples/roberta-sentiment/) — Sentiment analysis
- [**yolov10**](./examples/yolov10/) — Object detection
- [**string-tensor**](./examples/string-tensor/) — String tensor inputs for NLP
- [**metadata**](./examples/metadata/) — Model introspection
- [**pool**](./examples/pool/) — Concurrent inference with session pooling, hooks, warm-up
- [**cancellation**](./examples/cancellation/) — Context-based cancellation
- [**genai/phi3**](./examples/genai/phi3/) — Text generation with Phi-3
- [**genai/phi3.5-vision**](./examples/genai/phi3.5-vision/) — Multimodal vision-language

## ONNX Runtime GenAI Support

This library also includes experimental support for [ONNX Runtime GenAI](https://github.com/microsoft/onnxruntime-genai), enabling text generation with large language models. See the GenAI examples for details.
