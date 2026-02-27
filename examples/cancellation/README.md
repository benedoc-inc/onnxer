# Context Cancellation Example

This example demonstrates how to use Go's `context.Context` to cancel ONNX Runtime inference mid-execution.

This is useful for:
- Setting timeouts on inference calls
- Cancelling long-running inference when a request is aborted
- Graceful shutdown of inference servers

## Usage

```bash
export ONNXRUNTIME_LIB_PATH=/path/to/libonnxruntime.dylib

# Run with a timeout
go run main.go -f ./model.onnx -cancel-after 100ms

# Run without timeout (normal execution)
go run main.go -f ./model.onnx
```

## How It Works

When a cancellable `context.Context` is passed to `session.Run()`, onnxer creates ORT RunOptions and spawns a goroutine that watches for context cancellation. When the context is cancelled, it calls `RunOptionsSetTerminate` to signal ONNX Runtime to abort the current inference.
