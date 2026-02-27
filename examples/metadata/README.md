# Model Metadata Example

This example demonstrates how to inspect an ONNX model's metadata, input/output types, and shapes without running inference.

## Usage

```bash
export ONNXRUNTIME_LIB_PATH=/path/to/libonnxruntime.dylib

go run main.go -f ./model.onnx
```

## Output

```
=== Model Metadata ===
Producer:    pytorch
Graph Name:  main_graph
Domain:
Description:
Version:     0

=== Inputs ===
[0] input
    Type: Tensor
    Element Type: float32
    Shape: [1 10]

=== Outputs ===
[0] output
    Type: Tensor
    Element Type: float32
    Shape: [1 5]
```
