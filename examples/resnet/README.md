# ResNet Image Classification Example

This example demonstrates image classification using a ResNet model with ONNX Runtime.

## Model

Download a ResNet model from onnx-community using curl:

```bash
# ResNet-50 (recommended)
curl -L -o resnet50.onnx \
  "https://huggingface.co/onnx-community/resnet-50-ONNX/resolve/main/onnx/model.onnx"

# Or ResNet-18 (faster, smaller)
curl -L -o resnet18.onnx \
  "https://huggingface.co/onnx-community/resnet-18-ONNX/resolve/main/onnx/model.onnx"
```

## Usage

```bash
# Set the ONNX Runtime library path
export ONNXRUNTIME_LIB_PATH=/path/to/libonnxruntime.dylib  # macOS
# or
export ONNXRUNTIME_LIB_PATH=/path/to/libonnxruntime.so     # Linux

# Run the example
go run main.go -f ./resnet50.onnx image.jpg
```

## References

- https://huggingface.co/onnx-community/resnet-18-ONNX
