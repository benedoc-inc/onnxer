# YOLOv10 Object Detection Example

This example demonstrates real-time object detection using YOLOv10 with ONNX Runtime.

## Model

Download YOLOv10 model using curl:

```bash
# YOLOv10n (nano - recommended for testing, fastest)
curl -L -o yolov10n.onnx \
  "https://huggingface.co/onnx-community/yolov10n/resolve/main/onnx/model.onnx"

# YOLOv10s (small - better accuracy)
curl -L -o yolov10s.onnx \
  "https://huggingface.co/onnx-community/yolov10s/resolve/main/onnx/model.onnx"

# YOLOv10m (medium - high accuracy)
curl -L -o yolov10m.onnx \
  "https://huggingface.co/onnx-community/yolov10m/resolve/main/onnx/model.onnx"
```

Available models from onnx-community:
- **YOLOv10n**: https://huggingface.co/onnx-community/yolov10n (Nano - fastest)
- **YOLOv10s**: https://huggingface.co/onnx-community/yolov10s (Small)
- **YOLOv10m**: https://huggingface.co/onnx-community/yolov10m (Medium)
- **YOLOv10b**: https://huggingface.co/onnx-community/yolov10b (Balanced)
- **YOLOv10l**: https://huggingface.co/onnx-community/yolov10l (Large)
- **YOLOv10x**: https://huggingface.co/onnx-community/yolov10x (Extra Large)

## Usage

```bash
# Set the ONNX Runtime library path
export ONNXRUNTIME_LIB_PATH=/path/to/libonnxruntime.dylib  # macOS
# or
export ONNXRUNTIME_LIB_PATH=/path/to/libonnxruntime.so     # Linux

# Run the example (detections printed to console only)
go run main.go -f ./yolov10n.onnx image.jpg

# With custom confidence threshold
go run main.go -f ./yolov10n.onnx -conf 0.3 image.jpg

# Save output with bounding boxes as PNG
go run main.go -f ./yolov10n.onnx -o result.png image.jpg

# Save output as JPEG (format auto-detected from extension)
go run main.go -f ./yolov10n.onnx -o result.jpg image.jpg
```

## References

- https://huggingface.co/onnx-community/yolov10n
