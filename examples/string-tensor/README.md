# String Tensor Example

This example demonstrates creating and using string tensors with ONNX Runtime.

String tensors are required for NLP models that accept raw text input (e.g., text classification, named entity recognition).

## Model

Any ONNX model that accepts string tensor inputs will work. For example, an sklearn text classifier exported to ONNX:

```python
# Export an sklearn pipeline with TfidfVectorizer to ONNX
from skl2onnx import convert_sklearn
```

## Usage

```bash
export ONNXRUNTIME_LIB_PATH=/path/to/libonnxruntime.dylib

go run main.go -f ./model.onnx "This is great" "This is terrible"
```
