# RoBERTa Sentiment Analysis Example

This example demonstrates sentiment analysis using the RoBERTa model with ONNX Runtime. The model classifies text into three sentiment categories: positive, neutral, and negative.

## Requirements

This example uses [sugarme/tokenizer](https://github.com/sugarme/tokenizer), a Go implementation of HuggingFace tokenizers.

## Model

Download the twitter-roberta-base-sentiment model from onnx-community using curl:

```bash
# RoBERTa Base Sentiment (recommended)
curl -L -o roberta-sentiment.onnx \
  "https://huggingface.co/onnx-community/twitter-roberta-base-sentiment-ONNX/resolve/main/onnx/model.onnx"

# Download the tokenizer
curl -L -o tokenizer.json \
  "https://huggingface.co/onnx-community/twitter-roberta-base-sentiment-ONNX/resolve/main/tokenizer.json"
```

### Available Model Variants

The model offers several optimized variants:
- **model.onnx** (1.11 GB) - Full precision (recommended)
- **model_fp16.onnx** (558 MB) - Half precision for faster inference
- **model_int8.onnx** (281 MB) - 8-bit quantized for edge devices
- **model_q4.onnx** (281 MB) - 4-bit quantized

Download quantized models:
```bash
# FP16 (half precision)
curl -L -o roberta-sentiment-fp16.onnx \
  "https://huggingface.co/onnx-community/twitter-roberta-base-sentiment-ONNX/resolve/main/onnx/model_fp16.onnx"

# INT8 (quantized)
curl -L -o roberta-sentiment-int8.onnx \
  "https://huggingface.co/onnx-community/twitter-roberta-base-sentiment-ONNX/resolve/main/onnx/model_int8.onnx"
```

## Usage

```bash
# Set the ONNX Runtime library path
export ONNXRUNTIME_LIB_PATH=/path/to/libonnxruntime.dylib  # macOS
# or
export ONNXRUNTIME_LIB_PATH=/path/to/libonnxruntime.so     # Linux

# Run the example
go run main.go -f ./roberta-sentiment.onnx -tokenizer ./tokenizer.json "I love this movie!"

# Negative sentiment example
go run main.go -f ./roberta-sentiment.onnx -tokenizer ./tokenizer.json "This is terrible."

# Neutral sentiment example
go run main.go -f ./roberta-sentiment.onnx -tokenizer ./tokenizer.json "The movie was okay."

# With custom max length
go run main.go -f ./roberta-sentiment.onnx -tokenizer ./tokenizer.json -max-length 256 "Your text here"
```

## Command-line Options

- `-f`: Path to ONNX model file (required)
- `-tokenizer`: Path to tokenizer.json file (required)
- `-max-length`: Maximum sequence length (default: 128)

## Sentiment Labels

The model classifies text into three categories:

1. **Negative** - Expresses negative sentiment (anger, sadness, disappointment)
2. **Neutral** - Expresses neutral sentiment (factual, objective)
3. **Positive** - Expresses positive sentiment (joy, satisfaction, enthusiasm)


## References

- https://huggingface.co/onnx-community/twitter-roberta-base-sentiment-ONNX
