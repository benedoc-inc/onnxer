# ONNX Runtime API Code Generator

A tool to automatically generate Go binding code from ONNX Runtime C API header files.

## Overview

This tool automates the following processes:

1. Downloads ONNX Runtime header files from GitHub
2. Parses the `OrtApi` structure to extract the function list
3. Generates Go binding code (1 file)

## Usage

### Basic Usage

```bash
go run tools/codegen/main.go -version <VERSION> -out <OUTPUT_DIR>
```

## Generated Files

### api.go
- API version constant (`APIVersion`)
- `APIBase` structure (entry point)
- `API` structure (all function pointers, uintptr type)
- Function order matches the C header file
- `CStringToString` helper function

## Header File Parsing

Recognizes the following 3 patterns:

1. **ORT_API2_STATUS macro**
   ```c
   ORT_API2_STATUS(CreateSession, ...)
   ```

2. **ORT_CLASS_RELEASE macro**
   ```c
   ORT_CLASS_RELEASE(Env);  // → ReleaseEnv
   ```

3. **Direct function pointer definition**
   ```c
   void(ORT_API_CALL* GetVersion)(...);
   ```
