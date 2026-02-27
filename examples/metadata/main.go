package main

import (
	"errors"
	"flag"
	"fmt"
	"log"
	"os"

	ort "github.com/benedoc-inc/onnxer/onnxruntime"
)

var (
	modelPath = flag.String("f", "", "path to ONNX model file (required)")
)

func run(modelPath string) error {
	libraryPath := os.Getenv("ONNXRUNTIME_LIB_PATH")
	if libraryPath == "" {
		return errors.New("ONNXRUNTIME_LIB_PATH environment variable not set")
	}

	runtime, err := ort.NewRuntime(libraryPath, 23)
	if err != nil {
		return fmt.Errorf("failed to create runtime: %w", err)
	}
	defer runtime.Close()

	env, err := runtime.NewEnv("metadata-example", ort.LoggingLevelWarning)
	if err != nil {
		return fmt.Errorf("failed to create environment: %w", err)
	}
	defer env.Close()

	modelFile, err := os.Open(modelPath)
	if err != nil {
		return fmt.Errorf("failed to open model file: %w", err)
	}
	defer modelFile.Close()

	session, err := runtime.NewSessionFromReader(env, modelFile, nil)
	if err != nil {
		return fmt.Errorf("failed to create session: %w", err)
	}
	defer session.Close()

	// Model metadata
	metadata, err := session.GetModelMetadata()
	if err != nil {
		return fmt.Errorf("failed to get model metadata: %w", err)
	}

	fmt.Println("=== Model Metadata ===")
	fmt.Printf("Producer:    %s\n", metadata.ProducerName)
	fmt.Printf("Graph Name:  %s\n", metadata.GraphName)
	fmt.Printf("Domain:      %s\n", metadata.Domain)
	fmt.Printf("Description: %s\n", metadata.Description)
	fmt.Printf("Version:     %d\n", metadata.Version)

	if len(metadata.CustomMetadata) > 0 {
		fmt.Println("\nCustom Metadata:")
		for k, v := range metadata.CustomMetadata {
			fmt.Printf("  %s = %s\n", k, v)
		}
	}

	// Input/output info
	fmt.Println("\n=== Inputs ===")
	inputInfo, err := session.GetInputInfo()
	if err != nil {
		return fmt.Errorf("failed to get input info: %w", err)
	}
	for i, info := range inputInfo {
		fmt.Printf("[%d] %s\n", i, info.Name)
		fmt.Printf("    Type: %s\n", onnxTypeName(info.Type))
		if info.TensorInfo != nil {
			fmt.Printf("    Element Type: %s\n", elementTypeName(info.TensorInfo.ElementType))
			fmt.Printf("    Shape: %v\n", info.TensorInfo.Shape)
		}
	}

	fmt.Println("\n=== Outputs ===")
	outputInfo, err := session.GetOutputInfo()
	if err != nil {
		return fmt.Errorf("failed to get output info: %w", err)
	}
	for i, info := range outputInfo {
		fmt.Printf("[%d] %s\n", i, info.Name)
		fmt.Printf("    Type: %s\n", onnxTypeName(info.Type))
		if info.TensorInfo != nil {
			fmt.Printf("    Element Type: %s\n", elementTypeName(info.TensorInfo.ElementType))
			fmt.Printf("    Shape: %v\n", info.TensorInfo.Shape)
		}
	}

	return nil
}

func onnxTypeName(t ort.ONNXType) string {
	switch t {
	case ort.ONNXTypeTensor:
		return "Tensor"
	case ort.ONNXTypeSequence:
		return "Sequence"
	case ort.ONNXTypeMap:
		return "Map"
	case ort.ONNXTypeOpaque:
		return "Opaque"
	case ort.ONNXTypeSparsetensor:
		return "SparseTensor"
	case ort.ONNXTypeOptional:
		return "Optional"
	default:
		return fmt.Sprintf("Unknown(%d)", t)
	}
}

func elementTypeName(t ort.ONNXTensorElementDataType) string {
	switch t {
	case ort.ONNXTensorElementDataTypeFloat:
		return "float32"
	case ort.ONNXTensorElementDataTypeUint8:
		return "uint8"
	case ort.ONNXTensorElementDataTypeInt8:
		return "int8"
	case ort.ONNXTensorElementDataTypeUint16:
		return "uint16"
	case ort.ONNXTensorElementDataTypeInt16:
		return "int16"
	case ort.ONNXTensorElementDataTypeInt32:
		return "int32"
	case ort.ONNXTensorElementDataTypeInt64:
		return "int64"
	case ort.ONNXTensorElementDataTypeString:
		return "string"
	case ort.ONNXTensorElementDataTypeBool:
		return "bool"
	case ort.ONNXTensorElementDataTypeFloat16:
		return "float16"
	case ort.ONNXTensorElementDataTypeDouble:
		return "float64"
	case ort.ONNXTensorElementDataTypeUint32:
		return "uint32"
	case ort.ONNXTensorElementDataTypeUint64:
		return "uint64"
	case ort.ONNXTensorElementDataTypeBFloat16:
		return "bfloat16"
	default:
		return fmt.Sprintf("Unknown(%d)", t)
	}
}

func main() {
	flag.Parse()

	if *modelPath == "" {
		fmt.Fprintf(os.Stderr, "Error: -f flag is required\n\n")
		fmt.Fprintf(os.Stderr, "Usage: %s -f <model_path>\n", os.Args[0])
		flag.PrintDefaults()
		os.Exit(1)
	}

	if err := run(*modelPath); err != nil {
		log.Fatalf("Error: %v", err)
	}
}
