package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log"
	"math"
	"os"

	ort "github.com/benedoc-inc/onnxer/onnxruntime"
)

var (
	modelPath = flag.String("f", "", "path to ONNX model file (required)")
)

func run(ctx context.Context, modelPath string, texts []string) error {
	libraryPath := os.Getenv("ONNXRUNTIME_LIB_PATH")
	if libraryPath == "" {
		return errors.New("ONNXRUNTIME_LIB_PATH environment variable not set")
	}

	runtime, err := ort.NewRuntime(libraryPath, 23)
	if err != nil {
		return fmt.Errorf("failed to create runtime: %w", err)
	}
	defer runtime.Close()

	env, err := runtime.NewEnv("string-tensor-example", ort.LoggingLevelWarning)
	if err != nil {
		return fmt.Errorf("failed to create environment: %w", err)
	}
	defer env.Close()

	modelFile, err := os.Open(modelPath)
	if err != nil {
		return fmt.Errorf("failed to open model file: %w", err)
	}
	defer modelFile.Close()

	session, err := runtime.NewSessionFromReader(env, modelFile, &ort.SessionOptions{
		IntraOpNumThreads: 1,
	})
	if err != nil {
		return fmt.Errorf("failed to create session: %w", err)
	}
	defer session.Close()

	inputNames := session.InputNames()
	outputNames := session.OutputNames()

	fmt.Printf("Input names:  %v\n", inputNames)
	fmt.Printf("Output names: %v\n", outputNames)

	// Create a string tensor from the input texts
	shape := []int64{int64(len(texts))}
	stringTensor, err := runtime.NewStringTensorValue(texts, shape)
	if err != nil {
		return fmt.Errorf("failed to create string tensor: %w", err)
	}
	defer stringTensor.Close()

	fmt.Printf("\nInput strings: %v\n", texts)

	// Verify the tensor is correctly created by reading it back
	readBack, readShape, err := ort.GetStringTensorData(stringTensor)
	if err != nil {
		return fmt.Errorf("failed to read back string tensor: %w", err)
	}
	fmt.Printf("Read back:     %v (shape: %v)\n", readBack, readShape)

	// Run inference
	outputs, err := session.Run(ctx, map[string]*ort.Value{
		inputNames[0]: stringTensor,
	})
	if err != nil {
		return fmt.Errorf("failed to run inference: %w", err)
	}

	// Process outputs
	output := outputs[outputNames[0]]
	defer output.Close()

	valueType, err := output.GetValueType()
	if err != nil {
		return fmt.Errorf("failed to get output type: %w", err)
	}

	switch valueType {
	case ort.ONNXTypeTensor:
		// Try float32 output (common for classification models)
		data, outShape, err := ort.GetTensorData[float32](output)
		if err != nil {
			return fmt.Errorf("failed to get output tensor data: %w", err)
		}

		fmt.Printf("\nOutput shape: %v\n", outShape)
		fmt.Println("\nResults:")
		if len(outShape) == 2 {
			numClasses := int(outShape[1])
			for i, text := range texts {
				scores := data[i*numClasses : (i+1)*numClasses]
				probs := softmax(scores)
				maxIdx := argmax(probs)
				fmt.Printf("  %q -> class %d (%.2f%%)\n", text, maxIdx, probs[maxIdx]*100)
			}
		} else {
			fmt.Printf("  Raw output: %v\n", data)
		}
	default:
		fmt.Printf("Output type: %d\n", valueType)
	}

	return nil
}

func softmax(logits []float32) []float32 {
	maxVal := logits[0]
	for _, v := range logits[1:] {
		if v > maxVal {
			maxVal = v
		}
	}
	var sum float32
	probs := make([]float32, len(logits))
	for i, v := range logits {
		probs[i] = float32(math.Exp(float64(v - maxVal)))
		sum += probs[i]
	}
	for i := range probs {
		probs[i] /= sum
	}
	return probs
}

func argmax(vals []float32) int {
	maxIdx := 0
	for i, v := range vals {
		if v > vals[maxIdx] {
			maxIdx = i
		}
	}
	return maxIdx
}

func main() {
	flag.Parse()

	if *modelPath == "" {
		fmt.Fprintf(os.Stderr, "Error: -f flag is required\n\n")
		fmt.Fprintf(os.Stderr, "Usage: %s -f <model_path> \"text1\" \"text2\" ...\n", os.Args[0])
		flag.PrintDefaults()
		os.Exit(1)
	}

	texts := flag.Args()
	if len(texts) == 0 {
		texts = []string{"This is a test sentence", "Hello world"}
	}

	ctx := context.Background()
	if err := run(ctx, *modelPath, texts); err != nil {
		log.Fatalf("Error: %v", err)
	}
}
