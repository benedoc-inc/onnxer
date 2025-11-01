package main

import (
	"context"
	_ "embed"
	"errors"
	"flag"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"log"
	"math"
	"os"
	"sort"
	"strings"

	"github.com/nfnt/resize"
	ort "github.com/shota3506/onnxruntime-purego/onnxruntime"
)

var (
	modelPath = flag.String("f", "", "path to ONNX model file (required)")
)

//go:embed imagenet_classes.txt
var imagenetClassesData string

var (
	imagenetMean = []float32{0.485, 0.456, 0.406}
	imagenetStd  = []float32{0.229, 0.224, 0.225}
)

// imagenetClasses contains the 1000 ImageNet class labels, loaded from embedded file
var imagenetClasses []string

func init() {
	for line := range strings.SplitSeq(strings.TrimSpace(imagenetClassesData), "\n") {
		parts := strings.SplitN(line, ", ", 2)
		if len(parts) != 2 {
			continue
		}
		imagenetClasses = append(imagenetClasses, strings.ReplaceAll(parts[1], "_", " "))
	}
}

// getImageNetClassName returns the ImageNet class name for a given class ID
func getImageNetClassName(classID int) string {
	if classID < 0 || classID >= len(imagenetClasses) {
		return fmt.Sprintf("Unknown (ID: %d)", classID)
	}
	return imagenetClasses[classID]
}

// preprocessImage loads and preprocesses an image for ResNet
func preprocessImage(imagePath string) ([]float32, []int64, error) {
	// Open image file
	file, err := os.Open(imagePath)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to open image: %w", err)
	}
	defer file.Close()

	// Decode image
	img, _, err := image.Decode(file)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to decode image: %w", err)
	}

	// Resize to 224x224
	resized := resize.Resize(224, 224, img, resize.Bilinear)

	// Convert to float32 tensor with shape [1, 3, 224, 224]
	// and normalize using ImageNet mean and std
	data := make([]float32, 1*3*224*224)

	for y := range 224 {
		for x := range 224 {
			r, g, b, _ := resized.At(x, y).RGBA()

			// RGBA() returns 16-bit color values (0-65535)
			// Convert to 8-bit (0-255) then to 0-1 range
			rVal := float32(r>>8) / 255.0
			gVal := float32(g>>8) / 255.0
			bVal := float32(b>>8) / 255.0

			// Apply ImageNet normalization: (value - mean) / std
			rNorm := (rVal - imagenetMean[0]) / imagenetStd[0]
			gNorm := (gVal - imagenetMean[1]) / imagenetStd[1]
			bNorm := (bVal - imagenetMean[2]) / imagenetStd[2]

			// Store in CHW format (channels, height, width)
			idx := y*224 + x
			data[0*224*224+idx] = rNorm // R channel
			data[1*224*224+idx] = gNorm // G channel
			data[2*224*224+idx] = bNorm // B channel
		}
	}

	return data, []int64{1, 3, 224, 224}, nil
}

// softmax applies softmax function to convert logits to probabilities
func softmax(logits []float32) []float32 {
	probs := make([]float32, len(logits))

	// Find max for numerical stability
	maxLogit := logits[0]
	for _, v := range logits {
		if v > maxLogit {
			maxLogit = v
		}
	}

	// Compute exp(logit - max) and sum
	sum := float32(0.0)
	for i, v := range logits {
		x := float64(v - maxLogit)
		exp := float32(math.Exp(x))
		probs[i] = exp
		sum += exp
	}

	// Normalize
	for i := range probs {
		probs[i] /= sum
	}

	return probs
}

// ClassScore represents a class prediction with its score
type ClassScore struct {
	ClassID int
	Score   float32
}

func run(ctx context.Context, modelPath, imagePath string) error {
	// Load and preprocess image
	fmt.Printf("Loading image: %s\n", imagePath)
	inputData, inputShape, err := preprocessImage(imagePath)
	if err != nil {
		return fmt.Errorf("failed to preprocess image: %w", err)
	}

	libraryPath := os.Getenv("ONNXRUNTIME_LIB_PATH")
	if libraryPath == "" {
		return errors.New("ONNXRUNTIME_LIB_PATH environment variable not set")
	}

	// Create runtime
	runtime, err := ort.NewRuntime(libraryPath, 23)
	if err != nil {
		return fmt.Errorf("failed to create runtime: %w", err)
	}
	defer runtime.Close()

	// Create environment
	env, err := runtime.NewEnv("resnet-example", ort.LoggingLevelWarning)
	if err != nil {
		return fmt.Errorf("failed to create environment: %w", err)
	}
	defer env.Close()

	// Open model file
	fmt.Printf("Loading model: %s\n", modelPath)
	modelFile, err := os.Open(modelPath)
	if err != nil {
		return fmt.Errorf("failed to open model file: %w", err)
	}
	defer modelFile.Close()

	// Create session
	session, err := runtime.NewSessionFromReader(env, modelFile, &ort.SessionOptions{
		IntraOpNumThreads: 1,
	})
	if err != nil {
		return fmt.Errorf("failed to create session: %w", err)
	}
	defer session.Close()

	inputNames := session.InputNames()
	outputNames := session.OutputNames()

	fmt.Printf("Input names: %v\n", inputNames)
	fmt.Printf("Output names: %v\n", outputNames)

	// Create input tensor
	inputTensor, err := ort.NewTensorValue(runtime, inputData, inputShape)
	if err != nil {
		return fmt.Errorf("failed to create input tensor: %w", err)
	}
	defer inputTensor.Close()

	// Run inference using RunWithMap (computes all outputs by default)
	outputs, err := session.Run(ctx, map[string]*ort.Value{
		inputNames[0]: inputTensor,
	})
	if err != nil {
		return fmt.Errorf("failed to run inference: %w", err)
	}

	// Get the first output
	output := outputs[outputNames[0]]
	defer output.Close()

	logits, _, err := ort.GetTensorData[float32](output)
	if err != nil {
		return fmt.Errorf("failed to get output tensor data: %w", err)
	}

	// Apply softmax to get probabilities
	probs := softmax(logits)

	// Get top 10 predictions
	scores := make([]ClassScore, len(probs))
	for i, prob := range probs {
		scores[i] = ClassScore{
			ClassID: i,
			Score:   prob,
		}
	}

	// Sort by score descending
	sort.Slice(scores, func(i, j int) bool {
		return scores[i].Score > scores[j].Score
	})

	// Display top 10
	fmt.Println("\nImage Classification")
	for i := 0; i < 10 && i < len(scores); i++ {
		className := getImageNetClassName(scores[i].ClassID)
		fmt.Printf("%2d. %-30s (ID: %4d) - %.2f%%\n",
			i+1,
			className,
			scores[i].ClassID,
			scores[i].Score*100)
	}

	return nil
}

func main() {
	flag.Parse()

	if *modelPath == "" {
		fmt.Fprintf(os.Stderr, "Error: -f flag is required\n\n")
		flag.Usage()
		os.Exit(1)
	}

	if flag.NArg() < 1 {
		fmt.Fprintf(os.Stderr, "Usage: %s -f <model_path> <image_path>\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "Options:\n")
		flag.PrintDefaults()
		os.Exit(1)
	}

	imagePath := flag.Arg(0)

	ctx := context.Background()
	if err := run(ctx, *modelPath, imagePath); err != nil {
		log.Fatalf("Error: %v", err)
	}
}
