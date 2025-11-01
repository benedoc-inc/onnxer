package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log"
	"math"
	"os"

	ort "github.com/shota3506/onnxruntime-purego/onnxruntime"
	"github.com/sugarme/tokenizer/pretrained"
)

var (
	modelPath     = flag.String("f", "", "path to ONNX model file (required)")
	tokenizerPath = flag.String("tokenizer", "", "path to tokenizer.json file (required)")
	maxLength     = flag.Int("max-length", 128, "maximum sequence length")
)

// Sentiment labels for twitter-xlm-roberta-base-sentiment
var sentimentLabels = []string{"negative", "neutral", "positive"}

// softmax converts logits to probabilities
func softmax(logits []float32) []float32 {
	maxLogit := float32(-math.MaxFloat32)
	for _, v := range logits {
		if v > maxLogit {
			maxLogit = v
		}
	}

	var sum float32
	probs := make([]float32, len(logits))
	for i, v := range logits {
		probs[i] = float32(math.Exp(float64(v - maxLogit)))
		sum += probs[i]
	}

	for i := range probs {
		probs[i] /= sum
	}

	return probs
}

func run(ctx context.Context, modelPath, tokenizerPath, text string, maxLength int) error {
	// Load tokenizer
	fmt.Printf("Loading tokenizer: %s\n", tokenizerPath)
	tk, err := pretrained.FromFile(tokenizerPath)
	if err != nil {
		return fmt.Errorf("failed to load tokenizer: %w", err)
	}

	// Tokenize input text
	fmt.Printf("Processing text: \"%s\"\n", text)
	encoding, err := tk.EncodeSingle(text)
	if err != nil {
		return fmt.Errorf("failed to tokenize text: %w", err)
	}

	// Get token IDs
	inputIds := encoding.GetIds()
	attentionMask := encoding.GetAttentionMask()

	// Pad or truncate to max_length
	if len(inputIds) > maxLength {
		inputIds = inputIds[:maxLength]
		attentionMask = attentionMask[:maxLength]
	}

	// Pad if needed
	for len(inputIds) < maxLength {
		inputIds = append(inputIds, 1) // PAD token ID for RoBERTa
		attentionMask = append(attentionMask, 0)
	}

	// Convert to int64
	inputIds64 := make([]int64, len(inputIds))
	attentionMask64 := make([]int64, len(attentionMask))

	for i := range inputIds {
		inputIds64[i] = int64(inputIds[i])
		attentionMask64[i] = int64(attentionMask[i])
	}

	fmt.Printf("Input IDs (first 10): %v...\n", inputIds64[:min(10, len(inputIds64))])

	// Initialize ONNX Runtime
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
	env, err := runtime.NewEnv("roberta-sentiment-example", ort.LoggingLevelWarning)
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

	// Create input tensors
	batchSize := int64(1)
	seqLength := int64(maxLength)

	inputIdsTensor, err := ort.NewTensorValue(runtime, inputIds64, []int64{batchSize, seqLength})
	if err != nil {
		return fmt.Errorf("failed to create input_ids tensor: %w", err)
	}
	defer inputIdsTensor.Close()

	attentionMaskTensor, err := ort.NewTensorValue(runtime, attentionMask64, []int64{batchSize, seqLength})
	if err != nil {
		return fmt.Errorf("failed to create attention_mask tensor: %w", err)
	}
	defer attentionMaskTensor.Close()

	// Prepare inputs - map to model's expected input names
	inputs := make(map[string]*ort.Value)
	for _, name := range inputNames {
		switch name {
		case "input_ids":
			inputs[name] = inputIdsTensor
		case "attention_mask":
			inputs[name] = attentionMaskTensor
		}
	}

	// Run inference
	outputs, err := session.Run(ctx, inputs)
	if err != nil {
		return fmt.Errorf("failed to run inference: %w", err)
	}

	// Process outputs
	output := outputs["logits"]
	defer output.Close()

	// Get the output logits (float32)
	logits, shape, err := ort.GetTensorData[float32](output)
	if err != nil {
		return fmt.Errorf("failed to get output tensor data: %w", err)
	}

	// For sentiment classification, the output is typically [batch_size, num_labels]
	if len(shape) == 2 && shape[0] == 1 {
		numLabels := int(shape[1])
		if numLabels != len(sentimentLabels) {
			fmt.Printf("  Warning: Expected %d labels but got %d\n", len(sentimentLabels), numLabels)
		}

		// Get logits for the first (and only) batch item
		batchLogits := logits[:numLabels]

		// Apply softmax to get probabilities
		probs := softmax(batchLogits)

		// Find the prediction with highest probability
		maxProb := float32(0)
		maxIdx := 0
		for i, prob := range probs {
			if prob > maxProb {
				maxProb = prob
				maxIdx = i
			}
		}

		fmt.Println("\nSentiment Analysis")
		fmt.Printf("Text: \"%s\"\n", text)
		fmt.Printf("Predicted Sentiment: %s (%.2f%% confidence)\n",
			sentimentLabels[maxIdx], maxProb*100)

		fmt.Println("All Scores:")
		for i, label := range sentimentLabels {
			fmt.Printf("  %s: %.2f%%\n", label, probs[i]*100)
		}
	}

	return nil
}

func main() {
	flag.Parse()

	if *modelPath == "" || *tokenizerPath == "" {
		fmt.Fprintf(os.Stderr, "Error: -f and -tokenizer flags are required\n\n")
		flag.Usage()
		os.Exit(1)
	}

	if flag.NArg() < 1 {
		fmt.Fprintf(os.Stderr, "Usage: %s -f <model_path> -tokenizer <tokenizer.json> \"<text>\"\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "Example: %s -f model.onnx -tokenizer tokenizer.json \"I love this movie!\"\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "Options:\n")
		flag.PrintDefaults()
		os.Exit(1)
	}

	text := flag.Arg(0)

	ctx := context.Background()
	if err := run(ctx, *modelPath, *tokenizerPath, text, *maxLength); err != nil {
		log.Fatalf("Error: %v", err)
	}
}
