package onnxruntime

import (
	"bytes"
	"fmt"
	"io"
	"slices"
)

// NewSessionWithProviderFallback creates a session trying each provider in order.
// It returns the session and the name of the provider that was used.
// If all requested providers fail, it falls back to CPUExecutionProvider.
//
// The modelReader is read once and buffered in memory so multiple provider
// attempts can be made.
//
// Example:
//
//	session, provider, err := runtime.NewSessionWithProviderFallback(env, modelReader, nil,
//	    ExecutionProvider{Name: "CUDAExecutionProvider", Options: map[string]string{"device_id": "0"}},
//	    ExecutionProvider{Name: "CoreMLExecutionProvider"},
//	)
//	fmt.Println("Using provider:", provider)
func (r *Runtime) NewSessionWithProviderFallback(env *Env, modelReader io.Reader, baseOptions *SessionOptions, providers ...ExecutionProvider) (*Session, string, error) {
	if baseOptions == nil {
		baseOptions = &SessionOptions{}
	}

	// Buffer the model data so we can retry with different providers
	modelData, err := io.ReadAll(modelReader)
	if err != nil {
		return nil, "", fmt.Errorf("failed to read model data: %w", err)
	}

	// Check which providers are actually available
	available, err := r.GetAvailableProviders()
	if err != nil {
		return nil, "", err
	}

	// Try each requested provider in order
	for _, provider := range providers {
		if !slices.Contains(available, provider.Name) {
			continue
		}

		opts := *baseOptions
		opts.ExecutionProviders = []ExecutionProvider{provider}

		session, err := r.NewSessionFromReader(env, bytes.NewReader(modelData), &opts)
		if err != nil {
			continue
		}
		return session, provider.Name, nil
	}

	// Fall back to CPU
	opts := *baseOptions
	opts.ExecutionProviders = nil
	session, err := r.NewSessionFromReader(env, bytes.NewReader(modelData), &opts)
	if err != nil {
		return nil, "", err
	}
	return session, "CPUExecutionProvider", nil
}
