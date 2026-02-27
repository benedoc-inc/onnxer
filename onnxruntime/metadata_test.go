package onnxruntime

import (
	"testing"
)

func TestGetModelMetadata(t *testing.T) {
	runtime := newTestRuntime(t)
	session := newTestSession(t, runtime)

	metadata, err := session.GetModelMetadata()
	if err != nil {
		t.Fatalf("Failed to get model metadata: %v", err)
	}

	// The test model should have some metadata fields (may be empty strings)
	if metadata == nil {
		t.Fatal("Metadata should not be nil")
	}

	// CustomMetadata should be non-nil (even if empty)
	if metadata.CustomMetadata == nil {
		t.Error("CustomMetadata map should not be nil")
	}

	t.Logf("ProducerName: %q", metadata.ProducerName)
	t.Logf("GraphName: %q", metadata.GraphName)
	t.Logf("Domain: %q", metadata.Domain)
	t.Logf("Description: %q", metadata.Description)
	t.Logf("Version: %d", metadata.Version)
	t.Logf("CustomMetadata: %v", metadata.CustomMetadata)
}

func TestGetModelMetadataClosedSession(t *testing.T) {
	runtime := newTestRuntime(t)
	session := newTestSession(t, runtime)
	session.Close()

	_, err := session.GetModelMetadata()
	if err == nil {
		t.Error("Expected error when getting metadata from closed session")
	}
}
