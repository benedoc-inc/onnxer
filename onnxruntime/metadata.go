package onnxruntime

import (
	"fmt"
	"unsafe"

	"github.com/benedoc-inc/onnxer/internal/cstrings"
	"github.com/benedoc-inc/onnxer/onnxruntime/internal/api"
)

// ModelMetadata contains metadata about an ONNX model.
// This is an immutable snapshot — no Destroy() needed.
type ModelMetadata struct {
	ProducerName   string
	GraphName      string
	Domain         string
	Description    string
	Version        int64
	CustomMetadata map[string]string
}

// GetModelMetadata retrieves the model metadata from the session.
// Returns an immutable snapshot of all metadata fields.
func (s *Session) GetModelMetadata() (*ModelMetadata, error) {
	if s.ptr == 0 {
		return nil, ErrSessionClosed
	}

	if s.runtime.allocator == nil {
		return nil, fmt.Errorf("allocator not initialized")
	}

	var metadataPtr api.OrtModelMetadata
	status := s.runtime.apiFuncs.SessionGetModelMetadata(s.ptr, &metadataPtr)
	if err := s.runtime.statusError(status); err != nil {
		return nil, fmt.Errorf("failed to get model metadata: %w", err)
	}
	defer s.runtime.apiFuncs.ReleaseModelMetadata(metadataPtr)

	alloc := s.runtime.allocator

	result := &ModelMetadata{}

	// Producer name
	var namePtr *byte
	status = s.runtime.apiFuncs.ModelMetadataGetProducerName(metadataPtr, alloc.ptr, &namePtr)
	if err := s.runtime.statusError(status); err != nil {
		return nil, fmt.Errorf("failed to get producer name: %w", err)
	}
	result.ProducerName = cstrings.CStringToString(namePtr)
	alloc.free(unsafe.Pointer(namePtr))

	// Graph name
	status = s.runtime.apiFuncs.ModelMetadataGetGraphName(metadataPtr, alloc.ptr, &namePtr)
	if err := s.runtime.statusError(status); err != nil {
		return nil, fmt.Errorf("failed to get graph name: %w", err)
	}
	result.GraphName = cstrings.CStringToString(namePtr)
	alloc.free(unsafe.Pointer(namePtr))

	// Domain
	status = s.runtime.apiFuncs.ModelMetadataGetDomain(metadataPtr, alloc.ptr, &namePtr)
	if err := s.runtime.statusError(status); err != nil {
		return nil, fmt.Errorf("failed to get domain: %w", err)
	}
	result.Domain = cstrings.CStringToString(namePtr)
	alloc.free(unsafe.Pointer(namePtr))

	// Description
	status = s.runtime.apiFuncs.ModelMetadataGetDescription(metadataPtr, alloc.ptr, &namePtr)
	if err := s.runtime.statusError(status); err != nil {
		return nil, fmt.Errorf("failed to get description: %w", err)
	}
	result.Description = cstrings.CStringToString(namePtr)
	alloc.free(unsafe.Pointer(namePtr))

	// Version
	status = s.runtime.apiFuncs.ModelMetadataGetVersion(metadataPtr, &result.Version)
	if err := s.runtime.statusError(status); err != nil {
		return nil, fmt.Errorf("failed to get version: %w", err)
	}

	// Custom metadata keys
	var keysPtr **byte
	var numKeys int64
	status = s.runtime.apiFuncs.ModelMetadataGetCustomMetadataMapKeys(metadataPtr, alloc.ptr, &keysPtr, &numKeys)
	if err := s.runtime.statusError(status); err != nil {
		return nil, fmt.Errorf("failed to get custom metadata keys: %w", err)
	}

	result.CustomMetadata = make(map[string]string, numKeys)
	if numKeys > 0 {
		keyPtrs := unsafe.Slice(keysPtr, numKeys)
		for i := int64(0); i < numKeys; i++ {
			key := cstrings.CStringToString(keyPtrs[i])
			alloc.free(unsafe.Pointer(keyPtrs[i]))

			// Look up value for this key
			keyBytes := append([]byte(key), 0)
			var valuePtr *byte
			status = s.runtime.apiFuncs.ModelMetadataLookupCustomMetadataMap(metadataPtr, alloc.ptr, &keyBytes[0], &valuePtr)
			if err := s.runtime.statusError(status); err != nil {
				return nil, fmt.Errorf("failed to get custom metadata value for key %q: %w", key, err)
			}
			result.CustomMetadata[key] = cstrings.CStringToString(valuePtr)
			alloc.free(unsafe.Pointer(valuePtr))
		}
		// Free the keys array itself
		alloc.free(unsafe.Pointer(keysPtr))
	}

	return result, nil
}
