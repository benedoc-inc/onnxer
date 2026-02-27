package onnxruntime

import (
	"fmt"
	"unsafe"

	"github.com/benedoc-inc/onnxer/onnxruntime/internal/api"
)

// LoraAdapter represents a loaded LoRA (Low-Rank Adaptation) adapter
// that can be applied to inference sessions at runtime.
//
// LoRA adapters allow fine-tuned model behavior without modifying the
// base model weights. Multiple adapters can be loaded and switched
// between runs.
//
// Example:
//
//	adapter, err := runtime.LoadLoraAdapterFromFile("adapter.onnx_adapter")
//	if err != nil {
//	    log.Fatal(err)
//	}
//	defer adapter.Close()
//
//	outputs, err := session.Run(ctx, inputs, ort.WithLoraAdapters(adapter))
type LoraAdapter struct {
	ptr     api.OrtLoraAdapter
	runtime *Runtime
}

// LoadLoraAdapterFromFile loads a LoRA adapter from a file path.
func (r *Runtime) LoadLoraAdapterFromFile(path string) (*LoraAdapter, error) {
	pathBytes := append([]byte(path), 0)
	var adapterPtr api.OrtLoraAdapter

	status := r.apiFuncs.CreateLoraAdapter(&pathBytes[0], r.allocator.ptr, &adapterPtr)
	if err := r.statusError(status); err != nil {
		return nil, fmt.Errorf("failed to load LoRA adapter from %q: %w", path, err)
	}

	return &LoraAdapter{
		ptr:     adapterPtr,
		runtime: r,
	}, nil
}

// LoadLoraAdapterFromBytes loads a LoRA adapter from in-memory data.
func (r *Runtime) LoadLoraAdapterFromBytes(data []byte) (*LoraAdapter, error) {
	if len(data) == 0 {
		return nil, fmt.Errorf("LoRA adapter data cannot be empty")
	}

	var adapterPtr api.OrtLoraAdapter
	status := r.apiFuncs.CreateLoraAdapterFromArray(
		unsafe.Pointer(&data[0]),
		uintptr(len(data)),
		r.allocator.ptr,
		&adapterPtr,
	)
	if err := r.statusError(status); err != nil {
		return nil, fmt.Errorf("failed to load LoRA adapter from bytes: %w", err)
	}

	return &LoraAdapter{
		ptr:     adapterPtr,
		runtime: r,
	}, nil
}

// Close releases the LoRA adapter resources.
// It is safe to call Close multiple times.
func (a *LoraAdapter) Close() {
	if a.ptr != 0 && a.runtime != nil && a.runtime.apiFuncs != nil {
		a.runtime.apiFuncs.ReleaseLoraAdapter(a.ptr)
		a.ptr = 0
	}
}

// WithLoraAdapters applies one or more LoRA adapters to a single inference run.
// The adapters are activated only for this run and do not affect other runs.
func WithLoraAdapters(adapters ...*LoraAdapter) RunOption {
	return func(c *runConfig) {
		c.loraAdapters = adapters
	}
}
