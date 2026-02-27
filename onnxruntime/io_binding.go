package onnxruntime

import (
	"context"
	"fmt"
	"runtime"
	"unsafe"

	"github.com/benedoc-inc/onnxer/onnxruntime/internal/api"
)

// MemoryInfo describes memory allocation properties.
// Use Runtime.NewCPUMemoryInfo() to create one.
type MemoryInfo struct {
	ptr     api.OrtMemoryInfo
	runtime *Runtime
}

// NewCPUMemoryInfo creates a MemoryInfo for CPU memory.
func (r *Runtime) NewCPUMemoryInfo() (*MemoryInfo, error) {
	mi, err := r.createCPUMemoryInfo(allocatorTypeDevice, memTypeCPU)
	if err != nil {
		return nil, err
	}
	return &MemoryInfo{ptr: mi.ptr, runtime: r}, nil
}

// Close releases the memory info resources.
func (mi *MemoryInfo) Close() {
	if mi.ptr != 0 && mi.runtime != nil && mi.runtime.apiFuncs != nil {
		mi.runtime.apiFuncs.ReleaseMemoryInfo(mi.ptr)
		mi.ptr = 0
	}
}

// IoBinding enables pre-binding of inputs and outputs for optimized inference.
// This avoids host-device copies between repeated runs with the same input/output shapes.
//
// An IoBinding is NOT safe for concurrent use. Close it when done to release resources.
type IoBinding struct {
	ptr     api.OrtIoBinding
	session *Session
}

// NewIoBinding creates a new IO binding for this session.
func (s *Session) NewIoBinding() (*IoBinding, error) {
	if s.ptr == 0 {
		return nil, ErrSessionClosed
	}

	var bindingPtr api.OrtIoBinding
	status := s.runtime.apiFuncs.CreateIoBinding(s.ptr, &bindingPtr)
	if err := s.runtime.statusError(status); err != nil {
		return nil, fmt.Errorf("failed to create IO binding: %w", err)
	}

	b := &IoBinding{
		ptr:     bindingPtr,
		session: s,
	}
	runtime.AddCleanup(b, func(_ struct{}) { b.Close() }, struct{}{})
	return b, nil
}

// BindInput binds an input tensor to the given name.
func (b *IoBinding) BindInput(name string, value *Value) error {
	nameBytes := append([]byte(name), 0)
	status := b.session.runtime.apiFuncs.BindInput(b.ptr, &nameBytes[0], value.ptr)
	if err := b.session.runtime.statusError(status); err != nil {
		return fmt.Errorf("failed to bind input %q: %w", name, err)
	}
	return nil
}

// BindOutput binds a pre-allocated output tensor to the given name.
func (b *IoBinding) BindOutput(name string, value *Value) error {
	nameBytes := append([]byte(name), 0)
	status := b.session.runtime.apiFuncs.BindOutput(b.ptr, &nameBytes[0], value.ptr)
	if err := b.session.runtime.statusError(status); err != nil {
		return fmt.Errorf("failed to bind output %q: %w", name, err)
	}
	return nil
}

// BindOutputToDevice binds an output to be allocated on the specified device.
// ONNX Runtime will allocate the output tensor on the device described by memInfo.
func (b *IoBinding) BindOutputToDevice(name string, memInfo *MemoryInfo) error {
	nameBytes := append([]byte(name), 0)
	status := b.session.runtime.apiFuncs.BindOutputToDevice(b.ptr, &nameBytes[0], memInfo.ptr)
	if err := b.session.runtime.statusError(status); err != nil {
		return fmt.Errorf("failed to bind output %q to device: %w", name, err)
	}
	return nil
}

// Run executes inference using the bound inputs and outputs.
// Context cancellation is supported — if ctx is cancelled, the run will be terminated.
func (b *IoBinding) Run(ctx context.Context) error {
	r := b.session.runtime

	runOptions, cleanup, err := b.session.createRunOptionsForCtx(ctx)
	if err != nil {
		return err
	}
	defer cleanup()

	status := r.apiFuncs.RunWithBinding(b.session.ptr, runOptions, b.ptr)
	if err := r.statusError(status); err != nil {
		return fmt.Errorf("failed to run with binding: %w", err)
	}
	return nil
}

// GetOutputValues returns the bound output values as a map from output name to Value.
func (b *IoBinding) GetOutputValues() (map[string]*Value, error) {
	r := b.session.runtime

	if r.allocator == nil {
		return nil, fmt.Errorf("allocator not initialized")
	}

	var valuesPtr *api.OrtValue
	var valueCount uintptr
	status := r.apiFuncs.GetBoundOutputValues(b.ptr, r.allocator.ptr, &valuesPtr, &valueCount)
	if err := r.statusError(status); err != nil {
		return nil, fmt.Errorf("failed to get bound output values: %w", err)
	}

	if valueCount == 0 {
		return map[string]*Value{}, nil
	}

	// Binding order matches session output order
	values := unsafe.Slice(valuesPtr, valueCount)
	result := make(map[string]*Value, valueCount)
	for i := uintptr(0); i < valueCount; i++ {
		result[b.session.outputNames[i]] = r.newValueFromPtr(values[i])
	}

	return result, nil
}

// ClearInputs clears all bound inputs.
func (b *IoBinding) ClearInputs() {
	b.session.runtime.apiFuncs.ClearBoundInputs(b.ptr)
}

// ClearOutputs clears all bound outputs.
func (b *IoBinding) ClearOutputs() {
	b.session.runtime.apiFuncs.ClearBoundOutputs(b.ptr)
}

// Close releases the IO binding resources.
func (b *IoBinding) Close() {
	if b.ptr != 0 && b.session != nil && b.session.runtime != nil && b.session.runtime.apiFuncs != nil {
		b.session.runtime.apiFuncs.ReleaseIoBinding(b.ptr)
		b.ptr = 0
	}
}
