package onnxruntime

import (
	"fmt"
	"unsafe"

	"github.com/benedoc-inc/onnxer/onnxruntime/internal/api"
)

// NewStringTensorValue creates a new string tensor value from a slice of strings.
// The shape defines the tensor dimensions. Everything is Value — no separate string tensor type.
func (r *Runtime) NewStringTensorValue(data []string, shape []int64) (*Value, error) {
	if len(data) == 0 {
		return nil, fmt.Errorf("data cannot be empty")
	}

	if r.allocator == nil {
		return nil, fmt.Errorf("allocator not initialized")
	}

	// Create an empty tensor with string type using the allocator
	var valuePtr api.OrtValue
	var shapePtr *int64
	if len(shape) > 0 {
		shapePtr = &shape[0]
	}

	status := r.apiFuncs.CreateTensorAsOrtValue(
		r.allocator.ptr,
		shapePtr,
		uintptr(len(shape)),
		ONNXTensorElementDataTypeString,
		&valuePtr,
	)
	if err := r.statusError(status); err != nil {
		return nil, fmt.Errorf("failed to create string tensor: %w", err)
	}

	// Convert Go strings to C strings and fill the tensor
	cstrings := make([]*byte, len(data))
	// Keep the backing byte slices alive until FillStringTensor returns
	backingSlices := make([][]byte, len(data))
	for i, s := range data {
		b := append([]byte(s), 0)
		backingSlices[i] = b
		cstrings[i] = &b[0]
	}

	status = r.apiFuncs.FillStringTensor(valuePtr, &cstrings[0], uintptr(len(data)))
	if err := r.statusError(status); err != nil {
		r.apiFuncs.ReleaseValue(valuePtr)
		return nil, fmt.Errorf("failed to fill string tensor: %w", err)
	}

	// Keep backingSlices alive past FillStringTensor
	_ = backingSlices

	return r.newValueFromPtr(valuePtr), nil
}

// GetStringTensorData extracts all string data and shape from a string tensor Value.
// The returned strings are copies — safe to use after the Value is closed.
func GetStringTensorData(v *Value) ([]string, []int64, error) {
	// Verify this is a string tensor
	elemType, err := v.GetTensorElementType()
	if err != nil {
		return nil, nil, fmt.Errorf("failed to get element type: %w", err)
	}
	if elemType != ONNXTensorElementDataTypeString {
		return nil, nil, fmt.Errorf("element type mismatch: expected string (%d), got %d", ONNXTensorElementDataTypeString, elemType)
	}

	shape, err := v.GetTensorShape()
	if err != nil {
		return nil, nil, fmt.Errorf("failed to get shape: %w", err)
	}

	count, err := v.GetTensorElementCount()
	if err != nil {
		return nil, nil, fmt.Errorf("failed to get element count: %w", err)
	}

	if count == 0 {
		return []string{}, shape, nil
	}

	// Get total data length
	var totalLen uintptr
	status := v.runtime.apiFuncs.GetStringTensorDataLength(v.ptr, &totalLen)
	if err := v.runtime.statusError(status); err != nil {
		return nil, nil, fmt.Errorf("failed to get string tensor data length: %w", err)
	}

	// Allocate buffers
	buf := make([]byte, totalLen)
	offsets := make([]uintptr, count)

	var bufPtr unsafe.Pointer
	if totalLen > 0 {
		bufPtr = unsafe.Pointer(&buf[0])
	}

	status = v.runtime.apiFuncs.GetStringTensorContent(v.ptr, bufPtr, totalLen, &offsets[0], uintptr(count))
	if err := v.runtime.statusError(status); err != nil {
		return nil, nil, fmt.Errorf("failed to get string tensor content: %w", err)
	}

	// Parse strings from the contiguous buffer using offsets
	result := make([]string, count)
	for i := 0; i < count; i++ {
		start := offsets[i]
		var end uintptr
		if i+1 < count {
			end = offsets[i+1]
		} else {
			end = totalLen
		}
		result[i] = string(buf[start:end])
	}

	return result, shape, nil
}

// IsTensor returns whether this Value is a tensor (as opposed to a sequence, map, etc.).
func (v *Value) IsTensor() (bool, error) {
	var out int32
	status := v.runtime.apiFuncs.IsTensor(v.ptr, &out)
	if err := v.runtime.statusError(status); err != nil {
		return false, fmt.Errorf("failed to check if value is tensor: %w", err)
	}
	return out != 0, nil
}

// SetStringTensorElement sets the string at the given flat index in a string tensor.
func (v *Value) SetStringTensorElement(index int, s string) error {
	b := append([]byte(s), 0)
	status := v.runtime.apiFuncs.FillStringTensorElement(v.ptr, &b[0], uintptr(index))
	if err := v.runtime.statusError(status); err != nil {
		return fmt.Errorf("failed to set string tensor element: %w", err)
	}
	return nil
}

// GetStringTensorElement returns the string at the given flat index in a string tensor.
func (v *Value) GetStringTensorElement(index int) (string, error) {
	// Get element length
	var length uintptr
	status := v.runtime.apiFuncs.GetStringTensorElementLength(v.ptr, uintptr(index), &length)
	if err := v.runtime.statusError(status); err != nil {
		return "", fmt.Errorf("failed to get string tensor element length: %w", err)
	}

	// Read the element
	buf := make([]byte, length)
	var bufPtr unsafe.Pointer
	if length > 0 {
		bufPtr = unsafe.Pointer(&buf[0])
	}

	status = v.runtime.apiFuncs.GetStringTensorElement(v.ptr, length, uintptr(index), bufPtr)
	if err := v.runtime.statusError(status); err != nil {
		return "", fmt.Errorf("failed to get string tensor element: %w", err)
	}

	return string(buf), nil
}
