package onnxruntime

import (
	"fmt"

	"github.com/benedoc-inc/onnxer/onnxruntime/internal/api"
)

// GetSequenceLength returns the number of elements in a sequence value.
func (v *Value) GetSequenceLength() (int, error) {
	var count uintptr
	status := v.runtime.apiFuncs.GetValueCount(v.ptr, &count)
	if err := v.runtime.statusError(status); err != nil {
		return 0, fmt.Errorf("failed to get sequence length: %w", err)
	}
	return int(count), nil
}

// GetSequenceValues extracts all elements from a sequence value.
// Each element is returned as a *Value that must be closed by the caller.
func (v *Value) GetSequenceValues() ([]*Value, error) {
	count, err := v.GetSequenceLength()
	if err != nil {
		return nil, err
	}

	if v.runtime.allocator == nil {
		return nil, fmt.Errorf("allocator not initialized")
	}

	values := make([]*Value, count)
	for i := 0; i < count; i++ {
		var elemPtr api.OrtValue
		status := v.runtime.apiFuncs.GetValue(v.ptr, int32(i), v.runtime.allocator.ptr, &elemPtr)
		if err := v.runtime.statusError(status); err != nil {
			// Clean up already-extracted values on error
			for j := 0; j < i; j++ {
				values[j].Close()
			}
			return nil, fmt.Errorf("failed to get sequence element at index %d: %w", i, err)
		}
		values[i] = v.runtime.newValueFromPtr(elemPtr)
	}

	return values, nil
}

// GetMapKeyValue extracts the keys and values from a map value.
// ONNX maps are represented as two tensors: one for keys and one for values.
// Both returned Values must be closed by the caller.
func (v *Value) GetMapKeyValue() (keys *Value, values *Value, err error) {
	if v.runtime.allocator == nil {
		return nil, nil, fmt.Errorf("allocator not initialized")
	}

	// Map has exactly 2 elements: index 0 = keys, index 1 = values
	var keysPtr api.OrtValue
	status := v.runtime.apiFuncs.GetValue(v.ptr, 0, v.runtime.allocator.ptr, &keysPtr)
	if err := v.runtime.statusError(status); err != nil {
		return nil, nil, fmt.Errorf("failed to get map keys: %w", err)
	}
	keysValue := v.runtime.newValueFromPtr(keysPtr)

	var valsPtr api.OrtValue
	status = v.runtime.apiFuncs.GetValue(v.ptr, 1, v.runtime.allocator.ptr, &valsPtr)
	if err := v.runtime.statusError(status); err != nil {
		keysValue.Close()
		return nil, nil, fmt.Errorf("failed to get map values: %w", err)
	}
	valsValue := v.runtime.newValueFromPtr(valsPtr)

	return keysValue, valsValue, nil
}
