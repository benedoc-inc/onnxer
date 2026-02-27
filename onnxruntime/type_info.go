package onnxruntime

import (
	"fmt"

	"github.com/benedoc-inc/onnxer/onnxruntime/internal/api"
)

// TensorTypeInfo describes the element type and shape of a tensor.
type TensorTypeInfo struct {
	ElementType ONNXTensorElementDataType
	Shape       []int64
}

// InputInfo describes a model input's name, type, and tensor info.
type InputInfo struct {
	Name       string
	Type       ONNXType
	TensorInfo *TensorTypeInfo // non-nil when Type == ONNXTypeTensor
}

// OutputInfo describes a model output's name, type, and tensor info.
type OutputInfo struct {
	Name       string
	Type       ONNXType
	TensorInfo *TensorTypeInfo // non-nil when Type == ONNXTypeTensor
}

// GetInputInfo returns complete type information for all model inputs.
// A single call returns structured info — no multi-step chain needed.
func (s *Session) GetInputInfo() ([]InputInfo, error) {
	if s.ptr == 0 {
		return nil, ErrSessionClosed
	}

	count := len(s.inputNames)
	infos := make([]InputInfo, count)

	for i := 0; i < count; i++ {
		info, err := s.getTypeInfo(true, i)
		if err != nil {
			return nil, fmt.Errorf("failed to get input type info at index %d: %w", i, err)
		}
		infos[i] = InputInfo{
			Name:       s.inputNames[i],
			Type:       info.onnxType,
			TensorInfo: info.tensorInfo,
		}
	}

	return infos, nil
}

// GetOutputInfo returns complete type information for all model outputs.
// A single call returns structured info — no multi-step chain needed.
func (s *Session) GetOutputInfo() ([]OutputInfo, error) {
	if s.ptr == 0 {
		return nil, ErrSessionClosed
	}

	count := len(s.outputNames)
	infos := make([]OutputInfo, count)

	for i := 0; i < count; i++ {
		info, err := s.getTypeInfo(false, i)
		if err != nil {
			return nil, fmt.Errorf("failed to get output type info at index %d: %w", i, err)
		}
		infos[i] = OutputInfo{
			Name:       s.outputNames[i],
			Type:       info.onnxType,
			TensorInfo: info.tensorInfo,
		}
	}

	return infos, nil
}

type rawTypeInfo struct {
	onnxType   ONNXType
	tensorInfo *TensorTypeInfo
}

func (s *Session) getTypeInfo(isInput bool, index int) (*rawTypeInfo, error) {
	var typeInfoPtr api.OrtTypeInfo
	var status api.OrtStatus

	if isInput {
		status = s.runtime.apiFuncs.SessionGetInputTypeInfo(s.ptr, uintptr(index), &typeInfoPtr)
	} else {
		status = s.runtime.apiFuncs.SessionGetOutputTypeInfo(s.ptr, uintptr(index), &typeInfoPtr)
	}
	if err := s.runtime.statusError(status); err != nil {
		return nil, err
	}
	defer s.runtime.apiFuncs.ReleaseTypeInfo(typeInfoPtr)

	// Get ONNX type
	var onnxType ONNXType
	status = s.runtime.apiFuncs.GetOnnxTypeFromTypeInfo(typeInfoPtr, &onnxType)
	if err := s.runtime.statusError(status); err != nil {
		return nil, fmt.Errorf("failed to get ONNX type: %w", err)
	}

	result := &rawTypeInfo{onnxType: onnxType}

	// If it's a tensor, extract tensor-specific info
	if onnxType == ONNXTypeTensor {
		var tensorInfoPtr api.OrtTensorTypeAndShapeInfo
		status = s.runtime.apiFuncs.CastTypeInfoToTensorInfo(typeInfoPtr, &tensorInfoPtr)
		if err := s.runtime.statusError(status); err != nil {
			return nil, fmt.Errorf("failed to cast to tensor info: %w", err)
		}
		// Note: tensorInfoPtr is owned by typeInfoPtr — do NOT release separately

		var elemType ONNXTensorElementDataType
		status = s.runtime.apiFuncs.GetTensorElementType(tensorInfoPtr, &elemType)
		if err := s.runtime.statusError(status); err != nil {
			return nil, fmt.Errorf("failed to get element type: %w", err)
		}

		var dimCount uintptr
		status = s.runtime.apiFuncs.GetDimensionsCount(tensorInfoPtr, &dimCount)
		if err := s.runtime.statusError(status); err != nil {
			return nil, fmt.Errorf("failed to get dimensions count: %w", err)
		}

		dims := make([]int64, dimCount)
		if dimCount > 0 {
			status = s.runtime.apiFuncs.GetDimensions(tensorInfoPtr, &dims[0], dimCount)
			if err := s.runtime.statusError(status); err != nil {
				return nil, fmt.Errorf("failed to get dimensions: %w", err)
			}
		}

		result.tensorInfo = &TensorTypeInfo{
			ElementType: elemType,
			Shape:       dims,
		}
	}

	return result, nil
}
