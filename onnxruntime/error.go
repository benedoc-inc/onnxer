package onnxruntime

import (
	"fmt"
)

// RuntimeError represents an error returned from the ONNX Runtime C API.
type RuntimeError struct {
	Code    ErrorCode
	Message string
}

func (e *RuntimeError) Error() string {
	return fmt.Sprintf("onnxruntime error (%s): %s", errorCodeName(e.Code), e.Message)
}

// errorCodeName returns a human-readable name for an error code.
func errorCodeName(code ErrorCode) string {
	switch code {
	case ErrorCodeOK:
		return "OK"
	case ErrorCodeFail:
		return "Fail"
	case ErrorCodeInvalidArgument:
		return "InvalidArgument"
	case ErrorCodeNoSuchFile:
		return "NoSuchFile"
	case ErrorCodeNoModel:
		return "NoModel"
	case ErrorCodeEngineError:
		return "EngineError"
	case ErrorCodeRuntimeException:
		return "RuntimeException"
	case ErrorCodeInvalidProtobuf:
		return "InvalidProtobuf"
	case ErrorCodeModelLoaded:
		return "ModelLoaded"
	case ErrorCodeNotImplemented:
		return "NotImplemented"
	case ErrorCodeInvalidGraph:
		return "InvalidGraph"
	case ErrorCodeEPFail:
		return "EPFail"
	default:
		return fmt.Sprintf("ErrorCode(%d)", code)
	}
}
