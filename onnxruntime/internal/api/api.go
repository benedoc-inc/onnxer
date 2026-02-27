package api

import "unsafe"

// OrtStatus is an opaque pointer to an ONNX Runtime status object.
type OrtStatus uintptr

// OrtEnv is an opaque pointer to an ONNX Runtime environment.
type OrtEnv uintptr

// OrtSession is an opaque pointer to an ONNX Runtime inference session.
type OrtSession uintptr

// OrtSessionOptions is an opaque pointer to ONNX Runtime session options.
type OrtSessionOptions uintptr

// OrtValue is an opaque pointer to an ONNX Runtime value (typically a tensor).
type OrtValue uintptr

// OrtAllocator is an opaque pointer to an ONNX Runtime memory allocator.
type OrtAllocator uintptr

// OrtMemoryInfo is an opaque pointer to ONNX Runtime memory information.
type OrtMemoryInfo uintptr

// OrtTensorTypeAndShapeInfo is an opaque pointer to ONNX Runtime tensor type and shape information.
type OrtTensorTypeAndShapeInfo uintptr

// OrtRunOptions is an opaque pointer to ONNX Runtime run options.
type OrtRunOptions uintptr

// OrtModelMetadata is an opaque pointer to ONNX Runtime model metadata.
type OrtModelMetadata uintptr

// OrtTypeInfo is an opaque pointer to ONNX Runtime type information.
type OrtTypeInfo uintptr

// OrtIoBinding is an opaque pointer to ONNX Runtime IO binding.
type OrtIoBinding uintptr

// OrtMapTypeInfo is an opaque pointer to ONNX Runtime map type information.
type OrtMapTypeInfo uintptr

// OrtSequenceTypeInfo is an opaque pointer to ONNX Runtime sequence type information.
type OrtSequenceTypeInfo uintptr

// OrtLoraAdapter is an opaque pointer to an ONNX Runtime LoRA adapter.
type OrtLoraAdapter uintptr

// OrtErrorCode represents error codes returned by the ONNX Runtime C API.
type OrtErrorCode int32

// OrtLoggingLevel represents logging verbosity levels for ONNX Runtime.
type OrtLoggingLevel int32

// ONNXType represents the type of an ONNX value.
type ONNXType int32

// ONNXTensorElementDataType represents the data type of tensor elements.
type ONNXTensorElementDataType int32

// OrtAllocatorType represents memory allocator types.
type OrtAllocatorType int32

// OrtMemType represents memory types for allocations.
type OrtMemType int32

// APIFuncs is an interface for ONNX Runtime C API functions.
type APIFuncs interface {
	// Status and error handling
	CreateStatus(OrtErrorCode, *byte) OrtStatus
	GetErrorCode(OrtStatus) OrtErrorCode
	GetErrorMessage(OrtStatus) unsafe.Pointer
	ReleaseStatus(OrtStatus)

	// Environment
	CreateEnv(OrtLoggingLevel, *byte, *OrtEnv) OrtStatus
	ReleaseEnv(OrtEnv)

	// Allocator
	GetAllocatorWithDefaultOptions(*OrtAllocator) OrtStatus
	AllocatorFree(OrtAllocator, unsafe.Pointer)

	// Memory info
	CreateCpuMemoryInfo(OrtAllocatorType, OrtMemType, *OrtMemoryInfo) OrtStatus
	ReleaseMemoryInfo(OrtMemoryInfo)

	// Telemetry
	EnableTelemetryEvents(OrtEnv) OrtStatus
	DisableTelemetryEvents(OrtEnv) OrtStatus

	// Session options
	CreateSessionOptions(*OrtSessionOptions) OrtStatus
	SetOptimizedModelFilePath(OrtSessionOptions, *byte) OrtStatus
	SetIntraOpNumThreads(OrtSessionOptions, int32) OrtStatus
	SetInterOpNumThreads(OrtSessionOptions, int32) OrtStatus
	SetSessionExecutionMode(OrtSessionOptions, int32) OrtStatus
	SetSessionGraphOptimizationLevel(OrtSessionOptions, int32) OrtStatus
	EnableCpuMemArena(OrtSessionOptions) OrtStatus
	DisableCpuMemArena(OrtSessionOptions) OrtStatus
	EnableMemPattern(OrtSessionOptions) OrtStatus
	DisableMemPattern(OrtSessionOptions) OrtStatus
	SetSessionLogSeverityLevel(OrtSessionOptions, int32) OrtStatus
	AddSessionConfigEntry(OrtSessionOptions, *byte, *byte) OrtStatus
	AddFreeDimensionOverrideByName(OrtSessionOptions, *byte, int64) OrtStatus
	SetDeterministicCompute(OrtSessionOptions, int32) OrtStatus
	EnableProfiling(OrtSessionOptions, *byte) OrtStatus
	DisableProfiling(OrtSessionOptions) OrtStatus
	SessionOptionsAppendExecutionProvider(OrtSessionOptions, *byte, **byte, **byte, uintptr) OrtStatus
	ReleaseSessionOptions(OrtSessionOptions)

	// Run options
	CreateRunOptions(*OrtRunOptions) OrtStatus
	ReleaseRunOptions(OrtRunOptions)
	RunOptionsSetTerminate(OrtRunOptions) OrtStatus
	RunOptionsUnsetTerminate(OrtRunOptions) OrtStatus
	RunOptionsSetRunTag(OrtRunOptions, *byte) OrtStatus
	AddRunConfigEntry(OrtRunOptions, *byte, *byte) OrtStatus
	RunOptionsAddActiveLoraAdapter(OrtRunOptions, OrtLoraAdapter) OrtStatus

	// Session
	CreateSession(OrtEnv, *byte, OrtSessionOptions, *OrtSession) OrtStatus
	CreateSessionFromArray(OrtEnv, unsafe.Pointer, uintptr, OrtSessionOptions, *OrtSession) OrtStatus
	SessionGetInputCount(OrtSession, *uintptr) OrtStatus
	SessionGetOutputCount(OrtSession, *uintptr) OrtStatus
	SessionGetInputName(OrtSession, uintptr, OrtAllocator, **byte) OrtStatus
	SessionGetOutputName(OrtSession, uintptr, OrtAllocator, **byte) OrtStatus
	Run(OrtSession, OrtRunOptions, **byte, *OrtValue, uintptr, **byte, uintptr, *OrtValue) OrtStatus
	ReleaseSession(OrtSession)

	// Profiling
	SessionEndProfiling(OrtSession, OrtAllocator, **byte) OrtStatus
	SessionGetProfilingStartTimeNs(OrtSession, *uint64) OrtStatus

	// LoRA adapters
	CreateLoraAdapter(*byte, OrtAllocator, *OrtLoraAdapter) OrtStatus
	CreateLoraAdapterFromArray(unsafe.Pointer, uintptr, OrtAllocator, *OrtLoraAdapter) OrtStatus
	ReleaseLoraAdapter(OrtLoraAdapter)

	// Build info
	GetBuildInfoString() unsafe.Pointer

	// Model metadata
	SessionGetModelMetadata(OrtSession, *OrtModelMetadata) OrtStatus
	ModelMetadataGetProducerName(OrtModelMetadata, OrtAllocator, **byte) OrtStatus
	ModelMetadataGetGraphName(OrtModelMetadata, OrtAllocator, **byte) OrtStatus
	ModelMetadataGetDomain(OrtModelMetadata, OrtAllocator, **byte) OrtStatus
	ModelMetadataGetDescription(OrtModelMetadata, OrtAllocator, **byte) OrtStatus
	ModelMetadataLookupCustomMetadataMap(OrtModelMetadata, OrtAllocator, *byte, **byte) OrtStatus
	ModelMetadataGetVersion(OrtModelMetadata, *int64) OrtStatus
	ReleaseModelMetadata(OrtModelMetadata)
	ModelMetadataGetCustomMetadataMapKeys(OrtModelMetadata, OrtAllocator, ***byte, *int64) OrtStatus

	// Type introspection
	SessionGetInputTypeInfo(OrtSession, uintptr, *OrtTypeInfo) OrtStatus
	SessionGetOutputTypeInfo(OrtSession, uintptr, *OrtTypeInfo) OrtStatus
	CastTypeInfoToTensorInfo(OrtTypeInfo, *OrtTensorTypeAndShapeInfo) OrtStatus
	GetOnnxTypeFromTypeInfo(OrtTypeInfo, *ONNXType) OrtStatus
	GetSymbolicDimensions(OrtTensorTypeAndShapeInfo, **byte, uintptr) OrtStatus
	ReleaseTypeInfo(OrtTypeInfo)

	// Tensor/Value operations
	CreateTensorAsOrtValue(OrtAllocator, *int64, uintptr, ONNXTensorElementDataType, *OrtValue) OrtStatus
	CreateTensorWithDataAsOrtValue(OrtMemoryInfo, unsafe.Pointer, uintptr, *int64, uintptr, ONNXTensorElementDataType, *OrtValue) OrtStatus
	IsTensor(OrtValue, *int32) OrtStatus
	GetValueType(OrtValue, *ONNXType) OrtStatus
	HasValue(OrtValue, *int32) OrtStatus
	GetTensorMutableData(OrtValue, *unsafe.Pointer) OrtStatus
	GetTensorTypeAndShape(OrtValue, *OrtTensorTypeAndShapeInfo) OrtStatus
	GetTensorElementType(OrtTensorTypeAndShapeInfo, *ONNXTensorElementDataType) OrtStatus
	GetDimensionsCount(OrtTensorTypeAndShapeInfo, *uintptr) OrtStatus
	GetDimensions(OrtTensorTypeAndShapeInfo, *int64, uintptr) OrtStatus
	GetTensorShapeElementCount(OrtTensorTypeAndShapeInfo, *uintptr) OrtStatus
	ReleaseValue(OrtValue)
	ReleaseTensorTypeAndShapeInfo(OrtTensorTypeAndShapeInfo)

	// String tensor operations
	FillStringTensor(OrtValue, **byte, uintptr) OrtStatus
	GetStringTensorDataLength(OrtValue, *uintptr) OrtStatus
	GetStringTensorContent(OrtValue, unsafe.Pointer, uintptr, *uintptr, uintptr) OrtStatus
	GetStringTensorElementLength(OrtValue, uintptr, *uintptr) OrtStatus
	GetStringTensorElement(OrtValue, uintptr, uintptr, unsafe.Pointer) OrtStatus
	FillStringTensorElement(OrtValue, *byte, uintptr) OrtStatus

	// Sequence/Map operations
	GetValue(OrtValue, int32, OrtAllocator, *OrtValue) OrtStatus
	GetValueCount(OrtValue, *uintptr) OrtStatus
	CastTypeInfoToMapTypeInfo(OrtTypeInfo, *OrtMapTypeInfo) OrtStatus
	CastTypeInfoToSequenceTypeInfo(OrtTypeInfo, *OrtSequenceTypeInfo) OrtStatus
	GetMapKeyType(OrtMapTypeInfo, *ONNXTensorElementDataType) OrtStatus
	GetSequenceElementType(OrtSequenceTypeInfo, *OrtTypeInfo) OrtStatus
	ReleaseMapTypeInfo(OrtMapTypeInfo)
	ReleaseSequenceTypeInfo(OrtSequenceTypeInfo)

	// IO Binding
	CreateIoBinding(OrtSession, *OrtIoBinding) OrtStatus
	ReleaseIoBinding(OrtIoBinding)
	BindInput(OrtIoBinding, *byte, OrtValue) OrtStatus
	BindOutput(OrtIoBinding, *byte, OrtValue) OrtStatus
	BindOutputToDevice(OrtIoBinding, *byte, OrtMemoryInfo) OrtStatus
	GetBoundOutputNames(OrtIoBinding, OrtAllocator, **byte, *uintptr, *uintptr) OrtStatus
	GetBoundOutputValues(OrtIoBinding, OrtAllocator, **OrtValue, *uintptr) OrtStatus
	ClearBoundInputs(OrtIoBinding)
	ClearBoundOutputs(OrtIoBinding)
	RunWithBinding(OrtSession, OrtRunOptions, OrtIoBinding) OrtStatus
	SynchronizeBoundInputs(OrtIoBinding) OrtStatus
	SynchronizeBoundOutputs(OrtIoBinding) OrtStatus

	// Execution provider information
	GetAvailableProviders(***byte, *int32) OrtStatus
	ReleaseAvailableProviders(**byte, int32) OrtStatus
}
