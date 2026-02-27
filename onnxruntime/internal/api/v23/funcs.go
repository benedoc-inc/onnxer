package v23

import (
	"fmt"
	"unsafe"

	"github.com/benedoc-inc/onnxer/onnxruntime/internal/api"
	"github.com/ebitengine/purego"
)

// Funcs contains cached function pointers to ONNX Runtime C API functions.
type Funcs struct {
	// Status and error handling
	createStatus    func(api.OrtErrorCode, *byte) api.OrtStatus
	getErrorCode    func(api.OrtStatus) api.OrtErrorCode
	getErrorMessage func(api.OrtStatus) unsafe.Pointer
	releaseStatus   func(api.OrtStatus)

	// Environment
	createEnv  func(api.OrtLoggingLevel, *byte, *api.OrtEnv) api.OrtStatus
	releaseEnv func(api.OrtEnv)

	// Allocator
	getAllocatorWithDefaultOptions func(*api.OrtAllocator) api.OrtStatus
	allocatorFree                  func(api.OrtAllocator, unsafe.Pointer)

	// Memory info
	createCpuMemoryInfo func(api.OrtAllocatorType, api.OrtMemType, *api.OrtMemoryInfo) api.OrtStatus
	releaseMemoryInfo   func(api.OrtMemoryInfo)

	// Telemetry
	enableTelemetryEvents  func(api.OrtEnv) api.OrtStatus
	disableTelemetryEvents func(api.OrtEnv) api.OrtStatus

	// Session options
	createSessionOptions                  func(*api.OrtSessionOptions) api.OrtStatus
	setOptimizedModelFilePath             func(api.OrtSessionOptions, *byte) api.OrtStatus
	setIntraOpNumThreads                  func(api.OrtSessionOptions, int32) api.OrtStatus
	setInterOpNumThreads                  func(api.OrtSessionOptions, int32) api.OrtStatus
	setSessionExecutionMode               func(api.OrtSessionOptions, int32) api.OrtStatus
	setSessionGraphOptimizationLevel      func(api.OrtSessionOptions, int32) api.OrtStatus
	enableCpuMemArena                     func(api.OrtSessionOptions) api.OrtStatus
	disableCpuMemArena                    func(api.OrtSessionOptions) api.OrtStatus
	enableMemPattern                      func(api.OrtSessionOptions) api.OrtStatus
	disableMemPattern                     func(api.OrtSessionOptions) api.OrtStatus
	setSessionLogSeverityLevel            func(api.OrtSessionOptions, int32) api.OrtStatus
	addSessionConfigEntry                 func(api.OrtSessionOptions, *byte, *byte) api.OrtStatus
	enableProfiling                       func(api.OrtSessionOptions, *byte) api.OrtStatus
	disableProfiling                      func(api.OrtSessionOptions) api.OrtStatus
	sessionOptionsAppendExecutionProvider func(api.OrtSessionOptions, *byte, **byte, **byte, uintptr) api.OrtStatus
	releaseSessionOptions                 func(api.OrtSessionOptions)

	// Run options
	createRunOptions               func(*api.OrtRunOptions) api.OrtStatus
	releaseRunOptions              func(api.OrtRunOptions)
	runOptionsSetTerminate         func(api.OrtRunOptions) api.OrtStatus
	runOptionsUnsetTerminate       func(api.OrtRunOptions) api.OrtStatus
	addRunConfigEntry              func(api.OrtRunOptions, *byte, *byte) api.OrtStatus
	runOptionsAddActiveLoraAdapter func(api.OrtRunOptions, api.OrtLoraAdapter) api.OrtStatus

	// Session
	createSession          func(api.OrtEnv, *byte, api.OrtSessionOptions, *api.OrtSession) api.OrtStatus
	createSessionFromArray func(api.OrtEnv, unsafe.Pointer, uintptr, api.OrtSessionOptions, *api.OrtSession) api.OrtStatus
	sessionGetInputCount   func(api.OrtSession, *uintptr) api.OrtStatus
	sessionGetOutputCount  func(api.OrtSession, *uintptr) api.OrtStatus
	sessionGetInputName    func(api.OrtSession, uintptr, api.OrtAllocator, **byte) api.OrtStatus
	sessionGetOutputName   func(api.OrtSession, uintptr, api.OrtAllocator, **byte) api.OrtStatus
	run                    func(api.OrtSession, api.OrtRunOptions, **byte, *api.OrtValue, uintptr, **byte, uintptr, *api.OrtValue) api.OrtStatus
	releaseSession         func(api.OrtSession)

	// Profiling
	sessionEndProfiling            func(api.OrtSession, api.OrtAllocator, **byte) api.OrtStatus
	sessionGetProfilingStartTimeNs func(api.OrtSession, *uint64) api.OrtStatus

	// LoRA adapters
	createLoraAdapter          func(*byte, api.OrtAllocator, *api.OrtLoraAdapter) api.OrtStatus
	createLoraAdapterFromArray func(unsafe.Pointer, uintptr, api.OrtAllocator, *api.OrtLoraAdapter) api.OrtStatus
	releaseLoraAdapter         func(api.OrtLoraAdapter)

	// Build info
	getBuildInfoString func() unsafe.Pointer

	// Model metadata
	sessionGetModelMetadata               func(api.OrtSession, *api.OrtModelMetadata) api.OrtStatus
	modelMetadataGetProducerName          func(api.OrtModelMetadata, api.OrtAllocator, **byte) api.OrtStatus
	modelMetadataGetGraphName             func(api.OrtModelMetadata, api.OrtAllocator, **byte) api.OrtStatus
	modelMetadataGetDomain                func(api.OrtModelMetadata, api.OrtAllocator, **byte) api.OrtStatus
	modelMetadataGetDescription           func(api.OrtModelMetadata, api.OrtAllocator, **byte) api.OrtStatus
	modelMetadataLookupCustomMetadataMap  func(api.OrtModelMetadata, api.OrtAllocator, *byte, **byte) api.OrtStatus
	modelMetadataGetVersion               func(api.OrtModelMetadata, *int64) api.OrtStatus
	releaseModelMetadata                  func(api.OrtModelMetadata)
	modelMetadataGetCustomMetadataMapKeys func(api.OrtModelMetadata, api.OrtAllocator, ***byte, *int64) api.OrtStatus

	// Type introspection
	sessionGetInputTypeInfo  func(api.OrtSession, uintptr, *api.OrtTypeInfo) api.OrtStatus
	sessionGetOutputTypeInfo func(api.OrtSession, uintptr, *api.OrtTypeInfo) api.OrtStatus
	castTypeInfoToTensorInfo func(api.OrtTypeInfo, *api.OrtTensorTypeAndShapeInfo) api.OrtStatus
	getOnnxTypeFromTypeInfo  func(api.OrtTypeInfo, *api.ONNXType) api.OrtStatus
	releaseTypeInfo          func(api.OrtTypeInfo)

	// Tensor/Value operations
	createTensorAsOrtValue         func(api.OrtAllocator, *int64, uintptr, api.ONNXTensorElementDataType, *api.OrtValue) api.OrtStatus
	createTensorWithDataAsOrtValue func(api.OrtMemoryInfo, unsafe.Pointer, uintptr, *int64, uintptr, api.ONNXTensorElementDataType, *api.OrtValue) api.OrtStatus
	isTensor                       func(api.OrtValue, *int32) api.OrtStatus
	getValueType                   func(api.OrtValue, *api.ONNXType) api.OrtStatus
	getTensorMutableData           func(api.OrtValue, *unsafe.Pointer) api.OrtStatus
	getTensorTypeAndShape          func(api.OrtValue, *api.OrtTensorTypeAndShapeInfo) api.OrtStatus
	getTensorElementType           func(api.OrtTensorTypeAndShapeInfo, *api.ONNXTensorElementDataType) api.OrtStatus
	getDimensionsCount             func(api.OrtTensorTypeAndShapeInfo, *uintptr) api.OrtStatus
	getDimensions                  func(api.OrtTensorTypeAndShapeInfo, *int64, uintptr) api.OrtStatus
	getTensorShapeElementCount     func(api.OrtTensorTypeAndShapeInfo, *uintptr) api.OrtStatus
	releaseValue                   func(api.OrtValue)
	releaseTensorTypeAndShapeInfo  func(api.OrtTensorTypeAndShapeInfo)

	// String tensor operations
	fillStringTensor             func(api.OrtValue, **byte, uintptr) api.OrtStatus
	getStringTensorDataLength    func(api.OrtValue, *uintptr) api.OrtStatus
	getStringTensorContent       func(api.OrtValue, unsafe.Pointer, uintptr, *uintptr, uintptr) api.OrtStatus
	getStringTensorElementLength func(api.OrtValue, uintptr, *uintptr) api.OrtStatus
	getStringTensorElement       func(api.OrtValue, uintptr, uintptr, unsafe.Pointer) api.OrtStatus
	fillStringTensorElement      func(api.OrtValue, *byte, uintptr) api.OrtStatus

	// Sequence/Map operations
	getValue                       func(api.OrtValue, int32, api.OrtAllocator, *api.OrtValue) api.OrtStatus
	getValueCount                  func(api.OrtValue, *uintptr) api.OrtStatus
	castTypeInfoToMapTypeInfo      func(api.OrtTypeInfo, *api.OrtMapTypeInfo) api.OrtStatus
	castTypeInfoToSequenceTypeInfo func(api.OrtTypeInfo, *api.OrtSequenceTypeInfo) api.OrtStatus
	getMapKeyType                  func(api.OrtMapTypeInfo, *api.ONNXTensorElementDataType) api.OrtStatus
	getSequenceElementType         func(api.OrtSequenceTypeInfo, *api.OrtTypeInfo) api.OrtStatus
	releaseMapTypeInfo             func(api.OrtMapTypeInfo)
	releaseSequenceTypeInfo        func(api.OrtSequenceTypeInfo)

	// IO Binding
	createIoBinding      func(api.OrtSession, *api.OrtIoBinding) api.OrtStatus
	releaseIoBinding     func(api.OrtIoBinding)
	bindInput            func(api.OrtIoBinding, *byte, api.OrtValue) api.OrtStatus
	bindOutput           func(api.OrtIoBinding, *byte, api.OrtValue) api.OrtStatus
	bindOutputToDevice   func(api.OrtIoBinding, *byte, api.OrtMemoryInfo) api.OrtStatus
	getBoundOutputNames  func(api.OrtIoBinding, api.OrtAllocator, **byte, *uintptr, *uintptr) api.OrtStatus
	getBoundOutputValues func(api.OrtIoBinding, api.OrtAllocator, **api.OrtValue, *uintptr) api.OrtStatus
	clearBoundInputs     func(api.OrtIoBinding)
	clearBoundOutputs    func(api.OrtIoBinding)
	runWithBinding       func(api.OrtSession, api.OrtRunOptions, api.OrtIoBinding) api.OrtStatus

	// Execution provider information
	getAvailableProviders     func(***byte, *int32) api.OrtStatus
	releaseAvailableProviders func(**byte, int32) api.OrtStatus
}

// InitializeFuncs initializes the v23 API function pointers from the library handle.
// This is called once during initialization to avoid repeated RegisterFunc calls.
func InitializeFuncs(libraryHandle uintptr) (*Funcs, error) {
	// Get the OrtApiBase from the library
	var ortGetAPIBase func() *APIBase
	purego.RegisterLibFunc(&ortGetAPIBase, libraryHandle, "OrtGetApiBase")

	apiBase := ortGetAPIBase()
	if apiBase == nil {
		return nil, fmt.Errorf("OrtGetApiBase returned nil")
	}

	// Get the versioned API
	var getAPIFunc func(uint32) unsafe.Pointer
	purego.RegisterFunc(&getAPIFunc, apiBase.GetAPI)

	apiPtr := getAPIFunc(APIVersion)
	if apiPtr == nil {
		return nil, fmt.Errorf("failed to get OrtAPI for version %d", APIVersion)
	}

	api := (*API)(apiPtr)

	funcs := &Funcs{}

	// Register all function pointers
	purego.RegisterFunc(&funcs.createStatus, api.CreateStatus)
	purego.RegisterFunc(&funcs.getErrorCode, api.GetErrorCode)
	purego.RegisterFunc(&funcs.getErrorMessage, api.GetErrorMessage)
	purego.RegisterFunc(&funcs.releaseStatus, api.ReleaseStatus)

	purego.RegisterFunc(&funcs.createEnv, api.CreateEnv)
	purego.RegisterFunc(&funcs.releaseEnv, api.ReleaseEnv)

	purego.RegisterFunc(&funcs.enableTelemetryEvents, api.EnableTelemetryEvents)
	purego.RegisterFunc(&funcs.disableTelemetryEvents, api.DisableTelemetryEvents)

	purego.RegisterFunc(&funcs.getAllocatorWithDefaultOptions, api.GetAllocatorWithDefaultOptions)
	purego.RegisterFunc(&funcs.allocatorFree, api.AllocatorFree)

	purego.RegisterFunc(&funcs.createCpuMemoryInfo, api.CreateCpuMemoryInfo)
	purego.RegisterFunc(&funcs.releaseMemoryInfo, api.ReleaseMemoryInfo)

	purego.RegisterFunc(&funcs.createSessionOptions, api.CreateSessionOptions)
	purego.RegisterFunc(&funcs.setOptimizedModelFilePath, api.SetOptimizedModelFilePath)
	purego.RegisterFunc(&funcs.setIntraOpNumThreads, api.SetIntraOpNumThreads)
	purego.RegisterFunc(&funcs.setInterOpNumThreads, api.SetInterOpNumThreads)
	purego.RegisterFunc(&funcs.setSessionExecutionMode, api.SetSessionExecutionMode)
	purego.RegisterFunc(&funcs.setSessionGraphOptimizationLevel, api.SetSessionGraphOptimizationLevel)
	purego.RegisterFunc(&funcs.enableCpuMemArena, api.EnableCpuMemArena)
	purego.RegisterFunc(&funcs.disableCpuMemArena, api.DisableCpuMemArena)
	purego.RegisterFunc(&funcs.enableMemPattern, api.EnableMemPattern)
	purego.RegisterFunc(&funcs.disableMemPattern, api.DisableMemPattern)
	purego.RegisterFunc(&funcs.setSessionLogSeverityLevel, api.SetSessionLogSeverityLevel)
	purego.RegisterFunc(&funcs.addSessionConfigEntry, api.AddSessionConfigEntry)
	purego.RegisterFunc(&funcs.enableProfiling, api.EnableProfiling)
	purego.RegisterFunc(&funcs.disableProfiling, api.DisableProfiling)
	purego.RegisterFunc(&funcs.sessionOptionsAppendExecutionProvider, api.SessionOptionsAppendExecutionProvider)
	purego.RegisterFunc(&funcs.releaseSessionOptions, api.ReleaseSessionOptions)

	purego.RegisterFunc(&funcs.createRunOptions, api.CreateRunOptions)
	purego.RegisterFunc(&funcs.releaseRunOptions, api.ReleaseRunOptions)
	purego.RegisterFunc(&funcs.runOptionsSetTerminate, api.RunOptionsSetTerminate)
	purego.RegisterFunc(&funcs.runOptionsUnsetTerminate, api.RunOptionsUnsetTerminate)
	purego.RegisterFunc(&funcs.addRunConfigEntry, api.AddRunConfigEntry)
	purego.RegisterFunc(&funcs.runOptionsAddActiveLoraAdapter, api.RunOptionsAddActiveLoraAdapter)

	purego.RegisterFunc(&funcs.createSession, api.CreateSession)
	purego.RegisterFunc(&funcs.createSessionFromArray, api.CreateSessionFromArray)
	purego.RegisterFunc(&funcs.sessionGetInputCount, api.SessionGetInputCount)
	purego.RegisterFunc(&funcs.sessionGetOutputCount, api.SessionGetOutputCount)
	purego.RegisterFunc(&funcs.sessionGetInputName, api.SessionGetInputName)
	purego.RegisterFunc(&funcs.sessionGetOutputName, api.SessionGetOutputName)
	purego.RegisterFunc(&funcs.run, api.Run)
	purego.RegisterFunc(&funcs.releaseSession, api.ReleaseSession)

	purego.RegisterFunc(&funcs.sessionEndProfiling, api.SessionEndProfiling)
	purego.RegisterFunc(&funcs.sessionGetProfilingStartTimeNs, api.SessionGetProfilingStartTimeNs)

	purego.RegisterFunc(&funcs.createLoraAdapter, api.CreateLoraAdapter)
	purego.RegisterFunc(&funcs.createLoraAdapterFromArray, api.CreateLoraAdapterFromArray)
	purego.RegisterFunc(&funcs.releaseLoraAdapter, api.ReleaseLoraAdapter)

	purego.RegisterFunc(&funcs.getBuildInfoString, api.GetBuildInfoString)

	purego.RegisterFunc(&funcs.sessionGetModelMetadata, api.SessionGetModelMetadata)
	purego.RegisterFunc(&funcs.modelMetadataGetProducerName, api.ModelMetadataGetProducerName)
	purego.RegisterFunc(&funcs.modelMetadataGetGraphName, api.ModelMetadataGetGraphName)
	purego.RegisterFunc(&funcs.modelMetadataGetDomain, api.ModelMetadataGetDomain)
	purego.RegisterFunc(&funcs.modelMetadataGetDescription, api.ModelMetadataGetDescription)
	purego.RegisterFunc(&funcs.modelMetadataLookupCustomMetadataMap, api.ModelMetadataLookupCustomMetadataMap)
	purego.RegisterFunc(&funcs.modelMetadataGetVersion, api.ModelMetadataGetVersion)
	purego.RegisterFunc(&funcs.releaseModelMetadata, api.ReleaseModelMetadata)
	purego.RegisterFunc(&funcs.modelMetadataGetCustomMetadataMapKeys, api.ModelMetadataGetCustomMetadataMapKeys)

	purego.RegisterFunc(&funcs.sessionGetInputTypeInfo, api.SessionGetInputTypeInfo)
	purego.RegisterFunc(&funcs.sessionGetOutputTypeInfo, api.SessionGetOutputTypeInfo)
	purego.RegisterFunc(&funcs.castTypeInfoToTensorInfo, api.CastTypeInfoToTensorInfo)
	purego.RegisterFunc(&funcs.getOnnxTypeFromTypeInfo, api.GetOnnxTypeFromTypeInfo)
	purego.RegisterFunc(&funcs.releaseTypeInfo, api.ReleaseTypeInfo)

	purego.RegisterFunc(&funcs.createTensorAsOrtValue, api.CreateTensorAsOrtValue)
	purego.RegisterFunc(&funcs.createTensorWithDataAsOrtValue, api.CreateTensorWithDataAsOrtValue)
	purego.RegisterFunc(&funcs.isTensor, api.IsTensor)
	purego.RegisterFunc(&funcs.getValueType, api.GetValueType)
	purego.RegisterFunc(&funcs.getTensorMutableData, api.GetTensorMutableData)
	purego.RegisterFunc(&funcs.getTensorTypeAndShape, api.GetTensorTypeAndShape)
	purego.RegisterFunc(&funcs.getTensorElementType, api.GetTensorElementType)
	purego.RegisterFunc(&funcs.getDimensionsCount, api.GetDimensionsCount)
	purego.RegisterFunc(&funcs.getDimensions, api.GetDimensions)
	purego.RegisterFunc(&funcs.getTensorShapeElementCount, api.GetTensorShapeElementCount)
	purego.RegisterFunc(&funcs.releaseValue, api.ReleaseValue)
	purego.RegisterFunc(&funcs.releaseTensorTypeAndShapeInfo, api.ReleaseTensorTypeAndShapeInfo)

	purego.RegisterFunc(&funcs.fillStringTensor, api.FillStringTensor)
	purego.RegisterFunc(&funcs.getStringTensorDataLength, api.GetStringTensorDataLength)
	purego.RegisterFunc(&funcs.getStringTensorContent, api.GetStringTensorContent)
	purego.RegisterFunc(&funcs.getStringTensorElementLength, api.GetStringTensorElementLength)
	purego.RegisterFunc(&funcs.getStringTensorElement, api.GetStringTensorElement)
	purego.RegisterFunc(&funcs.fillStringTensorElement, api.FillStringTensorElement)

	purego.RegisterFunc(&funcs.getValue, api.GetValue)
	purego.RegisterFunc(&funcs.getValueCount, api.GetValueCount)
	purego.RegisterFunc(&funcs.castTypeInfoToMapTypeInfo, api.CastTypeInfoToMapTypeInfo)
	purego.RegisterFunc(&funcs.castTypeInfoToSequenceTypeInfo, api.CastTypeInfoToSequenceTypeInfo)
	purego.RegisterFunc(&funcs.getMapKeyType, api.GetMapKeyType)
	purego.RegisterFunc(&funcs.getSequenceElementType, api.GetSequenceElementType)
	purego.RegisterFunc(&funcs.releaseMapTypeInfo, api.ReleaseMapTypeInfo)
	purego.RegisterFunc(&funcs.releaseSequenceTypeInfo, api.ReleaseSequenceTypeInfo)

	purego.RegisterFunc(&funcs.createIoBinding, api.CreateIoBinding)
	purego.RegisterFunc(&funcs.releaseIoBinding, api.ReleaseIoBinding)
	purego.RegisterFunc(&funcs.bindInput, api.BindInput)
	purego.RegisterFunc(&funcs.bindOutput, api.BindOutput)
	purego.RegisterFunc(&funcs.bindOutputToDevice, api.BindOutputToDevice)
	purego.RegisterFunc(&funcs.getBoundOutputNames, api.GetBoundOutputNames)
	purego.RegisterFunc(&funcs.getBoundOutputValues, api.GetBoundOutputValues)
	purego.RegisterFunc(&funcs.clearBoundInputs, api.ClearBoundInputs)
	purego.RegisterFunc(&funcs.clearBoundOutputs, api.ClearBoundOutputs)
	purego.RegisterFunc(&funcs.runWithBinding, api.RunWithBinding)

	purego.RegisterFunc(&funcs.getAvailableProviders, api.GetAvailableProviders)
	purego.RegisterFunc(&funcs.releaseAvailableProviders, api.ReleaseAvailableProviders)

	return funcs, nil
}

// Status and error handling methods

func (f *Funcs) CreateStatus(code api.OrtErrorCode, msg *byte) api.OrtStatus {
	return f.createStatus(code, msg)
}

func (f *Funcs) GetErrorCode(status api.OrtStatus) api.OrtErrorCode {
	return f.getErrorCode(status)
}

func (f *Funcs) GetErrorMessage(status api.OrtStatus) unsafe.Pointer {
	return f.getErrorMessage(status)
}

func (f *Funcs) ReleaseStatus(status api.OrtStatus) {
	f.releaseStatus(status)
}

// Environment methods

func (f *Funcs) CreateEnv(logLevel api.OrtLoggingLevel, logID *byte, env *api.OrtEnv) api.OrtStatus {
	return f.createEnv(logLevel, logID, env)
}

func (f *Funcs) ReleaseEnv(env api.OrtEnv) {
	f.releaseEnv(env)
}

// Telemetry methods

func (f *Funcs) EnableTelemetryEvents(env api.OrtEnv) api.OrtStatus {
	return f.enableTelemetryEvents(env)
}

func (f *Funcs) DisableTelemetryEvents(env api.OrtEnv) api.OrtStatus {
	return f.disableTelemetryEvents(env)
}

// Allocator methods

func (f *Funcs) GetAllocatorWithDefaultOptions(allocator *api.OrtAllocator) api.OrtStatus {
	return f.getAllocatorWithDefaultOptions(allocator)
}

func (f *Funcs) AllocatorFree(allocator api.OrtAllocator, ptr unsafe.Pointer) {
	f.allocatorFree(allocator, ptr)
}

// Memory info methods

func (f *Funcs) CreateCpuMemoryInfo(allocType api.OrtAllocatorType, memType api.OrtMemType, memInfo *api.OrtMemoryInfo) api.OrtStatus {
	return f.createCpuMemoryInfo(allocType, memType, memInfo)
}

func (f *Funcs) ReleaseMemoryInfo(memInfo api.OrtMemoryInfo) {
	f.releaseMemoryInfo(memInfo)
}

// Session options methods

func (f *Funcs) CreateSessionOptions(options *api.OrtSessionOptions) api.OrtStatus {
	return f.createSessionOptions(options)
}

func (f *Funcs) SetOptimizedModelFilePath(options api.OrtSessionOptions, path *byte) api.OrtStatus {
	return f.setOptimizedModelFilePath(options, path)
}

func (f *Funcs) SetIntraOpNumThreads(options api.OrtSessionOptions, numThreads int32) api.OrtStatus {
	return f.setIntraOpNumThreads(options, numThreads)
}

func (f *Funcs) SetInterOpNumThreads(options api.OrtSessionOptions, numThreads int32) api.OrtStatus {
	return f.setInterOpNumThreads(options, numThreads)
}

func (f *Funcs) SetSessionExecutionMode(options api.OrtSessionOptions, mode int32) api.OrtStatus {
	return f.setSessionExecutionMode(options, mode)
}

func (f *Funcs) SetSessionGraphOptimizationLevel(options api.OrtSessionOptions, level int32) api.OrtStatus {
	return f.setSessionGraphOptimizationLevel(options, level)
}

func (f *Funcs) EnableCpuMemArena(options api.OrtSessionOptions) api.OrtStatus {
	return f.enableCpuMemArena(options)
}

func (f *Funcs) DisableCpuMemArena(options api.OrtSessionOptions) api.OrtStatus {
	return f.disableCpuMemArena(options)
}

func (f *Funcs) EnableMemPattern(options api.OrtSessionOptions) api.OrtStatus {
	return f.enableMemPattern(options)
}

func (f *Funcs) DisableMemPattern(options api.OrtSessionOptions) api.OrtStatus {
	return f.disableMemPattern(options)
}

func (f *Funcs) SetSessionLogSeverityLevel(options api.OrtSessionOptions, level int32) api.OrtStatus {
	return f.setSessionLogSeverityLevel(options, level)
}

func (f *Funcs) AddSessionConfigEntry(options api.OrtSessionOptions, key *byte, value *byte) api.OrtStatus {
	return f.addSessionConfigEntry(options, key, value)
}

func (f *Funcs) EnableProfiling(options api.OrtSessionOptions, path *byte) api.OrtStatus {
	return f.enableProfiling(options, path)
}

func (f *Funcs) DisableProfiling(options api.OrtSessionOptions) api.OrtStatus {
	return f.disableProfiling(options)
}

func (f *Funcs) SessionOptionsAppendExecutionProvider(options api.OrtSessionOptions, providerName *byte, keys **byte, values **byte, numKeys uintptr) api.OrtStatus {
	return f.sessionOptionsAppendExecutionProvider(options, providerName, keys, values, numKeys)
}

func (f *Funcs) ReleaseSessionOptions(options api.OrtSessionOptions) {
	f.releaseSessionOptions(options)
}

// Run options methods

func (f *Funcs) CreateRunOptions(options *api.OrtRunOptions) api.OrtStatus {
	return f.createRunOptions(options)
}

func (f *Funcs) ReleaseRunOptions(options api.OrtRunOptions) {
	f.releaseRunOptions(options)
}

func (f *Funcs) RunOptionsSetTerminate(options api.OrtRunOptions) api.OrtStatus {
	return f.runOptionsSetTerminate(options)
}

func (f *Funcs) RunOptionsUnsetTerminate(options api.OrtRunOptions) api.OrtStatus {
	return f.runOptionsUnsetTerminate(options)
}

func (f *Funcs) AddRunConfigEntry(options api.OrtRunOptions, key *byte, value *byte) api.OrtStatus {
	return f.addRunConfigEntry(options, key, value)
}

func (f *Funcs) RunOptionsAddActiveLoraAdapter(options api.OrtRunOptions, adapter api.OrtLoraAdapter) api.OrtStatus {
	return f.runOptionsAddActiveLoraAdapter(options, adapter)
}

// Session methods

func (f *Funcs) CreateSession(env api.OrtEnv, modelPath *byte, options api.OrtSessionOptions, session *api.OrtSession) api.OrtStatus {
	return f.createSession(env, modelPath, options, session)
}

func (f *Funcs) CreateSessionFromArray(env api.OrtEnv, modelData unsafe.Pointer, modelDataLength uintptr, options api.OrtSessionOptions, session *api.OrtSession) api.OrtStatus {
	return f.createSessionFromArray(env, modelData, modelDataLength, options, session)
}

func (f *Funcs) SessionGetInputCount(session api.OrtSession, count *uintptr) api.OrtStatus {
	return f.sessionGetInputCount(session, count)
}

func (f *Funcs) SessionGetOutputCount(session api.OrtSession, count *uintptr) api.OrtStatus {
	return f.sessionGetOutputCount(session, count)
}

func (f *Funcs) SessionGetInputName(session api.OrtSession, index uintptr, allocator api.OrtAllocator, name **byte) api.OrtStatus {
	return f.sessionGetInputName(session, index, allocator, name)
}

func (f *Funcs) SessionGetOutputName(session api.OrtSession, index uintptr, allocator api.OrtAllocator, name **byte) api.OrtStatus {
	return f.sessionGetOutputName(session, index, allocator, name)
}

func (f *Funcs) Run(session api.OrtSession, runOptions api.OrtRunOptions, inputNames **byte, inputs *api.OrtValue, inputCount uintptr, outputNames **byte, outputCount uintptr, outputs *api.OrtValue) api.OrtStatus {
	return f.run(session, runOptions, inputNames, inputs, inputCount, outputNames, outputCount, outputs)
}

func (f *Funcs) ReleaseSession(session api.OrtSession) {
	f.releaseSession(session)
}

// Profiling methods

func (f *Funcs) SessionEndProfiling(session api.OrtSession, allocator api.OrtAllocator, out **byte) api.OrtStatus {
	return f.sessionEndProfiling(session, allocator, out)
}

func (f *Funcs) SessionGetProfilingStartTimeNs(session api.OrtSession, out *uint64) api.OrtStatus {
	return f.sessionGetProfilingStartTimeNs(session, out)
}

// LoRA adapter methods

func (f *Funcs) CreateLoraAdapter(path *byte, allocator api.OrtAllocator, out *api.OrtLoraAdapter) api.OrtStatus {
	return f.createLoraAdapter(path, allocator, out)
}

func (f *Funcs) CreateLoraAdapterFromArray(data unsafe.Pointer, dataLen uintptr, allocator api.OrtAllocator, out *api.OrtLoraAdapter) api.OrtStatus {
	return f.createLoraAdapterFromArray(data, dataLen, allocator, out)
}

func (f *Funcs) ReleaseLoraAdapter(adapter api.OrtLoraAdapter) {
	f.releaseLoraAdapter(adapter)
}

// Build info methods

func (f *Funcs) GetBuildInfoString() unsafe.Pointer {
	return f.getBuildInfoString()
}

// Model metadata methods

func (f *Funcs) SessionGetModelMetadata(session api.OrtSession, metadata *api.OrtModelMetadata) api.OrtStatus {
	return f.sessionGetModelMetadata(session, metadata)
}

func (f *Funcs) ModelMetadataGetProducerName(metadata api.OrtModelMetadata, allocator api.OrtAllocator, value **byte) api.OrtStatus {
	return f.modelMetadataGetProducerName(metadata, allocator, value)
}

func (f *Funcs) ModelMetadataGetGraphName(metadata api.OrtModelMetadata, allocator api.OrtAllocator, value **byte) api.OrtStatus {
	return f.modelMetadataGetGraphName(metadata, allocator, value)
}

func (f *Funcs) ModelMetadataGetDomain(metadata api.OrtModelMetadata, allocator api.OrtAllocator, value **byte) api.OrtStatus {
	return f.modelMetadataGetDomain(metadata, allocator, value)
}

func (f *Funcs) ModelMetadataGetDescription(metadata api.OrtModelMetadata, allocator api.OrtAllocator, value **byte) api.OrtStatus {
	return f.modelMetadataGetDescription(metadata, allocator, value)
}

func (f *Funcs) ModelMetadataLookupCustomMetadataMap(metadata api.OrtModelMetadata, allocator api.OrtAllocator, key *byte, value **byte) api.OrtStatus {
	return f.modelMetadataLookupCustomMetadataMap(metadata, allocator, key, value)
}

func (f *Funcs) ModelMetadataGetVersion(metadata api.OrtModelMetadata, version *int64) api.OrtStatus {
	return f.modelMetadataGetVersion(metadata, version)
}

func (f *Funcs) ReleaseModelMetadata(metadata api.OrtModelMetadata) {
	f.releaseModelMetadata(metadata)
}

func (f *Funcs) ModelMetadataGetCustomMetadataMapKeys(metadata api.OrtModelMetadata, allocator api.OrtAllocator, keys ***byte, numKeys *int64) api.OrtStatus {
	return f.modelMetadataGetCustomMetadataMapKeys(metadata, allocator, keys, numKeys)
}

// Type introspection methods

func (f *Funcs) SessionGetInputTypeInfo(session api.OrtSession, index uintptr, typeInfo *api.OrtTypeInfo) api.OrtStatus {
	return f.sessionGetInputTypeInfo(session, index, typeInfo)
}

func (f *Funcs) SessionGetOutputTypeInfo(session api.OrtSession, index uintptr, typeInfo *api.OrtTypeInfo) api.OrtStatus {
	return f.sessionGetOutputTypeInfo(session, index, typeInfo)
}

func (f *Funcs) CastTypeInfoToTensorInfo(typeInfo api.OrtTypeInfo, tensorInfo *api.OrtTensorTypeAndShapeInfo) api.OrtStatus {
	return f.castTypeInfoToTensorInfo(typeInfo, tensorInfo)
}

func (f *Funcs) GetOnnxTypeFromTypeInfo(typeInfo api.OrtTypeInfo, onnxType *api.ONNXType) api.OrtStatus {
	return f.getOnnxTypeFromTypeInfo(typeInfo, onnxType)
}

func (f *Funcs) ReleaseTypeInfo(typeInfo api.OrtTypeInfo) {
	f.releaseTypeInfo(typeInfo)
}

// Tensor/Value operations methods

func (f *Funcs) CreateTensorAsOrtValue(allocator api.OrtAllocator, shape *int64, shapeLen uintptr, dataType api.ONNXTensorElementDataType, value *api.OrtValue) api.OrtStatus {
	return f.createTensorAsOrtValue(allocator, shape, shapeLen, dataType, value)
}

func (f *Funcs) CreateTensorWithDataAsOrtValue(memInfo api.OrtMemoryInfo, data unsafe.Pointer, dataSize uintptr, shape *int64, shapeLen uintptr, dataType api.ONNXTensorElementDataType, value *api.OrtValue) api.OrtStatus {
	return f.createTensorWithDataAsOrtValue(memInfo, data, dataSize, shape, shapeLen, dataType, value)
}

func (f *Funcs) IsTensor(value api.OrtValue, out *int32) api.OrtStatus {
	return f.isTensor(value, out)
}

func (f *Funcs) GetValueType(value api.OrtValue, valueType *api.ONNXType) api.OrtStatus {
	return f.getValueType(value, valueType)
}

func (f *Funcs) GetTensorMutableData(value api.OrtValue, data *unsafe.Pointer) api.OrtStatus {
	return f.getTensorMutableData(value, data)
}

func (f *Funcs) GetTensorTypeAndShape(value api.OrtValue, typeAndShape *api.OrtTensorTypeAndShapeInfo) api.OrtStatus {
	return f.getTensorTypeAndShape(value, typeAndShape)
}

func (f *Funcs) GetTensorElementType(typeAndShape api.OrtTensorTypeAndShapeInfo, dataType *api.ONNXTensorElementDataType) api.OrtStatus {
	return f.getTensorElementType(typeAndShape, dataType)
}

func (f *Funcs) GetDimensionsCount(typeAndShape api.OrtTensorTypeAndShapeInfo, count *uintptr) api.OrtStatus {
	return f.getDimensionsCount(typeAndShape, count)
}

func (f *Funcs) GetDimensions(typeAndShape api.OrtTensorTypeAndShapeInfo, dims *int64, dimsLen uintptr) api.OrtStatus {
	return f.getDimensions(typeAndShape, dims, dimsLen)
}

func (f *Funcs) GetTensorShapeElementCount(typeAndShape api.OrtTensorTypeAndShapeInfo, count *uintptr) api.OrtStatus {
	return f.getTensorShapeElementCount(typeAndShape, count)
}

func (f *Funcs) ReleaseValue(value api.OrtValue) {
	f.releaseValue(value)
}

func (f *Funcs) ReleaseTensorTypeAndShapeInfo(typeAndShape api.OrtTensorTypeAndShapeInfo) {
	f.releaseTensorTypeAndShapeInfo(typeAndShape)
}

// String tensor methods

func (f *Funcs) FillStringTensor(value api.OrtValue, s **byte, sLen uintptr) api.OrtStatus {
	return f.fillStringTensor(value, s, sLen)
}

func (f *Funcs) GetStringTensorDataLength(value api.OrtValue, length *uintptr) api.OrtStatus {
	return f.getStringTensorDataLength(value, length)
}

func (f *Funcs) GetStringTensorContent(value api.OrtValue, s unsafe.Pointer, sLen uintptr, offsets *uintptr, offsetsLen uintptr) api.OrtStatus {
	return f.getStringTensorContent(value, s, sLen, offsets, offsetsLen)
}

func (f *Funcs) GetStringTensorElementLength(value api.OrtValue, index uintptr, length *uintptr) api.OrtStatus {
	return f.getStringTensorElementLength(value, index, length)
}

func (f *Funcs) GetStringTensorElement(value api.OrtValue, sLen uintptr, index uintptr, s unsafe.Pointer) api.OrtStatus {
	return f.getStringTensorElement(value, sLen, index, s)
}

func (f *Funcs) FillStringTensorElement(value api.OrtValue, s *byte, index uintptr) api.OrtStatus {
	return f.fillStringTensorElement(value, s, index)
}

// Sequence/Map methods

func (f *Funcs) GetValue(value api.OrtValue, index int32, allocator api.OrtAllocator, out *api.OrtValue) api.OrtStatus {
	return f.getValue(value, index, allocator, out)
}

func (f *Funcs) GetValueCount(value api.OrtValue, count *uintptr) api.OrtStatus {
	return f.getValueCount(value, count)
}

func (f *Funcs) CastTypeInfoToMapTypeInfo(typeInfo api.OrtTypeInfo, mapTypeInfo *api.OrtMapTypeInfo) api.OrtStatus {
	return f.castTypeInfoToMapTypeInfo(typeInfo, mapTypeInfo)
}

func (f *Funcs) CastTypeInfoToSequenceTypeInfo(typeInfo api.OrtTypeInfo, seqTypeInfo *api.OrtSequenceTypeInfo) api.OrtStatus {
	return f.castTypeInfoToSequenceTypeInfo(typeInfo, seqTypeInfo)
}

func (f *Funcs) GetMapKeyType(mapTypeInfo api.OrtMapTypeInfo, keyType *api.ONNXTensorElementDataType) api.OrtStatus {
	return f.getMapKeyType(mapTypeInfo, keyType)
}

func (f *Funcs) GetSequenceElementType(seqTypeInfo api.OrtSequenceTypeInfo, typeInfo *api.OrtTypeInfo) api.OrtStatus {
	return f.getSequenceElementType(seqTypeInfo, typeInfo)
}

func (f *Funcs) ReleaseMapTypeInfo(mapTypeInfo api.OrtMapTypeInfo) {
	f.releaseMapTypeInfo(mapTypeInfo)
}

func (f *Funcs) ReleaseSequenceTypeInfo(seqTypeInfo api.OrtSequenceTypeInfo) {
	f.releaseSequenceTypeInfo(seqTypeInfo)
}

// IO Binding methods

func (f *Funcs) CreateIoBinding(session api.OrtSession, binding *api.OrtIoBinding) api.OrtStatus {
	return f.createIoBinding(session, binding)
}

func (f *Funcs) ReleaseIoBinding(binding api.OrtIoBinding) {
	f.releaseIoBinding(binding)
}

func (f *Funcs) BindInput(binding api.OrtIoBinding, name *byte, value api.OrtValue) api.OrtStatus {
	return f.bindInput(binding, name, value)
}

func (f *Funcs) BindOutput(binding api.OrtIoBinding, name *byte, value api.OrtValue) api.OrtStatus {
	return f.bindOutput(binding, name, value)
}

func (f *Funcs) BindOutputToDevice(binding api.OrtIoBinding, name *byte, memInfo api.OrtMemoryInfo) api.OrtStatus {
	return f.bindOutputToDevice(binding, name, memInfo)
}

func (f *Funcs) GetBoundOutputNames(binding api.OrtIoBinding, allocator api.OrtAllocator, buffer **byte, lengths *uintptr, count *uintptr) api.OrtStatus {
	return f.getBoundOutputNames(binding, allocator, buffer, lengths, count)
}

func (f *Funcs) GetBoundOutputValues(binding api.OrtIoBinding, allocator api.OrtAllocator, output **api.OrtValue, count *uintptr) api.OrtStatus {
	return f.getBoundOutputValues(binding, allocator, output, count)
}

func (f *Funcs) ClearBoundInputs(binding api.OrtIoBinding) {
	f.clearBoundInputs(binding)
}

func (f *Funcs) ClearBoundOutputs(binding api.OrtIoBinding) {
	f.clearBoundOutputs(binding)
}

func (f *Funcs) RunWithBinding(session api.OrtSession, runOptions api.OrtRunOptions, binding api.OrtIoBinding) api.OrtStatus {
	return f.runWithBinding(session, runOptions, binding)
}

// Execution provider information methods

func (f *Funcs) GetAvailableProviders(providers ***byte, length *int32) api.OrtStatus {
	return f.getAvailableProviders(providers, length)
}

func (f *Funcs) ReleaseAvailableProviders(providers **byte, length int32) api.OrtStatus {
	return f.releaseAvailableProviders(providers, length)
}
