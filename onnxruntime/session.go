package onnxruntime

import (
	"context"
	"errors"
	"fmt"
	"io"
	goruntime "runtime"
	"sync"
	"unsafe"

	"github.com/benedoc-inc/onnxer/internal/cstrings"
	"github.com/benedoc-inc/onnxer/onnxruntime/internal/api"
)

// ExecutionProvider specifies an execution provider and its configuration options.
type ExecutionProvider struct {
	// Name is the execution provider name (e.g., "CPUExecutionProvider", "CUDAExecutionProvider").
	Name string

	// Options is an optional map of key-value configuration options for this provider.
	// For example, CUDA provider accepts "device_id", "gpu_mem_limit", etc.
	// If nil, the provider is configured with default settings.
	Options map[string]string
}

// SessionOptions configures options for creating an inference session.
type SessionOptions struct {
	// IntraOpNumThreads sets the number of threads used for parallelizing
	// execution within nodes. A value of 0 uses the default number of threads.
	IntraOpNumThreads int

	// InterOpNumThreads sets the number of threads used for parallelizing
	// execution of the graph (across nodes). A value of 0 uses the default.
	InterOpNumThreads int

	// ExecutionProviders specifies the execution providers to use, in order of preference.
	// If empty, the default provider(s) will be used.
	ExecutionProviders []ExecutionProvider

	// GraphOptimization sets the graph optimization level.
	// Zero value (GraphOptimizationDisabled) means no optimization.
	GraphOptimization GraphOptimizationLevel

	// ExecutionMode controls sequential vs parallel operator execution.
	// Zero value (ExecutionModeSequential) means sequential.
	ExecutionMode ExecutionMode

	// CpuMemArena controls whether the CPU memory arena is enabled.
	// nil means use the ORT default (enabled). Explicit true/false overrides.
	CpuMemArena *bool

	// MemPattern controls whether memory pattern optimization is enabled.
	// nil means use the ORT default (enabled). Explicit true/false overrides.
	MemPattern *bool

	// LogSeverityLevel overrides the session's log severity level.
	// nil means use the environment default.
	LogSeverityLevel *LoggingLevel

	// FreeDimensionOverrides fixes symbolic dimensions by name at session creation time.
	// Keys are symbolic dimension names (e.g., "batch_size"), values are the fixed sizes.
	// This improves memory allocation and kernel selection for models with dynamic shapes.
	FreeDimensionOverrides map[string]int64

	// DeterministicCompute when non-nil enables or disables deterministic computation.
	// When true, ORT avoids non-deterministic GPU kernels for reproducible results.
	DeterministicCompute *bool

	// ConfigEntries provides arbitrary key-value configuration entries.
	ConfigEntries map[string]string

	// ProfilingOutputPath enables profiling and sets the output file path prefix.
	// If non-empty, profiling data will be written when EndProfiling is called.
	// The actual output file will have a timestamp suffix appended.
	ProfilingOutputPath string

	// OptimizedModelFilePath saves the graph-optimized model to the given path.
	// This allows subsequent sessions to load the optimized model directly,
	// avoiding re-optimization and reducing session creation time.
	OptimizedModelFilePath string
}

// Session represents an ONNX Runtime inference session that can execute
// a loaded ONNX model.
//
// A Session is NOT safe for concurrent use from multiple goroutines.
// If you need concurrent inference, create separate Sessions or
// synchronize access with a sync.Mutex.
type Session struct {
	ptr     api.OrtSession
	runtime *Runtime

	// metadata
	inputNames  []string
	outputNames []string

	// cached null-terminated name bytes to avoid per-Run allocations
	inputNameCStrs  [][]byte
	outputNameCStrs [][]byte
}

// NewSession creates a new inference session from a model file.
func (r *Runtime) NewSession(env *Env, modelPath string, options *SessionOptions) (*Session, error) {
	var optsPtr api.OrtSessionOptions
	if options != nil {
		status := r.apiFuncs.CreateSessionOptions(&optsPtr)
		if err := r.statusError(status); err != nil {
			return nil, fmt.Errorf("failed to create session options: %w", err)
		}
		defer func() {
			if optsPtr != 0 {
				r.apiFuncs.ReleaseSessionOptions(optsPtr)
			}
		}()

		if err := r.configureSessionOptions(optsPtr, options); err != nil {
			return nil, err
		}
	}

	modelPathBytes := append([]byte(modelPath), 0)
	var sessionPtr api.OrtSession

	status := r.apiFuncs.CreateSession(env.ptr, &modelPathBytes[0], api.OrtSessionOptions(optsPtr), &sessionPtr)
	if err := r.statusError(status); err != nil {
		return nil, fmt.Errorf("failed to create session: %w", err)
	}

	session := &Session{
		ptr:     sessionPtr,
		runtime: r,
	}
	goruntime.AddCleanup(session, func(_ struct{}) { session.Close() }, struct{}{})

	// Initialize metadata cache
	if err := session.initializeMetadata(); err != nil {
		session.Close()
		return nil, fmt.Errorf("failed to initialize session metadata: %w", err)
	}

	return session, nil
}

// NewSessionFromReader creates a new inference session from a model loaded from modelReader.
// The modelReader contains the ONNX model data, and options configures session-specific settings (may be nil for defaults).
func (r *Runtime) NewSessionFromReader(env *Env, modelReader io.Reader, options *SessionOptions) (*Session, error) {
	// Read all data from the reader
	modelData, err := io.ReadAll(modelReader)
	if err != nil {
		return nil, fmt.Errorf("failed to read model data: %w", err)
	}

	if len(modelData) == 0 {
		return nil, fmt.Errorf("model data cannot be empty")
	}

	var optsPtr api.OrtSessionOptions
	if options != nil {
		status := r.apiFuncs.CreateSessionOptions(&optsPtr)
		if err := r.statusError(status); err != nil {
			return nil, fmt.Errorf("failed to create session options: %w", err)
		}
		defer func() {
			if optsPtr != 0 {
				r.apiFuncs.ReleaseSessionOptions(optsPtr)
			}
		}()

		if err := r.configureSessionOptions(optsPtr, options); err != nil {
			return nil, err
		}
	}

	var sessionPtr api.OrtSession

	status := r.apiFuncs.CreateSessionFromArray(env.ptr, unsafe.Pointer(&modelData[0]), uintptr(len(modelData)), api.OrtSessionOptions(optsPtr), &sessionPtr)
	if err := r.statusError(status); err != nil {
		return nil, fmt.Errorf("failed to create session: %w", err)
	}

	session := &Session{
		ptr:     sessionPtr,
		runtime: r,
	}
	goruntime.AddCleanup(session, func(_ struct{}) { session.Close() }, struct{}{})

	// Initialize metadata cache
	if err := session.initializeMetadata(); err != nil {
		session.Close()
		return nil, fmt.Errorf("failed to initialize session metadata: %w", err)
	}

	return session, nil
}

// initializeMetadata caches input and output names during session creation
func (s *Session) initializeMetadata() error {
	// Get input count and names
	inputCount, err := s.getInputCount()
	if err != nil {
		return fmt.Errorf("failed to get input count: %w", err)
	}

	s.inputNames = make([]string, inputCount)
	s.inputNameCStrs = make([][]byte, inputCount)
	for i := range inputCount {
		name, err := s.getInputName(i)
		if err != nil {
			return fmt.Errorf("failed to get input name at index %d: %w", i, err)
		}
		s.inputNames[i] = name
		s.inputNameCStrs[i] = append([]byte(name), 0)
	}

	// Get output count and names
	outputCount, err := s.getOutputCount()
	if err != nil {
		return fmt.Errorf("failed to get output count: %w", err)
	}

	s.outputNames = make([]string, outputCount)
	s.outputNameCStrs = make([][]byte, outputCount)
	for i := range outputCount {
		name, err := s.getOutputName(i)
		if err != nil {
			return fmt.Errorf("failed to get output name at index %d: %w", i, err)
		}
		s.outputNames[i] = name
		s.outputNameCStrs[i] = append([]byte(name), 0)
	}

	return nil
}

// InputNames returns all input names for the model.
func (s *Session) InputNames() []string {
	return s.inputNames
}

// OutputNames returns all output names for the model.
func (s *Session) OutputNames() []string {
	return s.outputNames
}

// getInputCount retrieves the input count from ONNX Runtime (internal use)
func (s *Session) getInputCount() (int, error) {
	if s.ptr == 0 {
		return 0, ErrSessionClosed
	}

	var count uintptr
	status := s.runtime.apiFuncs.SessionGetInputCount(s.ptr, &count)
	if err := s.runtime.statusError(status); err != nil {
		return 0, fmt.Errorf("failed to get input count: %w", err)
	}

	return int(count), nil
}

// getOutputCount retrieves the output count from ONNX Runtime (internal use)
func (s *Session) getOutputCount() (int, error) {
	if s.ptr == 0 {
		return 0, ErrSessionClosed
	}

	var count uintptr
	status := s.runtime.apiFuncs.SessionGetOutputCount(s.ptr, &count)
	if err := s.runtime.statusError(status); err != nil {
		return 0, fmt.Errorf("failed to get output count: %w", err)
	}

	return int(count), nil
}

// getInputName retrieves the input name from ONNX Runtime (internal use)
func (s *Session) getInputName(index int) (string, error) {
	if s.ptr == 0 {
		return "", ErrSessionClosed
	}

	if s.runtime.allocator == nil {
		return "", errors.New("allocator not initialized")
	}

	var namePtr *byte
	status := s.runtime.apiFuncs.SessionGetInputName(s.ptr, uintptr(index), s.runtime.allocator.ptr, &namePtr)
	if err := s.runtime.statusError(status); err != nil {
		return "", fmt.Errorf("failed to get input name: %w", err)
	}

	name := cstrings.CStringToString(namePtr)

	// Free the allocated name
	s.runtime.allocator.free(unsafe.Pointer(namePtr))

	return name, nil
}

// getOutputName retrieves the output name from ONNX Runtime (internal use)
func (s *Session) getOutputName(index int) (string, error) {
	if s.ptr == 0 {
		return "", ErrSessionClosed
	}

	if s.runtime.allocator == nil {
		return "", errors.New("allocator not initialized")
	}

	var namePtr *byte
	status := s.runtime.apiFuncs.SessionGetOutputName(s.ptr, uintptr(index), s.runtime.allocator.ptr, &namePtr)
	if err := s.runtime.statusError(status); err != nil {
		return "", fmt.Errorf("failed to get output name: %w", err)
	}

	name := cstrings.CStringToString(namePtr)

	// Free the allocated name
	s.runtime.allocator.free(unsafe.Pointer(namePtr))

	return name, nil
}

// RunOption is a functional option for configuring inference execution.
type RunOption func(*runConfig)

type runConfig struct {
	outputNames  []string
	loraAdapters []*LoraAdapter
	runTag       string
}

// WithOutputNames specifies which outputs to compute during inference.
// If not specified, all model outputs are computed.
func WithOutputNames(names ...string) RunOption {
	return func(c *runConfig) {
		c.outputNames = names
	}
}

// WithRunTag sets a tag on the run for log correlation and debugging.
// The tag appears in ORT log output to identify specific inference runs.
func WithRunTag(tag string) RunOption {
	return func(c *runConfig) {
		c.runTag = tag
	}
}

// Run executes the model with the provided inputs and returns the computed outputs.
// The inputs parameter is a map from input name to tensor value.
func (s *Session) Run(ctx context.Context, inputs map[string]*Value, opts ...RunOption) (map[string]*Value, error) {
	if s.ptr == 0 {
		return nil, ErrSessionClosed
	}

	config := &runConfig{
		outputNames: s.outputNames, // default: all outputs
	}
	for _, opt := range opts {
		opt(config)
	}

	// Build input arrays from map using cached metadata
	inputNames := make([]string, 0, len(s.inputNames))
	inputValues := make([]*Value, 0, len(s.inputNames))

	for _, name := range s.inputNames {
		if value, ok := inputs[name]; ok {
			inputNames = append(inputNames, name)
			inputValues = append(inputValues, value)
		} else {
			inputNames = append(inputNames, "")
			inputValues = append(inputValues, nil)
		}
	}

	// Call the low-level run method
	outputValues, err := s.run(ctx, inputNames, inputValues, config)
	if err != nil {
		return nil, err
	}

	// Convert output arrays to map
	outputs := make(map[string]*Value, len(outputValues))
	for i, value := range outputValues {
		outputs[config.outputNames[i]] = value
	}
	return outputs, nil
}

// createRunOptions creates OrtRunOptions with context cancellation and LoRA adapter support.
// Returns the run options pointer and a cleanup function that must be called.
func (s *Session) createRunOptions(ctx context.Context, config *runConfig) (api.OrtRunOptions, func(), error) {
	needsRunOpts := (ctx != nil && ctx.Done() != nil) || len(config.loraAdapters) > 0 || config.runTag != ""

	if !needsRunOpts {
		return 0, func() {}, nil
	}

	var runOpts api.OrtRunOptions
	status := s.runtime.apiFuncs.CreateRunOptions(&runOpts)
	if err := s.runtime.statusError(status); err != nil {
		return 0, nil, fmt.Errorf("failed to create run options: %w", err)
	}

	// Attach LoRA adapters to run options
	for _, adapter := range config.loraAdapters {
		if adapter == nil || adapter.ptr == 0 {
			continue
		}
		status := s.runtime.apiFuncs.RunOptionsAddActiveLoraAdapter(runOpts, adapter.ptr)
		if err := s.runtime.statusError(status); err != nil {
			s.runtime.apiFuncs.ReleaseRunOptions(runOpts)
			return 0, nil, fmt.Errorf("failed to add LoRA adapter to run options: %w", err)
		}
	}

	// Set run tag
	if config.runTag != "" {
		tagBytes := append([]byte(config.runTag), 0)
		status := s.runtime.apiFuncs.RunOptionsSetRunTag(runOpts, &tagBytes[0])
		if err := s.runtime.statusError(status); err != nil {
			s.runtime.apiFuncs.ReleaseRunOptions(runOpts)
			return 0, nil, fmt.Errorf("failed to set run tag: %w", err)
		}
	}

	// Watch for context cancellation in a goroutine.
	// Use WaitGroup to ensure the goroutine has fully exited before
	// we release the run options, preventing a race where
	// RunOptionsSetTerminate is called on freed memory.
	var wg sync.WaitGroup
	done := make(chan struct{})
	if ctx != nil && ctx.Done() != nil {
		wg.Add(1)
		go func() {
			defer wg.Done()
			select {
			case <-ctx.Done():
				s.runtime.apiFuncs.RunOptionsSetTerminate(runOpts)
			case <-done:
			}
		}()
	}

	cleanup := func() {
		close(done)
		wg.Wait()
		s.runtime.apiFuncs.ReleaseRunOptions(runOpts)
	}

	return runOpts, cleanup, nil
}

// run executes the model with the provided inputs and returns the computed outputs.
func (s *Session) run(ctx context.Context, inputNames []string, inputs []*Value, config *runConfig) ([]*Value, error) {
	if len(inputNames) != len(inputs) {
		return nil, fmt.Errorf("number of input names (%d) must match number of inputs (%d)", len(inputNames), len(inputs))
	}

	outputNames := config.outputNames

	runOpts, cleanup, err := s.createRunOptions(ctx, config)
	if err != nil {
		return nil, err
	}
	defer cleanup()

	// Prepare input name pointers using cached C strings
	inputNamePtrs := make([]*byte, len(inputNames))
	for i, name := range inputNames {
		if cstr := s.cachedCStr(name, s.inputNames, s.inputNameCStrs); cstr != nil {
			inputNamePtrs[i] = cstr
		} else {
			nameBytes := append([]byte(name), 0)
			inputNamePtrs[i] = &nameBytes[0]
		}
	}

	// Prepare input value pointers
	inputValuePtrs := make([]api.OrtValue, len(inputs))
	for i, input := range inputs {
		if input != nil {
			inputValuePtrs[i] = input.ptr
		}
	}

	// Prepare output name pointers using cached C strings
	outputNamePtrs := make([]*byte, len(outputNames))
	for i, name := range outputNames {
		if cstr := s.cachedCStr(name, s.outputNames, s.outputNameCStrs); cstr != nil {
			outputNamePtrs[i] = cstr
		} else {
			nameBytes := append([]byte(name), 0)
			outputNamePtrs[i] = &nameBytes[0]
		}
	}

	// Prepare output value pointers
	outputValuePtrs := make([]api.OrtValue, len(outputNames))

	// Call Run
	status := s.runtime.apiFuncs.Run(
		s.ptr,
		runOpts,
		&inputNamePtrs[0],
		&inputValuePtrs[0],
		uintptr(len(inputs)),
		&outputNamePtrs[0],
		uintptr(len(outputNames)),
		&outputValuePtrs[0],
	)
	if err := s.runtime.statusError(status); err != nil {
		return nil, fmt.Errorf("failed to run inference: %w", err)
	}

	// Wrap output values
	outputs := make([]*Value, len(outputValuePtrs))
	for i, ptr := range outputValuePtrs {
		outputs[i] = s.runtime.newValueFromPtr(ptr)
	}

	return outputs, nil
}

// configureSessionOptions applies all session options to the ORT session options pointer.
func (r *Runtime) configureSessionOptions(optsPtr api.OrtSessionOptions, options *SessionOptions) error {
	if options.IntraOpNumThreads > 0 {
		status := r.apiFuncs.SetIntraOpNumThreads(optsPtr, int32(options.IntraOpNumThreads))
		if err := r.statusError(status); err != nil {
			return fmt.Errorf("failed to set intra-op num threads: %w", err)
		}
	}

	if options.InterOpNumThreads > 0 {
		status := r.apiFuncs.SetInterOpNumThreads(optsPtr, int32(options.InterOpNumThreads))
		if err := r.statusError(status); err != nil {
			return fmt.Errorf("failed to set inter-op num threads: %w", err)
		}
	}

	if options.GraphOptimization != 0 {
		status := r.apiFuncs.SetSessionGraphOptimizationLevel(optsPtr, int32(options.GraphOptimization))
		if err := r.statusError(status); err != nil {
			return fmt.Errorf("failed to set graph optimization level: %w", err)
		}
	}

	if options.ExecutionMode != 0 {
		status := r.apiFuncs.SetSessionExecutionMode(optsPtr, int32(options.ExecutionMode))
		if err := r.statusError(status); err != nil {
			return fmt.Errorf("failed to set execution mode: %w", err)
		}
	}

	if options.CpuMemArena != nil {
		var status api.OrtStatus
		if *options.CpuMemArena {
			status = r.apiFuncs.EnableCpuMemArena(optsPtr)
		} else {
			status = r.apiFuncs.DisableCpuMemArena(optsPtr)
		}
		if err := r.statusError(status); err != nil {
			return fmt.Errorf("failed to configure CPU memory arena: %w", err)
		}
	}

	if options.MemPattern != nil {
		var status api.OrtStatus
		if *options.MemPattern {
			status = r.apiFuncs.EnableMemPattern(optsPtr)
		} else {
			status = r.apiFuncs.DisableMemPattern(optsPtr)
		}
		if err := r.statusError(status); err != nil {
			return fmt.Errorf("failed to configure memory pattern: %w", err)
		}
	}

	if options.LogSeverityLevel != nil {
		status := r.apiFuncs.SetSessionLogSeverityLevel(optsPtr, int32(*options.LogSeverityLevel))
		if err := r.statusError(status); err != nil {
			return fmt.Errorf("failed to set log severity level: %w", err)
		}
	}

	for name, size := range options.FreeDimensionOverrides {
		nameBytes := append([]byte(name), 0)
		status := r.apiFuncs.AddFreeDimensionOverrideByName(optsPtr, &nameBytes[0], size)
		if err := r.statusError(status); err != nil {
			return fmt.Errorf("failed to add free dimension override %q: %w", name, err)
		}
	}

	if options.DeterministicCompute != nil {
		val := int32(0)
		if *options.DeterministicCompute {
			val = 1
		}
		status := r.apiFuncs.SetDeterministicCompute(optsPtr, val)
		if err := r.statusError(status); err != nil {
			return fmt.Errorf("failed to set deterministic compute: %w", err)
		}
	}

	for k, v := range options.ConfigEntries {
		keyBytes := append([]byte(k), 0)
		valBytes := append([]byte(v), 0)
		status := r.apiFuncs.AddSessionConfigEntry(optsPtr, &keyBytes[0], &valBytes[0])
		if err := r.statusError(status); err != nil {
			return fmt.Errorf("failed to add session config entry %q: %w", k, err)
		}
	}

	if options.ProfilingOutputPath != "" {
		pathBytes := append([]byte(options.ProfilingOutputPath), 0)
		status := r.apiFuncs.EnableProfiling(optsPtr, &pathBytes[0])
		if err := r.statusError(status); err != nil {
			return fmt.Errorf("failed to enable profiling: %w", err)
		}
	}

	if options.OptimizedModelFilePath != "" {
		pathBytes := append([]byte(options.OptimizedModelFilePath), 0)
		status := r.apiFuncs.SetOptimizedModelFilePath(optsPtr, &pathBytes[0])
		if err := r.statusError(status); err != nil {
			return fmt.Errorf("failed to set optimized model file path: %w", err)
		}
	}

	if err := r.configureExecutionProviders(optsPtr, options); err != nil {
		return err
	}

	return nil
}

// configureExecutionProviders configures execution providers for the session options.
func (r *Runtime) configureExecutionProviders(optsPtr api.OrtSessionOptions, options *SessionOptions) error {
	if len(options.ExecutionProviders) == 0 {
		return nil
	}

	for _, provider := range options.ExecutionProviders {
		providerNameBytes := append([]byte(provider.Name), 0)

		var keyPtrs **byte
		var valuePtrs **byte
		numOpts := uintptr(len(provider.Options))

		if numOpts > 0 {
			keys := make([]*byte, 0, numOpts)
			values := make([]*byte, 0, numOpts)
			for k, v := range provider.Options {
				kBytes := append([]byte(k), 0)
				vBytes := append([]byte(v), 0)
				keys = append(keys, &kBytes[0])
				values = append(values, &vBytes[0])
			}
			keyPtrs = &keys[0]
			valuePtrs = &values[0]
		}

		status := r.apiFuncs.SessionOptionsAppendExecutionProvider(
			optsPtr,
			&providerNameBytes[0],
			keyPtrs,
			valuePtrs,
			numOpts,
		)
		if err := r.statusError(status); err != nil {
			return fmt.Errorf("failed to append execution provider %q: %w", provider.Name, err)
		}
	}

	return nil
}

// cachedCStr returns a pointer to a cached null-terminated C string for the given name,
// or nil if the name is not in the cache. This avoids per-Run allocations.
func (s *Session) cachedCStr(name string, names []string, cstrs [][]byte) *byte {
	for i, n := range names {
		if n == name {
			return &cstrs[i][0]
		}
	}
	return nil
}

// Close releases the session and associated resources.
// It is safe to call Close multiple times.
func (s *Session) Close() {
	if s.ptr != 0 && s.runtime != nil && s.runtime.apiFuncs != nil {
		s.runtime.apiFuncs.ReleaseSession(s.ptr)
		s.ptr = 0
	}
}
