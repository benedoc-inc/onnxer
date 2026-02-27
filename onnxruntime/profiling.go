package onnxruntime

import (
	"fmt"
	"unsafe"

	"github.com/benedoc-inc/onnxer/internal/cstrings"
)

// EndProfiling stops profiling and returns the path to the profile output file.
// Profiling must have been enabled via SessionOptions.ProfilingOutputPath.
//
// The returned file is a JSON file containing per-operator timing data.
//
// Example:
//
//	session, _ := runtime.NewSessionFromReader(env, model, &ort.SessionOptions{
//	    ProfilingOutputPath: "/tmp/ort_profile",
//	})
//	defer session.Close()
//	// ... run inference ...
//	profilePath, _ := session.EndProfiling()
//	fmt.Println("Profile written to:", profilePath)
func (s *Session) EndProfiling() (string, error) {
	if s.ptr == 0 {
		return "", ErrSessionClosed
	}

	var pathPtr *byte
	status := s.runtime.apiFuncs.SessionEndProfiling(s.ptr, s.runtime.allocator.ptr, &pathPtr)
	if err := s.runtime.statusError(status); err != nil {
		return "", fmt.Errorf("failed to end profiling: %w", err)
	}

	path := cstrings.CStringToString(pathPtr)
	s.runtime.allocator.free(unsafe.Pointer(pathPtr))

	return path, nil
}

// ProfilingStartTimeNs returns the profiling start time in nanoseconds.
// This can be used to correlate profiling data with external timestamps.
func (s *Session) ProfilingStartTimeNs() (uint64, error) {
	if s.ptr == 0 {
		return 0, ErrSessionClosed
	}

	var startTime uint64
	status := s.runtime.apiFuncs.SessionGetProfilingStartTimeNs(s.ptr, &startTime)
	if err := s.runtime.statusError(status); err != nil {
		return 0, fmt.Errorf("failed to get profiling start time: %w", err)
	}

	return startTime, nil
}
