package onnxruntime

import (
	"fmt"
	"runtime"

	"github.com/benedoc-inc/onnxer/onnxruntime/internal/api"
)

// PrepackedWeightsContainer holds pre-packed kernel weights that can be shared
// across multiple sessions loading the same model. This significantly reduces
// memory usage when running multiple sessions (e.g., in a SessionPool) because
// the packed weight buffers are allocated once and shared.
//
// Use SharePrepackedWeights in PoolConfig for the simplest integration,
// or manage a container directly for advanced scenarios like sharing
// across multiple pools.
type PrepackedWeightsContainer struct {
	ptr     api.OrtPrepackedWeightsContainer
	runtime *Runtime
}

// NewPrepackedWeightsContainer creates a new empty container for sharing
// pre-packed weights across sessions.
func (r *Runtime) NewPrepackedWeightsContainer() (*PrepackedWeightsContainer, error) {
	var ptr api.OrtPrepackedWeightsContainer
	status := r.apiFuncs.CreatePrepackedWeightsContainer(&ptr)
	if err := r.statusError(status); err != nil {
		return nil, fmt.Errorf("failed to create prepacked weights container: %w", err)
	}

	c := &PrepackedWeightsContainer{
		ptr:     ptr,
		runtime: r,
	}
	runtime.AddCleanup(c, func(_ struct{}) { c.Close() }, struct{}{})
	return c, nil
}

// Close releases the prepacked weights container and its resources.
// It is safe to call Close multiple times.
func (c *PrepackedWeightsContainer) Close() {
	if c.ptr != 0 && c.runtime != nil && c.runtime.apiFuncs != nil {
		c.runtime.apiFuncs.ReleasePrepackedWeightsContainer(c.ptr)
		c.ptr = 0
	}
}
