// Package onnxruntime provides Go bindings for the ONNX Runtime C API using purego.
//
// This package allows you to load and execute ONNX models for machine learning inference
// in pure Go, without requiring cgo. It uses the purego library to call into the native
// ONNX Runtime shared library.
//
// # Thread Safety
//
// The following types are NOT safe for concurrent use from multiple goroutines:
//   - [Runtime], [Env], [Session], [Value], [IoBinding], [MemoryInfo]
//
// If you need concurrent inference, create separate Session instances per goroutine,
// or synchronize access with a sync.Mutex. Multiple Sessions can share the same
// Runtime and Env safely as long as they are not accessed concurrently themselves.
//
// # Resource Management
//
// All types that hold native resources (Session, Env, Value, IoBinding, MemoryInfo)
// implement Close(). While finalizers are set as a safety net, you should always
// call Close() explicitly (typically via defer) to ensure timely release of native
// memory, especially with large tensors or high-frequency inference.
package onnxruntime
