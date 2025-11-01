.PHONY: help test clean

# ONNX Runtime version (can be overridden)
ONNXRUNTIME_VERSION ?= 1.23.0

# Detect OS
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
	OS := linux
	LIB_EXT := so
	LIB_PREFIX := lib
endif
ifeq ($(UNAME_S),Darwin)
	OS := darwin
	LIB_EXT := dylib
	LIB_PREFIX := lib
endif
ifeq ($(OS),Windows_NT)
	OS := windows
	LIB_EXT := dll
	LIB_PREFIX :=
endif

# ONNX Runtime library path (can be overridden)
# Auto-constructed: ./libs/{version}/lib/{lib_prefix}onnxruntime.{ext}
ONNXRUNTIME_LIB_PATH ?= $(shell pwd)/libs/$(ONNXRUNTIME_VERSION)/lib/$(LIB_PREFIX)onnxruntime.$(LIB_EXT)

# Test flags (can be overridden)
# Usage: make test FLAGS="-v -run TestName"
FLAGS ?=

# Default target
.DEFAULT_GOAL := help

# Show help
help:
	@echo "Available targets:"
	@echo ""
	@echo "  make test                              - Run all tests"
	@echo "  make test FLAGS=-v                     - Run tests with verbose output"
	@echo "  make test FLAGS=\"-v -run TestName\"     - Run specific test with verbose"
	@echo "  make test FLAGS=-short                 - Run only short tests"
	@echo "  make clean                             - Clean test cache"
	@echo "  make help                              - Show this help message"
	@echo ""
	@echo "Environment variables:"
	@echo "  ONNXRUNTIME_VERSION                            - ONNX Runtime version (default: $(ONNXRUNTIME_VERSION))"
	@echo "  ONNXRUNTIME_LIB_PATH                           - Path to ONNX Runtime library"
	@echo "                                           (default: auto-detected based on OS and version)"
	@echo "  FLAGS                                  - Additional test flags"
	@echo ""
	@echo "Current configuration:"
	@echo "  OS: $(OS)"
	@echo "  Library path: $(ONNXRUNTIME_LIB_PATH)"
	@echo ""
	@echo "Examples:"
	@echo "  make test FLAGS=-v"
	@echo "  make test FLAGS=\"-v -run TestFullInferencePipeline\""
	@echo "  ONNXRUNTIME_VERSION=1.23.0 make test FLAGS=-v"
	@echo "  ONNXRUNTIME_LIB_PATH=/custom/path/libonnxruntime.dylib make test FLAGS=-v"

# Run tests
test:
	ONNXRUNTIME_LIB_PATH=$(ONNXRUNTIME_LIB_PATH) go test $(FLAGS) ./...

# Clean test cache
clean:
	go clean -testcache
