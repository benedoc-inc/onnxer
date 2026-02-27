package onnxruntime

import (
	"log/slog"
)

// SlogHook is a Hook that logs inference events via Go's structured logging (log/slog).
// It logs at Info level on success and Error level on failure.
//
// Example:
//
//	pool, _ := onnxruntime.NewSessionPool(runtime, env, modelData, 4, &onnxruntime.PoolConfig{
//	    Hooks: []onnxruntime.Hook{
//	        onnxruntime.NewSlogHook(slog.Default()),
//	    },
//	})
type SlogHook struct {
	logger *slog.Logger
}

// NewSlogHook creates a Hook that logs inference events to the given slog.Logger.
// If logger is nil, slog.Default() is used.
func NewSlogHook(logger *slog.Logger) *SlogHook {
	if logger == nil {
		logger = slog.Default()
	}
	return &SlogHook{logger: logger}
}

func (h *SlogHook) BeforeRun(_ *RunInfo) {}

func (h *SlogHook) AfterRun(info *RunInfo) {
	if info.Error != nil {
		h.logger.Error("inference failed",
			slog.Duration("duration", info.Duration),
			slog.Any("inputs", info.InputNames),
			slog.String("error", info.Error.Error()),
		)
	} else {
		h.logger.Info("inference completed",
			slog.Duration("duration", info.Duration),
			slog.Any("inputs", info.InputNames),
			slog.Any("outputs", info.OutputNames),
		)
	}
}
