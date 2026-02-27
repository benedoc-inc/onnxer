package onnxruntime

import (
	"math"
	"testing"
)

func TestFloat16RoundTrip(t *testing.T) {
	tests := []struct {
		name  string
		input float32
		want  float32
	}{
		{"zero", 0, 0},
		{"negative zero", float32(math.Copysign(0, -1)), float32(math.Copysign(0, -1))},
		{"one", 1.0, 1.0},
		{"negative one", -1.0, -1.0},
		{"half", 0.5, 0.5},
		{"two", 2.0, 2.0},
		{"small normal", 0.00006103515625, 0.00006103515625}, // smallest normal fp16
		{"max fp16", 65504, 65504},
		{"negative max", -65504, -65504},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f16 := NewFloat16(tt.input)
			got := f16.Float32()
			if got != tt.want {
				t.Errorf("Float16 roundtrip(%v) = %v, want %v", tt.input, got, tt.want)
			}
		})
	}
}

func TestFloat16Overflow(t *testing.T) {
	// Values beyond fp16 max should become infinity
	f16 := NewFloat16(100000)
	got := f16.Float32()
	if !math.IsInf(float64(got), 1) {
		t.Errorf("Float16 overflow: expected +Inf, got %v", got)
	}

	f16neg := NewFloat16(-100000)
	gotNeg := f16neg.Float32()
	if !math.IsInf(float64(gotNeg), -1) {
		t.Errorf("Float16 negative overflow: expected -Inf, got %v", gotNeg)
	}
}

func TestFloat16Underflow(t *testing.T) {
	// Very tiny values should become zero
	f16 := NewFloat16(1e-20)
	got := f16.Float32()
	if got != 0 {
		t.Errorf("Float16 underflow: expected 0, got %v", got)
	}
}

func TestFloat16Infinity(t *testing.T) {
	posInf := NewFloat16(float32(math.Inf(1)))
	if !math.IsInf(float64(posInf.Float32()), 1) {
		t.Errorf("expected +Inf, got %v", posInf.Float32())
	}

	negInf := NewFloat16(float32(math.Inf(-1)))
	if !math.IsInf(float64(negInf.Float32()), -1) {
		t.Errorf("expected -Inf, got %v", negInf.Float32())
	}
}

func TestFloat16NaN(t *testing.T) {
	nan := NewFloat16(float32(math.NaN()))
	if !math.IsNaN(float64(nan.Float32())) {
		t.Errorf("expected NaN, got %v", nan.Float32())
	}
}

func TestFloat16Subnormal(t *testing.T) {
	// fp16 subnormal: values between 0 and smallest normal (6.1e-5)
	// Smallest fp16 subnormal is 2^-24 ~= 5.96e-8
	val := float32(0.00003) // should be representable as subnormal in fp16
	f16 := NewFloat16(val)
	got := f16.Float32()
	// Subnormal roundtrip loses precision but should be close
	if got == 0 || math.Abs(float64(got-val)) > float64(val)*0.5 {
		t.Errorf("Float16 subnormal(%v) = %v, too far off", val, got)
	}
}

func TestBFloat16RoundTrip(t *testing.T) {
	tests := []struct {
		name  string
		input float32
		want  float32
	}{
		{"zero", 0, 0},
		{"one", 1.0, 1.0},
		{"negative one", -1.0, -1.0},
		{"two", 2.0, 2.0},
		{"large", 1000.0, 1000.0},
		{"power of two large", 1048576.0, 1048576.0}, // 2^20, exact in bf16
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			bf16 := NewBFloat16(tt.input)
			got := bf16.Float32()
			if got != tt.want {
				t.Errorf("BFloat16 roundtrip(%v) = %v, want %v", tt.input, got, tt.want)
			}
		})
	}
}

func TestBFloat16Precision(t *testing.T) {
	// BFloat16 has ~7-bit mantissa — values with more precision get truncated
	val := float32(1.234375) // this is exactly representable in bf16 (1 + 15/64)
	bf16 := NewBFloat16(val)
	got := bf16.Float32()
	if got != val {
		t.Errorf("BFloat16(%v) = %v, want exact match", val, got)
	}

	// Value requiring more precision should lose some
	val2 := float32(1.001)
	bf16_2 := NewBFloat16(val2)
	got2 := bf16_2.Float32()
	if math.Abs(float64(got2-val2)) > 0.01 {
		t.Errorf("BFloat16(%v) = %v, too far off", val2, got2)
	}
}

func TestBFloat16Infinity(t *testing.T) {
	posInf := NewBFloat16(float32(math.Inf(1)))
	if !math.IsInf(float64(posInf.Float32()), 1) {
		t.Errorf("expected +Inf, got %v", posInf.Float32())
	}

	negInf := NewBFloat16(float32(math.Inf(-1)))
	if !math.IsInf(float64(negInf.Float32()), -1) {
		t.Errorf("expected -Inf, got %v", negInf.Float32())
	}
}

func TestBFloat16NaN(t *testing.T) {
	nan := NewBFloat16(float32(math.NaN()))
	if !math.IsNaN(float64(nan.Float32())) {
		t.Errorf("expected NaN, got %v", nan.Float32())
	}
}

func TestBFloat16ExponentRange(t *testing.T) {
	// BFloat16 has same exponent range as float32 — can represent very large/small values
	large := float32(1e38)
	bf16 := NewBFloat16(large)
	got := bf16.Float32()
	// Should be close (within bf16 precision)
	ratio := got / large
	if ratio < 0.99 || ratio > 1.01 {
		t.Errorf("BFloat16 large value: expected ~%v, got %v", large, got)
	}

	small := float32(1e-38)
	bf16s := NewBFloat16(small)
	gots := bf16s.Float32()
	ratios := gots / small
	if ratios < 0.9 || ratios > 1.1 {
		t.Errorf("BFloat16 small value: expected ~%v, got %v", small, gots)
	}
}
