package onnxruntime

import "math"

// Float16 represents an IEEE 754 half-precision (16-bit) floating-point number.
type Float16 uint16

// NewFloat16 converts a float32 to Float16.
func NewFloat16(f float32) Float16 {
	bits := math.Float32bits(f)
	sign := (bits >> 16) & 0x8000
	exp := int((bits>>23)&0xFF) - 127
	frac := bits & 0x7FFFFF

	switch {
	case exp == 128:
		// Inf or NaN
		if frac == 0 {
			return Float16(sign | 0x7C00) // infinity
		}
		return Float16(sign | 0x7C00 | (frac >> 13) | 1) // NaN (preserve some payload, ensure non-zero frac)
	case exp > 15:
		// Overflow -> infinity
		return Float16(sign | 0x7C00)
	case exp > -15:
		// Normal range
		return Float16(sign | uint32(exp+15)<<10 | (frac >> 13))
	case exp >= -24:
		// Subnormal
		frac |= 0x800000
		shift := uint(-14 - exp)
		return Float16(sign | (frac >> (shift + 13)))
	default:
		// Underflow -> zero
		return Float16(sign)
	}
}

// Float32 converts a Float16 to float32.
func (f Float16) Float32() float32 {
	sign := uint32(f&0x8000) << 16
	exp := uint32(f>>10) & 0x1F
	frac := uint32(f) & 0x3FF

	switch {
	case exp == 0:
		if frac == 0 {
			// Zero
			return math.Float32frombits(sign)
		}
		// Subnormal -> normalize
		for frac&0x400 == 0 {
			frac <<= 1
			exp--
		}
		exp++
		frac &= 0x3FF
		return math.Float32frombits(sign | (exp+127-15)<<23 | frac<<13)
	case exp == 31:
		// Inf/NaN
		return math.Float32frombits(sign | 0x7F800000 | frac<<13)
	default:
		// Normal
		return math.Float32frombits(sign | (exp+127-15)<<23 | frac<<13)
	}
}

// BFloat16 represents a Brain Floating Point (16-bit) number.
// BFloat16 has the same exponent range as float32 but with reduced precision.
type BFloat16 uint16

// NewBFloat16 converts a float32 to BFloat16 by truncating the lower 16 bits.
func NewBFloat16(f float32) BFloat16 {
	bits := math.Float32bits(f)
	return BFloat16(bits >> 16)
}

// Float32 converts a BFloat16 to float32.
func (f BFloat16) Float32() float32 {
	return math.Float32frombits(uint32(f) << 16)
}
