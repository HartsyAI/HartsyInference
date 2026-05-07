using System.Diagnostics;

namespace SharpInference.Core.Tensors;

/// <summary>Describes a tensor element data type with inline metadata for quantized block formats.</summary>
public readonly record struct DType(string Name, int SizeInBytes, bool IsQuantized, int BlockByteSize = 0, int BlockElementCount = 1)
{
    /// <summary>32-bit IEEE 754 floating point.</summary>
    public static readonly DType F32 = new("F32", 4, false);

    /// <summary>16-bit IEEE 754 floating point (half precision).</summary>
    public static readonly DType F16 = new("F16", 2, false);

    /// <summary>16-bit Brain floating point (bfloat16).</summary>
    public static readonly DType BF16 = new("BF16", 2, false);

    /// <summary>8-bit floating point E4M3 (4-bit exponent, 3-bit mantissa, no NaN). Standard for FP8 inference weights.</summary>
    public static readonly DType F8E4M3 = new("F8_E4M3", 1, false);

    /// <summary>8-bit floating point E5M2 (5-bit exponent, 2-bit mantissa). Wider range, lower precision than E4M3.</summary>
    public static readonly DType F8E5M2 = new("F8_E5M2", 1, false);

    // ── Legacy 32-block quants (ggml type IDs 2-9) ─────────────────────

    /// <summary>4-bit quantization, 32 values per block: 2 bytes FP16 scale + 16 bytes packed nibbles. Reconstruction: <c>x = scale * (q - 8)</c> where q ∈ [0..15].</summary>
    public static readonly DType Q4_0 = new("Q4_0", 0, true, 18, 32);

    /// <summary>4-bit quantization with min, 32 values per block: 2 bytes FP16 scale + 2 bytes FP16 min + 16 bytes packed nibbles. Reconstruction: <c>x = scale * q + min</c>.</summary>
    public static readonly DType Q4_1 = new("Q4_1", 0, true, 20, 32);

    /// <summary>5-bit quantization, 32 values per block: 2 bytes FP16 scale + 4 bytes high-bit packing + 16 bytes low-nibble packing. Reconstruction: <c>x = scale * (q - 16)</c> where q ∈ [0..31].</summary>
    public static readonly DType Q5_0 = new("Q5_0", 0, true, 22, 32);

    /// <summary>5-bit quantization with min, 32 values per block: 2 scale + 2 min + 4 high bits + 16 low nibbles.</summary>
    public static readonly DType Q5_1 = new("Q5_1", 0, true, 24, 32);

    /// <summary>8-bit block quantization (32 values per block, 1 FP16 scale).</summary>
    public static readonly DType Q8_0 = new("Q8_0", 0, true, 34, 32);

    /// <summary>8-bit block quantization with cached row sums (used as activation quant during inference). 32 values, 4 bytes scale + 4 bytes sum + 32 bytes int8.</summary>
    public static readonly DType Q8_1 = new("Q8_1", 0, true, 40, 32);

    // ── K-quants (ggml type IDs 10-15) ─────────────────────────────────

    /// <summary>2-bit K-quant. 256 values / 84 bytes super-block. Mixed 2-bit and 4-bit packing.</summary>
    public static readonly DType Q2_K = new("Q2_K", 0, true, 84, 256);

    /// <summary>3-bit K-quant. 256 values / 110 bytes super-block.</summary>
    public static readonly DType Q3_K = new("Q3_K", 0, true, 110, 256);

    /// <summary>4-bit K-quant. 256 values / 144 bytes super-block. 6-bit scales/mins per sub-block.</summary>
    public static readonly DType Q4_K = new("Q4_K", 0, true, 144, 256);

    /// <summary>5-bit K-quant. 256 values / 176 bytes super-block (144 like Q4_K + 32 bytes high bits).</summary>
    public static readonly DType Q5_K = new("Q5_K", 0, true, 176, 256);

    /// <summary>6-bit K-quant. 256 values / 210 bytes super-block. Highest-quality k-quant.</summary>
    public static readonly DType Q6_K = new("Q6_K", 0, true, 210, 256);

    /// <summary>8-bit K-quant. 256 values / 292 bytes. Used as an intermediate in mixed-quant matmul, rare for stored weights.</summary>
    public static readonly DType Q8_K = new("Q8_K", 0, true, 292, 256);

    // ── i-quants (ggml type IDs 16-29) — importance-weighted with lookup tables ─────────────

    /// <summary>i-quant 2-bit XXS. 256 values / 66 bytes. Uses lookup tables.</summary>
    public static readonly DType IQ2_XXS = new("IQ2_XXS", 0, true, 66, 256);

    /// <summary>i-quant 2-bit XS. 256 values / 74 bytes.</summary>
    public static readonly DType IQ2_XS = new("IQ2_XS", 0, true, 74, 256);

    /// <summary>i-quant 3-bit XXS. 256 values / 98 bytes.</summary>
    public static readonly DType IQ3_XXS = new("IQ3_XXS", 0, true, 98, 256);

    /// <summary>i-quant 1-bit S. 256 values / 50 bytes.</summary>
    public static readonly DType IQ1_S = new("IQ1_S", 0, true, 50, 256);

    /// <summary>i-quant 4-bit non-linear. 32 values / 18 bytes per block (block-32 layout, lookup table for codepoints).</summary>
    public static readonly DType IQ4_NL = new("IQ4_NL", 0, true, 18, 32);

    /// <summary>i-quant 3-bit S. 256 values / 110 bytes.</summary>
    public static readonly DType IQ3_S = new("IQ3_S", 0, true, 110, 256);

    /// <summary>i-quant 2-bit S. 256 values / 82 bytes.</summary>
    public static readonly DType IQ2_S = new("IQ2_S", 0, true, 82, 256);

    /// <summary>i-quant 4-bit XS. 256 values / 136 bytes (super-block layout).</summary>
    public static readonly DType IQ4_XS = new("IQ4_XS", 0, true, 136, 256);

    /// <summary>i-quant 1-bit M. 256 values / 56 bytes.</summary>
    public static readonly DType IQ1_M = new("IQ1_M", 0, true, 56, 256);

    // ── Ternary (ggml type IDs 31-32) ──────────────────────────────────

    /// <summary>Ternary 1.6875-bit. 256 values / 54 bytes.</summary>
    public static readonly DType TQ1_0 = new("TQ1_0", 0, true, 54, 256);

    /// <summary>Ternary 2.0625-bit. 256 values / 66 bytes.</summary>
    public static readonly DType TQ2_0 = new("TQ2_0", 0, true, 66, 256);

    /// <summary>Signed 8-bit integer.</summary>
    public static readonly DType I8 = new("I8", 1, false);

    /// <summary>Unsigned 8-bit integer.</summary>
    public static readonly DType U8 = new("U8", 1, false);

    /// <summary>Signed 32-bit integer.</summary>
    public static readonly DType I32 = new("I32", 4, false);

    /// <summary>Signed 64-bit integer.</summary>
    public static readonly DType I64 = new("I64", 8, false);

    /// <summary>Boolean (1 byte per element).</summary>
    public static readonly DType Bool = new("BOOL", 1, false);

    /// <summary>64-bit IEEE 754 floating point (double precision).</summary>
    public static readonly DType F64 = new("F64", 8, false);

    /// <summary>Whether this dtype is a floating-point format (F32, F16, BF16, F8E4M3, F8E5M2).</summary>
    public bool IsFloatingPoint => this == F32 || this == F16 || this == BF16 || this == F8E4M3 || this == F8E5M2;

    /// <summary>Whether this dtype is an FP8 format.</summary>
    public bool IsFp8 => this == F8E4M3 || this == F8E5M2;

    /// <summary>Computes total byte count for a given element count. Asserts block alignment for quantized types.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public long ComputeByteCount(long elementCount)
    {
        if (IsQuantized)
        {
            Debug.Assert(elementCount % BlockElementCount == 0,
                $"Element count {elementCount} must be a multiple of block element count {BlockElementCount} for {Name}.");
            return (elementCount / BlockElementCount) * BlockByteSize;
        }

        return elementCount * SizeInBytes;
    }

    /// <summary>Returns the Name for diagnostic display.</summary>
    public override string ToString() => Name;
}
