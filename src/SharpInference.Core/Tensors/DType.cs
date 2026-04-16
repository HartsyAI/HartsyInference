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

    /// <summary>8-bit block quantization (32 values per block, 1 FP16 scale).</summary>
    public static readonly DType Q8_0 = new("Q8_0", 0, true, 34, 32);

    /// <summary>4-bit block quantization with K-quant layout (256 values per super-block).</summary>
    public static readonly DType Q4_K = new("Q4_K", 0, true, 144, 256);

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

    /// <summary>Whether this dtype is a floating-point format (F32, F16, or BF16).</summary>
    public bool IsFloatingPoint => this == F32 || this == F16 || this == BF16;

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
