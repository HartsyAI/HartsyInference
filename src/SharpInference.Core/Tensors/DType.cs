namespace SharpInference.Core.Tensors;

/// <summary>Data types supported by SharpInference tensors.</summary>
public enum DType : byte
{
    /// <summary>32-bit IEEE 754 floating point.</summary>
    F32 = 0,

    /// <summary>16-bit IEEE 754 floating point (half precision).</summary>
    F16 = 1,

    /// <summary>16-bit Brain floating point (bfloat16).</summary>
    BF16 = 2,

    /// <summary>8-bit block quantization (32 values per block, 1 FP16 scale).</summary>
    Q8_0 = 3,

    /// <summary>4-bit block quantization with K-quant layout.</summary>
    Q4_K = 4,

    /// <summary>Signed 8-bit integer.</summary>
    I8 = 5,

    /// <summary>Unsigned 8-bit integer.</summary>
    U8 = 6,
}

/// <summary>Extension methods for <see cref="DType"/>.</summary>
public static class DTypeExtensions
{
    /// <summary>Returns the size in bytes of a single element for non-quantized types.</summary>
    public static int ElementSize(this DType dtype) => dtype switch
    {
        DType.F32 => 4,
        DType.F16 => 2,
        DType.BF16 => 2,
        DType.I8 => 1,
        DType.U8 => 1,
        DType.Q8_0 => throw new InvalidOperationException("Q8_0 is block-quantized; use BlockSize/BlockByteSize instead."),
        DType.Q4_K => throw new InvalidOperationException("Q4_K is block-quantized; use BlockSize/BlockByteSize instead."),
        _ => throw new ArgumentOutOfRangeException(nameof(dtype)),
    };

    /// <summary>Returns the number of elements per quantization block, or 1 for non-quantized types.</summary>
    public static int BlockSize(this DType dtype) => dtype switch
    {
        DType.Q8_0 => 32,
        DType.Q4_K => 256,
        _ => 1,
    };

    /// <summary>Returns the byte size of one quantization block.</summary>
    public static int BlockByteSize(this DType dtype) => dtype switch
    {
        DType.Q8_0 => 34,   // 2 bytes scale (FP16) + 32 bytes data
        DType.Q4_K => 144,  // TODO: verify exact Q4_K_M block size from GGUF research
        _ => dtype.ElementSize(),
    };

    /// <summary>Returns true if this dtype is a block-quantized format.</summary>
    public static bool IsQuantized(this DType dtype) => dtype is DType.Q8_0 or DType.Q4_K;

    /// <summary>Returns true if this dtype is a floating-point format (including quantized).</summary>
    public static bool IsFloatingPoint(this DType dtype) => dtype is DType.F32 or DType.F16 or DType.BF16;
}
