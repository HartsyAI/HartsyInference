using System.Runtime.InteropServices;
using SharpInference.Core.Tensors;

namespace SharpInference.ModelHandler.Gguf.Codecs;

/// <summary>Base class providing default F16 fallback (via an F32 temp buffer) and a stock <see cref="QuantizeFromF32"/> that throws <see cref="NotSupportedException"/>. Codecs only need to override the methods they actually support.</summary>
public abstract unsafe class GgufCodecBase : IGgufCodec
{
    public abstract DType DType { get; }

    public virtual bool SupportsQuantize => false;

    public abstract void DequantizeToF32(byte* src, float* dst, long elementCount);

    public virtual void DequantizeToF16(byte* src, Half* dst, long elementCount)
    {
        nuint byteCount = (nuint)(elementCount * sizeof(float));
        float* tmp = (float*)NativeMemory.Alloc(byteCount);
        try
        {
            DequantizeToF32(src, tmp, elementCount);
            for (long i = 0; i < elementCount; i++) dst[i] = (Half)tmp[i];
        }
        finally
        {
            NativeMemory.Free(tmp);
        }
    }

    public virtual void QuantizeFromF32(float* src, byte* dst, long elementCount)
    {
        throw new NotSupportedException(
            $"Codec for {DType} does not implement quantize (read-only). Use llama.cpp's `quantize` tool or city96's GGUF dumps to author files of this dtype.");
    }
}

/// <summary>Helpers shared by codec implementations: 6-bit scale unpacking for K-quants (canonical ggml `get_scale_min_k4`), FP16↔F32 conversion, etc.</summary>
public static unsafe class GgufCodecHelpers
{
    /// <summary>Canonical ggml `get_scale_min_k4` — extracts a 6-bit scale and 6-bit min for sub-block <paramref name="j"/> (0..7) from a 12-byte packed scales array.</summary>
    public static void GetScaleMinK4(int j, byte* q, out byte d, out byte m)
    {
        if (j < 4)
        {
            d = (byte)(q[j] & 63);
            m = (byte)(q[j + 4] & 63);
        }
        else
        {
            d = (byte)((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
            m = (byte)((q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4));
        }
    }

    /// <summary>Reads 16-bit IEEE 754 half-precision float at the given byte pointer.</summary>
    public static float ReadHalf(byte* p) => (float)(*(Half*)p);
}
