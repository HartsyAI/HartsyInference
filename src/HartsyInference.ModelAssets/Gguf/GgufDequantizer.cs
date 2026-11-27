using HartsyInference.Core.Numerics;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Gguf.Codecs;

namespace HartsyInference.ModelAssets.Gguf;

/// <summary>Façade over <see cref="GgufCodecRegistry"/> — kept for back-compat with the original `GgufDequantizer.Dequantize` API. New code should reach for <see cref="GgufCodecRegistry"/> directly to access the per-codec interface (including the optional <see cref="IGgufCodec.QuantizeFromF32"/> direction for the future writer).</summary>
public static class GgufDequantizer
{
    /// <summary>Blocks per parallel range of the BF16 decode.</summary>
    private const long BlocksPerRange = 1024;

    /// <summary>Dequantizes a quantized tensor to <see cref="DType.F32"/>, <see cref="DType.F16"/> or <see cref="DType.BF16"/> (each value rounded
    /// to nearest, ties to even). Allocates a new tensor; the source is unchanged.</summary>
    public static unsafe Tensor Dequantize(Tensor source, DType targetDtype)
    {
        if (!source.DType.IsQuantized)
            throw new ArgumentException($"Source tensor is not quantized (dtype={source.DType}).", nameof(source));
        if (targetDtype != DType.F32 && targetDtype != DType.F16 && targetDtype != DType.BF16)
            throw new ArgumentException($"Target dtype must be F32, F16 or BF16, got {targetDtype}.", nameof(targetDtype));

        IGgufCodec codec = GgufCodecRegistry.Get(source.DType);
        Tensor result = new Tensor(source.Shape, targetDtype);
        long elementCount = source.Shape.ElementCount;

        if (targetDtype == DType.F32)
        {
            codec.DequantizeToF32((byte*)source.DataPointer, (float*)result.DataPointer, elementCount);
        }
        else if (targetDtype == DType.F16)
        {
            codec.DequantizeToF16((byte*)source.DataPointer, (Half*)result.DataPointer, elementCount);
        }
        else
        {
            DequantizeToBf16(codec, source, result);
        }
        return result;
    }

    /// <summary>Decodes whole blocks through a small F32 buffer, in parallel, so a large weight needs no F32 copy of itself.</summary>
    private static unsafe void DequantizeToBf16(IGgufCodec codec, Tensor source, Tensor result)
    {
        int perBlock = source.DType.BlockElementCount, blockBytes = source.DType.BlockByteSize;
        // Both pointers are taken here: a tensor's first host access allocates, which the workers must not race.
        nint from = (nint)source.DataPointer, to = (nint)result.DataPointer;
        CpuParallel.ForRanges(result.ElementCount / perBlock, BlocksPerRange, perBlock * 4L, (start, length) =>
        {
            float[] buffer = new float[length * perBlock];
            fixed (float* values = buffer)
            {
                codec.DequantizeToF32((byte*)from + start * blockBytes, values, buffer.Length);
                ushort* bits = (ushort*)to + start * perBlock;
                for (int i = 0; i < buffer.Length; i++)
                    bits[i] = BFloat16Bits.FromFloat(values[i]);
            }
        });
    }
}
