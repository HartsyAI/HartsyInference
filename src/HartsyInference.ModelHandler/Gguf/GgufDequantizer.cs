using HartsyInference.Core.Tensors;
using HartsyInference.ModelHandler.Gguf.Codecs;

namespace HartsyInference.ModelHandler.Gguf;

/// <summary>Façade over <see cref="GgufCodecRegistry"/> — kept for back-compat with the original `GgufDequantizer.Dequantize` API. New code should reach for <see cref="GgufCodecRegistry"/> directly to access the per-codec interface (including the optional <see cref="IGgufCodec.QuantizeFromF32"/> direction for the future writer).</summary>
public static class GgufDequantizer
{
    /// <summary>Dequantizes a quantized tensor to <see cref="DType.F32"/> or <see cref="DType.F16"/>. Allocates a new tensor; the source is unchanged.</summary>
    public static unsafe Tensor Dequantize(Tensor source, DType targetDtype)
    {
        if (!source.DType.IsQuantized)
            throw new ArgumentException($"Source tensor is not quantized (dtype={source.DType}).", nameof(source));
        if (targetDtype != DType.F32 && targetDtype != DType.F16)
            throw new ArgumentException($"Target dtype must be F32 or F16, got {targetDtype}.", nameof(targetDtype));

        IGgufCodec codec = GgufCodecRegistry.Get(source.DType);
        Tensor result = new Tensor(source.Shape, targetDtype);
        long elementCount = source.Shape.ElementCount;

        if (targetDtype == DType.F32)
        {
            codec.DequantizeToF32((byte*)source.DataPointer, (float*)result.DataPointer, elementCount);
        }
        else
        {
            codec.DequantizeToF16((byte*)source.DataPointer, (Half*)result.DataPointer, elementCount);
        }
        return result;
    }
}
