using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends.Latent;

/// <summary>CPU reference for the in-place quantize-dequantize of latent rows.</summary>
public static class LatentQuantReference
{

    /// <summary>Checks the operands of <see cref="ActQuantDequantInPlace"/>.</summary>
    public static void ValidateActQuant(Tensor x, LatentEncoding encoding)
    {
        if (x.DType != DType.F32) throw new NotSupportedException("ActQuantDequantInPlace supports F32 only.");
        int group = LatentEncodings.GroupSize(encoding);
        if (group != 0 && (x.ElementCount == 0 || x.Shape[x.Shape.Rank - 1] % group != 0))
            throw new ArgumentException($"The last dimension must be a multiple of {group} for {encoding}.", nameof(x));
    }

    /// <summary>Replaces each value with its quantize-then-dequantize round trip, as the reference <c>act_quant(inplace=True)</c>.</summary>
    public static unsafe void ActQuantDequantInPlace(Tensor x, LatentEncoding encoding)
    {
        ValidateActQuant(x, encoding);
        int group = LatentEncodings.GroupSize(encoding);
        if (group == 0) return;
        float* p = (float*)x.DataPointer;
        byte[] codes = new byte[group];
        long groups = x.ElementCount / group;
        for (long g = 0; g < groups; g++)
        {
            Span<float> span = new Span<float>(p + g * group, group);
            float scale = LatentCodec.QuantizeGroup(encoding, span, codes, out byte _);
            for (int e = 0; e < group; e++) span[e] = LatentCodec.DecodeCode(encoding, codes[e]) * scale;
        }
    }
}
