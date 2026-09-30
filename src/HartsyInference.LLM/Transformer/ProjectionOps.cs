using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.LLM.Transformer;

/// <summary>Shared projection, dtype and normalization helpers for the LLM transformer family; <see cref="GenericTransformer"/> delegates here so other decoders reuse the exact same dispatch.</summary>
internal static class ProjectionOps
{
    /// <summary>Projection dispatch: float weights always take cuBLAS <see cref="IBackend.Linear"/>; quantized weights take the low-VRAM <see cref="IBackend.QuantizedMatMul"/> when <paramref name="lowVram"/> is set (weight stays compressed, transient dequant), else the faster <see cref="IBackend.Linear"/> path (dequants + caches an F16 weight).</summary>
    public static void Project(IBackend backend, Tensor output, Tensor input, Tensor weight, Tensor? bias, bool lowVram)
    {
        if (weight.DType.IsQuantized && lowVram) backend.QuantizedMatMul(output, input, weight, bias);
        else backend.Linear(output, input, weight, bias);
    }

    /// <summary><see cref="Project"/> with <see cref="IBackend.HighPrecisionGemm"/> forced on for this call, so F32 operands skip TF32; the previous setting is restored even when the projection throws.</summary>
    public static void ProjectPrecise(IBackend backend, Tensor output, Tensor input, Tensor weight, Tensor? bias, bool lowVram)
    {
        bool previous = backend.HighPrecisionGemm;
        backend.HighPrecisionGemm = true;
        try
        {
            Project(backend, output, input, weight, bias, lowVram);
        }
        finally
        {
            backend.HighPrecisionGemm = previous;
        }
    }

    /// <summary>F32 view-or-copy: dequantizes quantized tensors, casts 16-bit floats, returns the SAME reference when already F32 (callers check <c>ReferenceEquals</c> before disposing).</summary>
    /// <remarks>A ComfyUI <c>int8_tensorwise</c> weight is plain I8 whose scale — and, under ConvRot, whose basis —
    /// lives on <see cref="Tensor.QuantInfo"/>, so it is not <c>IsQuantized</c> and a cast would read a different
    /// weight rather than refuse. The embedding table of an int8 checkpoint reaches here, and it is host-gathered.</remarks>
    public static Tensor EnsureF32(Tensor t)
    {
        if (t.DType == DType.F32) return t;
        if (t.DType.IsQuantized) return HartsyInference.ModelAssets.Gguf.GgufDequantizer.Dequantize(t, DType.F32);
        if (t.DType == DType.I8 && t.QuantInfo is { RowScale: not null } int8)
        {
            using Tensor bf16 = Int8ConvRotCodec.DequantToBf16(t, int8.RowScale, int8.ConvRotGroupSize);
            return bf16.CastTo(DType.F32);
        }
        return t.CastTo(DType.F32);
    }

    /// <summary>Normalizes <paramref name="input"/> with <paramref name="weight"/> using LayerNorm (mean-centered, Cohere) or RMSNorm (everything else).</summary>
    public static void Normalize(IBackend backend, Tensor output, Tensor input, Tensor weight, Tensor? bias, bool layerNorm, float eps)
    {
        if (layerNorm) backend.LayerNorm(output, input, weight, bias!, eps);
        else backend.RmsNorm(output, input, weight, eps);
    }
}
