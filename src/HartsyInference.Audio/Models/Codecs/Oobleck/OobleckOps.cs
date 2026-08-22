using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Codecs.Oobleck;

/// <summary>Shared load-time transforms for the Oobleck VAE: weight-norm fusion (reusing the codec-wide <see cref="WeightNormFusion"/> helpers) and the logscale snake parameter exponentiation that distinguishes Oobleck's Snake1d from DAC's (parameters are stored as logs; the runtime value is <c>exp(param)</c>, computed once here so the hot path uses <see cref="HartsyInference.Core.Backends.IBackend.Snake"/> directly).</summary>
internal static unsafe class OobleckOps
{
    /// <summary>Loads a snake's <c>{prefix}.alpha</c> / <c>{prefix}.beta</c> (stored <c>[1, dim, 1]</c>, logscale) as exponentiated <c>[dim]</c> tensors ready for the snake-beta kernel.</summary>
    public static (Tensor Alpha, Tensor Beta) LoadSnake(IReadOnlyDictionary<string, Tensor> w, string prefix, int dim)
    {
        Tensor alpha = ExpReshaped(w[$"{prefix}.alpha"], dim);
        Tensor beta = ExpReshaped(w[$"{prefix}.beta"], dim);
        return (alpha, beta);
    }

    /// <summary>Fuses a weight-normed Conv1d's <c>{prefix}.weight_g</c> / <c>.weight_v</c>.</summary>
    public static Tensor LoadFusedWeight(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        Tensor g = WhisperOps.EnsureF32(w[$"{prefix}.weight_g"]);
        Tensor v = WhisperOps.EnsureF32(w[$"{prefix}.weight_v"]);
        return WeightNormFusion.Fuse(g, v);
    }

    /// <summary>Fuses a weight-normed ConvTranspose1d's <c>{prefix}.weight_g</c> / <c>.weight_v</c>.</summary>
    /// <remarks>diffusers wraps transpose convs with torch's DEFAULT dim-0 weight_norm, so the norm is per
    /// INPUT channel (g shape <c>[C_in, 1, 1]</c>, verified against the ACE-Step 1.5 vae safetensors
    /// header) — the axis-0-keyed <see cref="WeightNormFusion.Fuse"/> applies verbatim. This differs
    /// from EnCodec's per-output-channel <c>WeightNormFusionT</c> convention.</remarks>
    public static Tensor LoadFusedTransposeWeight(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        Tensor g = WhisperOps.EnsureF32(w[$"{prefix}.weight_g"]);
        Tensor v = WhisperOps.EnsureF32(w[$"{prefix}.weight_v"]);
        return WeightNormFusion.Fuse(g, v);
    }

    private static Tensor ExpReshaped(Tensor raw, int dim)
    {
        Tensor f32 = WhisperOps.EnsureF32(raw);
        if (f32.Shape.ElementCount != dim)
            throw new ArgumentException($"Snake parameter has {f32.Shape.ElementCount} elements; expected {dim}.");
        Tensor result = new(new TensorShape(dim), DType.F32);
        float* src = (float*)f32.DataPointer;
        float* dst = (float*)result.DataPointer;
        for (int i = 0; i < dim; i++) dst[i] = MathF.Exp(src[i]);
        return result;
    }
}
