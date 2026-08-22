using HartsyInference.Audio.Models.Codecs;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Vits;

/// <summary>Weight-loading helpers for VITS modules; most VITS convolutions are PyTorch <c>weight_norm</c>'d (stored as <c>weight_g</c> + <c>weight_v</c>), so this fuses them at load (or takes a pre-fused <c>weight</c>) and model code only ever sees a plain conv weight.</summary>
public static class VitsWeights
{
    /// <summary>Loads conv weight <paramref name="key"/>: prefers a fused <c>{key}.weight</c>, else fuses <c>{key}.weight_g</c> + <c>{key}.weight_v</c>.</summary>
    public static Tensor Conv(IReadOnlyDictionary<string, Tensor> w, string key)
    {
        if (w.TryGetValue($"{key}.weight", out Tensor? fused)) return WhisperOps.EnsureF32(fused);
        return WeightNormFusion.Fuse(w[$"{key}.weight_g"], w[$"{key}.weight_v"]);
    }

    public static Tensor? Bias(IReadOnlyDictionary<string, Tensor> w, string key)
        => w.TryGetValue($"{key}.bias", out Tensor? b) ? WhisperOps.EnsureF32(b) : null;
}
