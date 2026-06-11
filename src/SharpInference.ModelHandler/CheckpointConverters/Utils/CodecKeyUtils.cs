using SharpInference.Core.Tensors;

namespace SharpInference.ModelHandler.CheckpointConverters.Utils;

/// <summary>Shared key/tensor helpers for the audio-codec checkpoint converters (MusicGen, YuE). The engine codec
/// classes (<c>EnCodec</c>, <c>XCodec</c>/DAC) fuse PyTorch weight-norm pairs themselves at <c>LoadWeights</c> time,
/// so converters must keep raw <c>weight_g</c>/<c>weight_v</c> keys — these helpers only normalize the newer
/// <c>parametrizations.weight.original0/original1</c> spelling back to the classic pair names.</summary>
public static class CodecKeyUtils
{
    /// <summary>Renames torch ≥2.1 <c>*.parametrizations.weight.original0/original1</c> keys to the classic
    /// <c>*.weight_g</c>/<c>*.weight_v</c> spelling the engine codec loaders expect. Other keys pass through.</summary>
    public static string NormalizeWeightNormKey(string key)
    {
        key = key.Replace(".parametrizations.weight.original0", ".weight_g", StringComparison.Ordinal);
        return key.Replace(".parametrizations.weight.original1", ".weight_v", StringComparison.Ordinal);
    }

    /// <summary>Removes <paramref name="prefix"/> when <paramref name="key"/> starts with it; otherwise returns the key unchanged.</summary>
    public static string StripPrefix(string key, string prefix) =>
        key.StartsWith(prefix, StringComparison.Ordinal) ? key[prefix.Length..] : key;

    /// <summary>Materializes the tensor as F32 when requested (CPU inference); otherwise the mmap-borrowing tensor passes through.</summary>
    public static Tensor MaybeCast(Tensor tensor, bool castToF32) =>
        castToF32 && tensor.DType != DType.F32 ? tensor.CastTo(DType.F32) : tensor;
}
