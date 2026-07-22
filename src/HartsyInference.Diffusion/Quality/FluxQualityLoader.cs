using HartsyInference.Core.Logging;
using HartsyInference.ModelAssets.CheckpointConverters;

namespace HartsyInference.Diffusion.Quality;

/// <summary>Applies a <see cref="QualityProfile"/> to Flux converter output. Call between <see cref="FluxCheckpointConverter.LoadAndConvert"/> and the per-component <c>LoadWeights</c> calls. Mutates the dictionaries in place; original tensors are disposed after the cast.
///
/// <para>Example:</para>
/// <code>
/// (FluxCheckpointConverter.ConvertedWeights converted, SafeTensorsLoader loader) =
///     FluxCheckpointConverter.LoadAndConvert(path);
/// QualityProfile profile = QualityProfile.From(QualityPreset.High);  // FP8 backbone, FP16 encoders/VAE
/// FluxQualityLoader.Apply(converted, profile);
/// // ...then construct FluxTransformer / ClipTextEncoder / T5TextEncoder / VaeDecoder and call LoadWeights as usual.
/// </code></summary>
public static class FluxQualityLoader
{
    /// <summary>Casts each component dict per the profile. <see cref="QualityProfile.Validate"/> runs first — throws on invalid combinations (e.g. FP8 VAE).</summary>
    public static void Apply(FluxCheckpointConverter.ConvertedWeights weights, QualityProfile profile)
    {
        profile.Validate();
        int t = QualityProfileApplier.Apply(weights.Transformer, profile.BackboneDType);
        int c = QualityProfileApplier.Apply(weights.ClipL, profile.TextEncoderDType);
        int t5 = QualityProfileApplier.Apply(weights.T5, profile.TextEncoderDType);
        int v = QualityProfileApplier.Apply(weights.Vae, profile.VaeDType);
        Logs.Info($"FluxQualityLoader: applied profile (backbone={profile.BackboneDType}, text={profile.TextEncoderDType}, vae={profile.VaeDType}). Casts: transformer={t}, clipL={c}, t5={t5}, vae={v}.");
    }
}
