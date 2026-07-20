using HartsyInference.Core.Logging;
using HartsyInference.ModelAssets.CheckpointConverters;

namespace HartsyInference.Diffusion.Quality;

/// <summary>Applies a <see cref="QualityProfile"/> to SDXL converter output. Call between <see cref="SdxlCheckpointConverter.LoadAndConvert"/> and the per-component <c>LoadWeights</c> calls. Mutates the dictionaries in place.</summary>
public static class SdxlQualityLoader
{
    /// <summary>Casts each component dict per the profile. <see cref="QualityProfile.Validate"/> runs first — throws on invalid combinations (e.g. FP8 VAE). SDXL has both CLIP-L and CLIP-G encoders; both are cast to <see cref="QualityProfile.TextEncoderDType"/>.</summary>
    public static void Apply(SdxlCheckpointConverter.ConvertedWeights weights, QualityProfile profile)
    {
        profile.Validate();
        int u = QualityProfileApplier.Apply(weights.UNet, profile.BackboneDType);
        int cL = QualityProfileApplier.Apply(weights.ClipL, profile.TextEncoderDType);
        int cG = QualityProfileApplier.Apply(weights.ClipG, profile.TextEncoderDType);
        int v = QualityProfileApplier.Apply(weights.Vae, profile.VaeDType);
        Logs.Info($"SdxlQualityLoader: applied profile (backbone={profile.BackboneDType}, text={profile.TextEncoderDType}, vae={profile.VaeDType}). Casts: unet={u}, clipL={cL}, clipG={cG}, vae={v}.");
    }
}
