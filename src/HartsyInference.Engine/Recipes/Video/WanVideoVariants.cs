using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.Checkpoints;
using HartsyInference.Engine.Variants;

namespace HartsyInference.Engine.Recipes.Video;

/// <summary>The Wan conditioning variants that share a compat class. VACE, Animate and S2V carry their own modules, so
/// only the weights may select them; Animate-2 is key-for-key an I2V-14B checkpoint and is recognised by its
/// <c>__metadata__</c> config alone (a GGUF repack drops it, so an Animate-2 GGUF loads as the I2V backbone).</summary>
public static class WanVideoVariants
{
    /// <summary>Plain T2V / I2V backbone (default).</summary>
    public static ModelVariant Base { get; } = new ModelVariant { Id = "base", DisplayName = "Wan T2V/I2V" };

    /// <summary>VACE control branch (<c>vace_patch_embedding</c> / <c>vace_blocks.*</c>).</summary>
    public static ModelVariant Vace { get; } = new ModelVariant
    {
        Id = "vace",
        DisplayName = "Wan VACE",
        StructureRequired = true,
        StructuralMatch = probe => probe.AnyKeyContains("vace_patch_embedding", "vace_blocks"),
    };

    /// <summary>Wan-Animate pose + face pathway.</summary>
    public static ModelVariant Animate { get; } = new ModelVariant
    {
        Id = "animate",
        DisplayName = "Wan-Animate",
        StructureRequired = true,
        StructuralMatch = probe => probe.AnyKeyContains("pose_patch_embedding", "motion_encoder", "face_adapter"),
    };

    /// <summary>Wan2.2-S2V audio injector.</summary>
    public static ModelVariant S2V { get; } = new ModelVariant
    {
        Id = "s2v",
        DisplayName = "Wan S2V",
        StructureRequired = true,
        StructuralMatch = probe => probe.AnyKeyContains("audio_encoder", "audio_injector"),
    };

    /// <summary>TI2V-5B: the plain backbone at 48 latent channels, which additionally takes an end frame.</summary>
    public static ModelVariant Ti2V5B { get; } = new ModelVariant
    {
        Id = "ti2v-5b",
        DisplayName = "Wan TI2V-5B",
        StructureRequired = true,
        StructuralMatch = IsTi2V5B,
    };

    /// <summary>Wan-Animate-2 driving-video stream, recognised by <c>config.transformer.model_type == "animate2"</c>.</summary>
    public static ModelVariant Animate2 { get; } = new ModelVariant
    {
        Id = "animate2",
        DisplayName = "Wan-Animate-2",
        MetadataMatch = probe => WanVideoCheckpointConverter.IsAnimate2Metadata(probe.Metadata),
    };

    /// <summary>The catalog every <see cref="WanVideoRecipe"/> instance declares.</summary>
    public static ModelVariantCatalog Catalog { get; } = new ModelVariantCatalog("wan", Base, [Vace, Animate, S2V, Ti2V5B, Animate2, Base]);

    // The plain embedding only: VACE and Animate carry their own patch embeddings with other channel counts.
    private static bool IsTi2V5B(CheckpointProbe probe)
    {
        foreach (KeyValuePair<string, Core.Tensors.TensorShape> entry in probe.Shapes)
        {
            string key = entry.Key;
            if (key.EndsWith("patch_embedding.weight", StringComparison.Ordinal) && !key.Contains("vace_", StringComparison.Ordinal)
                && !key.Contains("pose_", StringComparison.Ordinal) && entry.Value.Rank == 5)
            {
                return entry.Value[1] == WanVideoConfig.Ti2V5B.InChannels;
            }
        }
        return false;
    }
}
