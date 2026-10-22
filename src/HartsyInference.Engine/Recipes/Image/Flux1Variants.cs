using HartsyInference.ModelAssets.Checkpoints;
using HartsyInference.Engine.Variants;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>FLUX.1 Dev vs Schnell. The guidance embedder is the only weight difference, and it is what the converter
/// reads to decide the variant, so the weights prove it. The predicate mirrors
/// <c>FluxCheckpointConverter.DetectArchitecture</c> over the raw tensor names, including which prefixes and branch
/// (BFL or diffusers) each one is read under.</summary>
public static class Flux1Variants
{
    /// <summary>The guidance-distilled build (default): 28 steps at embedded guidance 3.5.</summary>
    public static ModelVariant Dev { get; } = new ModelVariant
    {
        Id = "dev",
        DisplayName = "FLUX.1 Dev",
        HintAliases = ["Flux.1-dev"],
        MetadataClassIds = ["Flux.1-dev"],
        StructuralMatch = HasGuidanceEmbedder,
    };

    /// <summary>The 4-step distilled build with no guidance embedder.</summary>
    public static ModelVariant Schnell { get; } = new ModelVariant
    {
        Id = "schnell",
        DisplayName = "FLUX.1 Schnell",
        HintAliases = ["Flux.1-schnell"],
        MetadataClassIds = ["Flux.1-schnell"],
        StructuralMatch = probe => !HasGuidanceEmbedder(probe) && HasTransformer(probe),
    };

    /// <summary>The catalog <see cref="Flux1Recipe"/> declares.</summary>
    public static ModelVariantCatalog Catalog { get; } = new ModelVariantCatalog("flux1", Dev, [Dev, Schnell]);

    /// <summary>True when the converter would emit a guidance embedder: a BFL checkpoint (<c>double_blocks</c>, optionally
    /// under <c>model.diffusion_model.</c>) whose <c>guidance_in</c> has an in- or out-layer, or a diffusers checkpoint
    /// carrying <c>time_text_embed.guidance_embedder</c>.</summary>
    private static bool HasGuidanceEmbedder(CheckpointProbe probe)
    {
        bool bfl = probe.Keys.Any(key => key.StartsWith("double_blocks.", StringComparison.Ordinal)
            || key.StartsWith("model.diffusion_model.double_blocks.", StringComparison.Ordinal));
        if (!bfl)
        {
            return probe.Keys.Any(key => key.StartsWith("time_text_embed.guidance_embedder", StringComparison.Ordinal));
        }
        return probe.Keys.Any(key =>
        {
            string bflKey = key.StartsWith("model.diffusion_model.", StringComparison.Ordinal)
                ? key["model.diffusion_model.".Length..] : key;
            return bflKey.StartsWith("guidance_in.in_layer.", StringComparison.Ordinal)
                || bflKey.StartsWith("guidance_in.out_layer.", StringComparison.Ordinal);
        });
    }

    /// <summary>A diffusion transformer is present; a file without one is not a Flux.1 checkpoint at all.</summary>
    private static bool HasTransformer(CheckpointProbe probe) =>
        probe.AnyKeyContains("double_blocks.", "transformer_blocks.");
}
