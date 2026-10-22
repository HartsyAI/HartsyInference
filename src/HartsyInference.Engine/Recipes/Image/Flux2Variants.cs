using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Engine.Variants;
using HartsyInference.ModelAssets.Checkpoints;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>FLUX.2 Klein 4B, Klein 9B and Dev. The transformer's input projection width is the hidden size, which
/// names the build. Only the names the converter's detection reads are structural: a checkpoint prefixed
/// <c>model.diffusion_model.</c> carries no such name, so it resolves from its hints, metadata or file name, as before.</summary>
public static class Flux2Variants
{
    /// <summary>The projection tensors whose leading dimension is the hidden size, in the order detection tries them.</summary>
    private static readonly string[] HiddenSizeTensors = ["img_in.weight", "x_embedder.weight", "patch_embed.proj.weight"];

    /// <summary>Klein 4B, the step-distilled, guidance-free build (default when no evidence matches).</summary>
    public static ModelVariant Klein4B { get; } = new ModelVariant
    {
        Id = "klein-4b",
        DisplayName = "FLUX.2 Klein 4B",
        HintAliases = ["flux.2-klein-4b"],
        MetadataClassIds = ["flux.2-klein-4b"],
        StructuralMatch = probe => HiddenSize(probe) == 3072,
        FilenameTokenSets = [["klein", "4b"], ["klein4b"]],
    };

    /// <summary>Klein 9B, the step-distilled, guidance-free build with the larger text encoder.</summary>
    public static ModelVariant Klein9B { get; } = new ModelVariant
    {
        Id = "klein-9b",
        DisplayName = "FLUX.2 Klein 9B",
        HintAliases = ["flux.2-klein-9b"],
        MetadataClassIds = ["flux.2-klein-9b"],
        StructuralMatch = probe => HiddenSize(probe) == 4096,
        FilenameTokenSets = [["klein", "9b"], ["klein9b"]],
    };

    /// <summary>The guidance-embedded Dev build with the Mistral text encoder.</summary>
    public static ModelVariant Dev { get; } = new ModelVariant
    {
        Id = "dev",
        DisplayName = "FLUX.2 Dev",
        HintAliases = ["flux.2-dev"],
        MetadataClassIds = ["flux.2-dev"],
        StructuralMatch = probe => HiddenSize(probe) == 6144,
        FilenameTokenSets = [["flux2", "dev"], ["flux", "2", "dev"]],
    };

    /// <summary>The catalog <see cref="Flux2Recipe"/> declares.</summary>
    public static ModelVariantCatalog Catalog { get; } = new ModelVariantCatalog("flux2", Klein4B, [Klein4B, Klein9B, Dev]);

    /// <summary>The transformer configuration a resolved variant builds with.</summary>
    public static Flux2Config ConfigFor(ResolvedModelVariant variant)
    {
        ArgumentNullException.ThrowIfNull(variant);
        if (variant.Is(Dev))
        {
            return Flux2Config.Dev;
        }
        return variant.Is(Klein9B) ? Flux2Config.Klein9B : Flux2Config.Klein4B;
    }

    /// <summary>The leading dimension of the first projection tensor present, or null when none is or its rank is 0.</summary>
    private static long? HiddenSize(CheckpointProbe probe)
    {
        foreach (string name in HiddenSizeTensors)
        {
            if (!probe.Keys.Contains(name))
            {
                continue;
            }
            return probe.Shapes.TryGetValue(name, out TensorShape shape) && shape.Rank >= 1 ? shape[0] : null;
        }
        return null;
    }
}
