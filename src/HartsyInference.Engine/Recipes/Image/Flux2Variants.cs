using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Engine.Variants;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.Checkpoints;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>FLUX.2 Klein 4B, Klein 9B and Dev, and the Iris-Z fine-tunes of the last two. The transformer's input
/// projection width is the hidden size, which names the build; Iris-Z adds a modality embedding. Only the names the converter's detection reads are structural: a checkpoint prefixed
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

    /// <summary>Iris-Z-small (speridlabs/iris-z): FLUX.2 Klein base 9B fine-tuned to generate images and depth.</summary>
    public static ModelVariant IrisZSmall { get; } = new ModelVariant
    {
        Id = "iris-z-small",
        DisplayName = "Iris-Z-small",
        HintAliases = ["iris-z-small"],
        StructuralMarkers = [Flux2CheckpointConverter.ModalityEmbeddingKey],
        StructuralMatch = probe => HiddenSize(probe) == 4096,
        StructureRequired = true,
        FineTuneOf = "klein-9b",
    };

    /// <summary>Iris-Z-big (speridlabs/iris-z): FLUX.2 Dev without its guidance embedding, fine-tuned the same way.</summary>
    public static ModelVariant IrisZBig { get; } = new ModelVariant
    {
        Id = "iris-z-big",
        DisplayName = "Iris-Z-big",
        HintAliases = ["iris-z-big"],
        StructuralMarkers = [Flux2CheckpointConverter.ModalityEmbeddingKey],
        StructuralMatch = probe => HiddenSize(probe) == 6144,
        StructureRequired = true,
        FineTuneOf = "dev",
    };

    /// <summary>The catalog <see cref="Flux2Recipe"/> declares. The Iris-Z builds come first: structure is tried in order,
    /// and their hidden sizes are also Klein 9B's and Dev's.</summary>
    public static ModelVariantCatalog Catalog { get; } =
        new ModelVariantCatalog("flux2", Klein4B, [IrisZSmall, IrisZBig, Klein4B, Klein9B, Dev]);

    /// <summary>True for the Iris-Z builds, which run <c>IrisZPipeline</c> instead of the FLUX.2 pipeline.</summary>
    public static bool IsIrisZ(ResolvedModelVariant? variant) => variant is not null && (variant.Is(IrisZSmall) || variant.Is(IrisZBig));

    /// <summary>The transformer configuration a resolved variant builds with.</summary>
    public static Flux2Config ConfigFor(ResolvedModelVariant variant)
    {
        ArgumentNullException.ThrowIfNull(variant);
        if (variant.Is(IrisZSmall))
        {
            return Flux2Config.IrisZSmall;
        }
        if (variant.Is(IrisZBig))
        {
            return Flux2Config.IrisZBig;
        }
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
