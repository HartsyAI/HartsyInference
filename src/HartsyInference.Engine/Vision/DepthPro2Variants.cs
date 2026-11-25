using HartsyInference.Core.Tensors;
using HartsyInference.Engine.Variants;
using HartsyInference.ModelAssets.Checkpoints;
using HartsyInference.Vision.DepthPro2;

namespace HartsyInference.Engine.Vision;

/// <summary>Depth Pro 2's four sizes as one variant family, told apart by the backbone width of the checkpoint itself, so a file named by
/// <c>--model-path</c> loads with the right preset. The catalog ids (<c>depthpro2-s</c> …) are the variant ids and serve as hints.</summary>
public static class DepthPro2Variants
{
    private const string QkvKey = "encoder.blocks.0.attn.qkv.weight";
    private const string HeadKey = "head.4.weight";

    /// <summary><c>apple/DepthPro2-S</c> (ViT-S/16), the default.</summary>
    public static ModelVariant Small { get; } = Variant(DepthPro2Preset.Small, "Depth Pro 2 S (ViT-S/16)");

    /// <summary><c>apple/DepthPro2-B</c> (ViT-B/16).</summary>
    public static ModelVariant Base { get; } = Variant(DepthPro2Preset.Base, "Depth Pro 2 B (ViT-B/16)");

    /// <summary><c>apple/DepthPro2-L</c> (ViT-L/16).</summary>
    public static ModelVariant Large { get; } = Variant(DepthPro2Preset.Large, "Depth Pro 2 L (ViT-L/16)");

    /// <summary><c>apple/DepthPro2-H</c> (ViT-H+/16).</summary>
    public static ModelVariant HugePlus { get; } = Variant(DepthPro2Preset.HugePlus, "Depth Pro 2 H+ (ViT-H+/16)");

    /// <summary>The family's catalog.</summary>
    public static ModelVariantCatalog Catalog { get; } = new ModelVariantCatalog("depthpro2", Small, [Small, Base, Large, HugePlus]);

    /// <summary>True when the checkpoint's own tensors are a Depth Pro 2 depth network of one of the four sizes.</summary>
    public static bool IsDepthPro2(CheckpointProbe probe)
    {
        ArgumentNullException.ThrowIfNull(probe);
        return Catalog.Variants.Any(v => probe.HasAllKeys(v.StructuralMarkers) && v.StructuralMatch!(probe));
    }

    /// <summary>The preset of a resolved variant.</summary>
    public static DepthPro2Preset PresetFor(ResolvedModelVariant variant)
    {
        ArgumentNullException.ThrowIfNull(variant);
        return DepthPro2Preset.All.First(p => string.Equals(p.Name, variant.Id, StringComparison.OrdinalIgnoreCase));
    }

    private static ModelVariant Variant(DepthPro2Preset preset, string displayName) => new ModelVariant
    {
        Id = preset.Name, DisplayName = displayName, StructuralMarkers = [QkvKey, HeadKey],
        StructuralMatch = probe => probe.Shapes.TryGetValue(QkvKey, out TensorShape shape) && shape.Rank == 2
            && shape[0] == 3 * preset.Backbone.HiddenSize && shape[1] == preset.Backbone.HiddenSize,
    };
}
