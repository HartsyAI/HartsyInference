using HartsyInference.Engine.Variants;

namespace HartsyInference.Engine.Recipes.Video;

/// <summary>LTX-2 dev vs distilled. The 2.5 dev and distilled transformers share a model version, config and tensor
/// keys, so nothing in the weights separates them: an explicit hint (<c>-m ltx-2.5:distilled</c>, the legacy
/// <c>ltx-2.5-distilled</c> id, a <c>hartsy.model_id</c> stamp) decides, and failing that a <c>distilled</c> token in
/// the file or bundle names is the logged last resort. The distilled variant routes to its own registered recipe,
/// which carries the 8-step unguided sampling contract.</summary>
public static class LtxVideo2Variants
{
    /// <summary>Family id the distilled <see cref="LtxVideo2Recipe"/> instance is registered under.</summary>
    public const string DistilledFamilyId = "ltx-2.5-distilled";

    /// <summary>The dev/base build (default).</summary>
    public static ModelVariant Dev { get; } = new ModelVariant { Id = "dev", DisplayName = "LTX-2 Dev" };

    /// <summary>The distilled 2.5 build, served by the <see cref="DistilledFamilyId"/> recipe.</summary>
    public static ModelVariant Distilled { get; } = new ModelVariant
    {
        Id = "distilled",
        DisplayName = "LTX-2.5 Distilled",
        HintAliases = [DistilledFamilyId],
        RoutesToFamily = DistilledFamilyId,
        FilenameTokenSets = [["distilled"]],
    };

    /// <summary>The catalog the dev-family recipe declares.</summary>
    public static ModelVariantCatalog Catalog { get; } = new ModelVariantCatalog("ltx-video-2", Dev, [Distilled, Dev]);
}
