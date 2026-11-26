namespace HartsyInference.Engine;

/// <summary>What a catalog entry is for beyond generation. Code asks <see cref="CatalogEntry.Features"/> instead of comparing model ids,
/// so a consumer routes on the role a model plays rather than on its name.</summary>
[Flags]
public enum CatalogFeatures
{
    /// <summary>No extra role: an ordinary generator.</summary>
    None = 0,

    /// <summary>A matter <c>ImageRequest.RemoveBackgroundModel</c> may name.</summary>
    BackgroundRemoval = 1,

    /// <summary>Draws its matte as an edit image that the matte step turns into alpha (RMBG 3.0).</summary>
    AlphaMatte = 2,

    /// <summary>Large enough that the generating pipeline must be released before the matter loads.</summary>
    ReleasesGeneratorFirst = 4,

    /// <summary>An MDX-Net vocal separator <c>fx separate</c> runs through the MDX loop. The weights loader is Kim_Vocal_2's, so a second entry needs its own.</summary>
    MdxNetSeparator = 8,
}
