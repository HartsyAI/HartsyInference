namespace HartsyInference.Engine;

/// <summary>What a catalog entry is for beyond generation. Code asks <see cref="CatalogEntry.Features"/> instead of comparing model ids,
/// so a new matte model is one catalog line rather than an edit in every consumer.</summary>
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
}
