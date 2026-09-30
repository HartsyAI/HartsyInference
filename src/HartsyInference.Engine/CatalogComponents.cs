namespace HartsyInference.Engine;

/// <summary>The separable parts of a multi-part checkpoint, so a catalog variant can say which it ships.</summary>
[Flags]
public enum CatalogComponents
{
    /// <summary>No component.</summary>
    None = 0,

    /// <summary>The backbone layers, embedding and head.</summary>
    Backbone = 1,

    /// <summary>The draft (<c>mtp</c>) layers used for speculative decoding.</summary>
    Draft = 2,

    /// <summary>The image tower and aligner.</summary>
    Vision = 4,

    /// <summary>The Engram lookup tables.</summary>
    Engram = 8,
}
