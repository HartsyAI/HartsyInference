namespace HartsyInference.Engine.Planning;

/// <summary>The part an entry in the built-in H3 artifact manifest plays in a composition.</summary>
internal enum VideoProfileArtifactRole
{
    /// <summary>A primary transformer checkpoint.</summary>
    Main = 0,

    /// <summary>A distillation adapter merged into a compatible main checkpoint.</summary>
    Adapter = 1,

    /// <summary>A ControlNet branch.</summary>
    ControlNet = 2,

    /// <summary>A separately selected video VAE.</summary>
    VideoVae = 3,

    /// <summary>An explicitly incompatible artifact.</summary>
    Rejected = 4,
}
