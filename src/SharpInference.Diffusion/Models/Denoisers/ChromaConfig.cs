namespace SharpInference.Diffusion.Models.Denoisers;

/// <summary>Configuration for Chroma (8.9B Flux derivative by Lodestone Rock). Architecturally identical to Flux with standard CFG instead of distilled guidance. Reuses FluxTransformer directly.</summary>
public record ChromaConfig
{
    /// <summary>Underlying Flux transformer config. Chroma uses the same architecture as Flux.1 Dev.</summary>
    public required FluxConfig TransformerConfig { get; init; }

    /// <summary>Whether to use standard classifier-free guidance. Always true for Chroma (not distilled to 1 like Flux Dev).</summary>
    public bool UseClassifierFreeGuidance { get; init; } = true;

    /// <summary>Default CFG scale for Chroma. Typical range: 5.0-9.0.</summary>
    public float DefaultCfgScale { get; init; } = 7.5f;

    /// <summary>Chroma preset (8.9B params, standard CFG, same block counts as Flux.1).</summary>
    public static ChromaConfig Default => new()
    {
        TransformerConfig = new FluxConfig
        {
            HiddenSize = 3072,
            NumHeads = 24,
            HeadDim = 128,
            Depth = 19,
            DepthSingleBlocks = 38,
            GuidanceEmbed = false,
        },
        UseClassifierFreeGuidance = true,
        DefaultCfgScale = 7.5f,
    };
}
