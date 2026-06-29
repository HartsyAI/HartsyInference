namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Configuration for Flux.1 Tools variants (Fill, Redux, Canny, Depth) and Kontext. These share the core Flux transformer architecture but add conditioning adapters for specific tasks. The base FluxTransformer is reused with additional conditioning inputs.</summary>
public sealed record FluxToolsConfig
{
    /// <summary>Base Flux transformer config. All tools variants use the same backbone.</summary>
    public required FluxConfig TransformerConfig { get; init; }

    /// <summary>Specific tool variant.</summary>
    public required FluxToolVariant Variant { get; init; }

    /// <summary>Number of additional <b>packed</b> input channels concatenated onto the 64-wide packed noise token
    /// before the transformer's <c>x_embedder</c> (so the effective <c>in_channels</c> = 64 + this). Spatial-
    /// conditioning variants: Canny / Depth = <b>64</b> (a 16-ch VAE latent, 2×2-packed → 64; in_channels 128);
    /// Fill = <b>320</b> (packed masked-image latent 64 + packed mask 256; in_channels 384); Kontext = <b>64</b>
    /// (reference-image latent, packed). Embedding-conditioning variants (Redux) = 0.</summary>
    public int AdditionalInChannels { get; init; }

    /// <summary>Whether this variant requires a condition image input.</summary>
    public bool RequiresConditionImage => Variant is FluxToolVariant.Fill or FluxToolVariant.Canny
        or FluxToolVariant.Depth or FluxToolVariant.Kontext;

    /// <summary>Whether this variant uses image embedding conditioning (CLIP/SigLIP image features).</summary>
    public bool RequiresImageEmbedding => Variant is FluxToolVariant.Redux;

    /// <summary>Flux.1 Fill preset (inpainting/outpainting). Accepts masked latent as additional input channels.</summary>
    public static FluxToolsConfig Fill => new()
    {
        TransformerConfig = FluxConfig.Flux1Fill,
        Variant = FluxToolVariant.Fill,
        AdditionalInChannels = 320, // packed masked-latent (64) + packed mask (256) → in_channels 384
    };

    /// <summary>Flux.1 Canny preset (edge-guided generation). Accepts canny edge map.</summary>
    public static FluxToolsConfig Canny => new()
    {
        TransformerConfig = FluxConfig.Flux1Tools,
        Variant = FluxToolVariant.Canny,
        AdditionalInChannels = 64, // 16-ch VAE latent of the edge map, 2×2-packed → in_channels 128
    };

    /// <summary>Flux.1 Depth preset (depth-guided generation). Accepts depth map.</summary>
    public static FluxToolsConfig Depth => new()
    {
        TransformerConfig = FluxConfig.Flux1Tools,
        Variant = FluxToolVariant.Depth,
        AdditionalInChannels = 64, // 16-ch VAE latent of the depth map, 2×2-packed → in_channels 128
    };

    /// <summary>Flux.1 Redux preset (image variation). Uses CLIP/SigLIP image embeddings as conditioning.</summary>
    public static FluxToolsConfig Redux => new()
    {
        TransformerConfig = FluxConfig.Dev,
        Variant = FluxToolVariant.Redux,
        AdditionalInChannels = 0,
    };

    /// <summary>Flux.1 Kontext preset (image editing via text). Accepts reference image + edit prompt.</summary>
    public static FluxToolsConfig Kontext => new()
    {
        TransformerConfig = FluxConfig.Dev,
        Variant = FluxToolVariant.Kontext,
        AdditionalInChannels = 64, // 16-ch reference-image latent, 2×2-packed
    };
}

/// <summary>Flux.1 Tools variant type.</summary>
public enum FluxToolVariant
{
    /// <summary>Flux.1 Fill — inpainting and outpainting.</summary>
    Fill,

    /// <summary>Flux.1 Canny — edge-guided generation.</summary>
    Canny,

    /// <summary>Flux.1 Depth — depth-guided generation.</summary>
    Depth,

    /// <summary>Flux.1 Redux — image variations from CLIP/SigLIP embeddings.</summary>
    Redux,

    /// <summary>Flux.1 Kontext — image editing via text descriptions.</summary>
    Kontext,
}
