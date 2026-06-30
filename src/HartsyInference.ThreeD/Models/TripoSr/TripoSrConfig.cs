namespace HartsyInference.ThreeD.Models.TripoSr;

/// <summary>TripoSR (Stability AI / Tripo, MIT) configuration — a feed-forward single-image→3D LRM: a DINO
/// ViT-B/16 image tokenizer feeds a diffusers <c>Transformer1D</c> over learned triplane tokens, the result
/// is upsampled (ConvTranspose2d) into a triplane and decoded by a <c>NeRFMLP</c> into density+color, meshed
/// via marching cubes. Deterministic (no diffusion). Dims match the public <c>stabilityai/TripoSR</c>
/// <c>config.yaml</c>.</summary>
public sealed record TripoSrConfig
{
    // --- Triplane (Triplane1DTokenizer + TriplaneUpsampleNetwork) ---
    /// <summary>Channels per upsampled triplane plane (post-upsample, NeRF input is 3× this).</summary>
    public required int TriplaneChannels { get; init; }

    /// <summary>Pre-upsample plane size (learned-token grid; ConvTranspose2d k2/s2 doubles it).</summary>
    public required int PlaneSize { get; init; }

    /// <summary>Backbone channel width = Triplane1DTokenizer <c>num_channels</c>.</summary>
    public required int NumChannels { get; init; }

    // --- Transformer1D backbone ---
    /// <summary>Inner transformer width (= NumHeads × HeadDim).</summary>
    public required int Width { get; init; }

    /// <summary>Number of transformer blocks.</summary>
    public required int Depth { get; init; }

    /// <summary>Attention heads.</summary>
    public required int NumHeads { get; init; }

    /// <summary>Per-head dim (scale = 1/√HeadDim).</summary>
    public required int HeadDim { get; init; }

    /// <summary>Cross-attention key/value dim (DINO hidden = 768).</summary>
    public required int CrossAttentionDim { get; init; }

    /// <summary>GroupNorm groups in the Transformer1D input norm.</summary>
    public int NormNumGroups { get; init; } = 32;

    // --- NeRF decoder (NeRFMLP) ---
    /// <summary>NeRF MLP hidden width (n_neurons).</summary>
    public required int NerfHidden { get; init; }

    /// <summary>Number of 64→64 hidden Linears between the input and output Linears (n_hidden_layers − 1).</summary>
    public required int NerfMidLayers { get; init; }

    // --- Renderer / extraction (TriplaneNeRFRenderer) ---
    /// <summary>Scene half-extent; query grid spans [−Radius, Radius]³ (config <c>renderer.radius</c>).</summary>
    public float Radius { get; init; } = 0.87f;

    /// <summary>Additive bias before the density exp activation (<c>density_bias</c>).</summary>
    public float DensityBias { get; init; } = -1.0f;

    /// <summary>Iso level on the activated density for marching cubes (TSR <c>extract_mesh</c> default).</summary>
    public float DensityThreshold { get; init; } = 25f;

    /// <summary>Default marching-cubes grid resolution per axis.</summary>
    public int GridResolution { get; init; } = 256;

    /// <summary>Upsampled triplane resolution (PlaneSize × 2).</summary>
    public int TriplaneResolution => PlaneSize * 2;

    /// <summary>Number of triplane tokens the backbone processes (3 planes × PlaneSize²).</summary>
    public int TriplaneTokens => 3 * PlaneSize * PlaneSize;

    /// <summary>Default config for the public <c>stabilityai/TripoSR</c> checkpoint. Pairs with a DINO
    /// ViT-B/16 tokenizer (CrossAttentionDim = 768) at a 512px conditioning image.</summary>
    public static TripoSrConfig TripoSr => new()
    {
        TriplaneChannels = 40, PlaneSize = 32, NumChannels = 1024,
        Width = 1024, Depth = 16, NumHeads = 16, HeadDim = 64, CrossAttentionDim = 768, NormNumGroups = 32,
        NerfHidden = 64, NerfMidLayers = 8,
        Radius = 0.87f, DensityBias = -1.0f, DensityThreshold = 25f, GridResolution = 256,
    };
}
