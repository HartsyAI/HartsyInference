namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Configuration for Hunyuan Image 2.1 MMDiT transformer (17B by Tencent). Features a 32×32 VAE downscale, native 2048×2048 resolution, and includes distilled + refiner variants. Uses dual text encoders and a unique double/single-stream MMDiT architecture.</summary>
public sealed record HunyuanImageConfig
{
    /// <summary>Hidden dimension of the transformer.</summary>
    public required int HiddenSize { get; init; }

    /// <summary>Number of attention heads.</summary>
    public required int NumHeads { get; init; }

    /// <summary>Per-head dimension.</summary>
    public int HeadDim { get; init; } = 128;

    /// <summary>Number of double-stream blocks (joint image+text attention).</summary>
    public required int NumDoubleBlocks { get; init; }

    /// <summary>Number of single-stream blocks (image-only after joint processing).</summary>
    public required int NumSingleBlocks { get; init; }

    /// <summary>Patch size for latent→token embedding.</summary>
    public int PatchSize { get; init; } = 2;

    /// <summary>Number of latent channels (64 for Hunyuan Image, the 32× VAE outputs 64-channel latents).</summary>
    public int InChannels { get; init; } = 64;

    /// <summary>Number of refiner layers in the MLLM context embedder (<c>HunyuanImageTokenRefiner</c>). Default 2.</summary>
    public int NumRefinerLayers { get; init; } = 2;

    /// <summary>Primary text context dimension — 3584 = Qwen2.5-VL-7B hidden size (diffusers <c>HunyuanImageTransformer2DModel.text_embed_dim</c> default).</summary>
    public int TextEmbedDim { get; init; } = 3584;

    /// <summary>Optional secondary text context dimension (ByT5 glyph encoder). Null when only the primary encoder is used.</summary>
    public int? TextEmbedDim2 { get; init; }

    /// <summary>RoPE base frequency for positional encoding.</summary>
    public float RopeTheta { get; init; } = 256.0f;

    /// <summary>Per-axis RoPE dim split (height, width). Must sum to <see cref="HeadDim"/>.</summary>
    public int[] RopeAxesDim { get; init; } = [64, 64];

    /// <summary>Whether to embed the guidance scale via an MLP into the modulation vector. Per the upstream
    /// <c>hunyuanimage_config.py</c> this is <b>false for the full model</b> (which runs real classifier-free
    /// guidance) and <b>true for the distilled model</b> (guidance-distilled, embedded-guidance MLP). Do not
    /// confuse with a "full embeds guidance" intuition — it is the distilled variant that carries the embed.</summary>
    public bool GuidanceEmbed { get; init; } = false;

    /// <summary>Whether the variant uses meanflow sampling (true for the distilled model and the refiner; false
    /// for the full model). The distilled/refiner checkpoints are trained with a meanflow objective and require
    /// the meanflow sampling schedule rather than plain flow-matching Euler.</summary>
    public bool UseMeanflow { get; init; }

    /// <summary>QK-norm epsilon.</summary>
    public float QkNormEps { get; init; } = 1e-6f;

    /// <summary>Whether to use QK-norm (RMSNorm on Q/K).</summary>
    public bool UseQkNorm { get; init; } = true;

    /// <summary>MLP ratio (typically 4.0).</summary>
    public float MlpRatio { get; init; } = 4.0f;

    /// <summary>Flow-match sampling shift. Reference: <b>5.0</b> for the full model, <b>4.0</b> for the distilled
    /// model (the upstream pipeline also uses a custom <c>get_timesteps_sigmas</c> schedule; this shift is the
    /// minimum-fidelity approximation until that is ported).</summary>
    public float SamplingShift { get; init; } = 5.0f;

    /// <summary>Hunyuan Image 2.1 full preset (17B params). Backbone is <c>hidden_size=3584, heads_num=28</c>
    /// per the upstream <c>hunyuanimage_config.py</c> (3584/28 → head_dim 128). The full model uses <b>real CFG</b>
    /// (<see cref="GuidanceEmbed"/>=false) and standard flow-matching (<see cref="UseMeanflow"/>=false).</summary>
    public static HunyuanImageConfig V21 => new()
    {
        HiddenSize = 3584,
        NumHeads = 28,
        HeadDim = 128,
        NumDoubleBlocks = 20,
        NumSingleBlocks = 40,
        NumRefinerLayers = 2,
        PatchSize = 1,
        InChannels = 64,
        TextEmbedDim = 3584,
        TextEmbedDim2 = 1472,
        GuidanceEmbed = false,
        UseMeanflow = false,
        SamplingShift = 5.0f,
        RopeTheta = 256.0f,
        RopeAxesDim = [64, 64],
    };

    /// <summary>Hunyuan Image 2.1 distilled preset (faster, fewer steps). Same 3584/28 backbone as the full model;
    /// the distilled variant is the one that carries the embedded-guidance MLP (<see cref="GuidanceEmbed"/>=true)
    /// and is trained with the meanflow objective (<see cref="UseMeanflow"/>=true).</summary>
    public static HunyuanImageConfig V21Distilled => new()
    {
        HiddenSize = 3584,
        NumHeads = 28,
        HeadDim = 128,
        NumDoubleBlocks = 20,
        NumSingleBlocks = 40,
        NumRefinerLayers = 2,
        PatchSize = 1,
        InChannels = 64,
        TextEmbedDim = 3584,
        TextEmbedDim2 = 1472,
        GuidanceEmbed = true,
        UseMeanflow = true,
        SamplingShift = 4.0f,
        RopeTheta = 256.0f,
        RopeAxesDim = [64, 64],
    };
}
