namespace HartsyInference.Audio.Models.Auk;

/// <summary>Configuration for Tencent AuK / AuK-Flash — instruction-conditioned flow-matching DiT with a Qwen2.5-Omni thinker encoder and a 24 kHz BigVGAN-flow VAE.</summary>
public sealed record AukConfig
{
    /// <summary>DiT hidden dim.</summary>
    public int Dim { get; init; } = 1_536;

    /// <summary>Attention heads; head_dim = Dim / Heads = 64.</summary>
    public int Heads { get; init; } = 24;

    public int HeadDim => Dim / Heads;

    /// <summary>Double-stream (MMDiT) block count.</summary>
    public int DoubleBlocks { get; init; } = 10;

    /// <summary>Single-stream block count.</summary>
    public int SingleBlocks { get; init; } = 20;

    /// <summary>SwiGLU feed-forward multiplier; the inner dim is Dim * FfMult.</summary>
    public int FfMult { get; init; } = 2;

    public int FfInner => Dim * FfMult;

    /// <summary>VAE latent channels consumed and produced by the DiT.</summary>
    public int LatentDim { get; init; } = 64;

    /// <summary>Text-conditioning width delivered by the Qwen2.5-Omni thinker (pre txt_proj).</summary>
    public int TextDim { get; init; } = 2_048;

    /// <summary>Number of thinker hidden states fused into the text conditioning (layer_weights length).</summary>
    public int FusionLayers { get; init; } = 36;

    /// <summary>Sinusoidal timestep embedding width before the time MLP.</summary>
    public int TimeFreqEmbedDim { get; init; } = 256;

    /// <summary>ConvPositionEmbedding kernel size.</summary>
    public int ConvPosKernel { get; init; } = 31;

    /// <summary>ConvPositionEmbedding groups.</summary>
    public int ConvPosGroups { get; init; } = 16;

    /// <summary>Output sample rate of the VAE.</summary>
    public int SampleRate { get; init; } = 24_000;

    /// <summary>Audio samples per latent frame (50 Hz latents at 24 kHz).</summary>
    public int Hop { get; init; } = 480;

    /// <summary>Layer-norm epsilon for the adaLN blocks.</summary>
    public float NormEps { get; init; } = 1e-6f;

    /// <summary>Default AuK-Flash configuration is identical to base; only the sampler contract differs.</summary>
    public static AukConfig Default { get; } = new();
}
