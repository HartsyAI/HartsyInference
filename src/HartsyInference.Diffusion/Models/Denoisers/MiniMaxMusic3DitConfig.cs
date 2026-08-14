namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Geometry of MiniMax Music 3's flow-matching transformer (<c>transformer/config.json</c>).</summary>
public sealed record MiniMaxMusic3DitConfig
{
    /// <summary>Flow-VAE latent channels.</summary>
    public int InChannels { get; init; } = 128;

    /// <summary>Width of the frame-aligned conditioning from the condition encoder.</summary>
    public int ConditionDim { get; init; } = 2048;

    public int NumLayers { get; init; } = 36;

    public int NumAttentionHeads { get; init; } = 32;

    public int AttentionHeadDim { get; init; } = 64;

    /// <summary>Feed-forward width after the gate split — <c>ff_in</c> emits twice this.</summary>
    public int FfInnerDim { get; init; } = 8192;

    /// <summary>Only the leading dimensions of each head rotate.</summary>
    public int RotaryDim { get; init; } = 32;

    /// <summary>Width of the random-Fourier time features before the timestep MLP.</summary>
    public int FourierEmbeddingDim { get; init; } = 256;

    /// <summary>Rotary base. Fixed at the module's default — the checkpoint stores no override.</summary>
    public float RopeTheta { get; init; } = 10_000f;

    /// <summary>PyTorch <c>nn.LayerNorm</c>'s default.</summary>
    public float LayerNormEps { get; init; } = 1e-5f;

    /// <summary>Transformer width.</summary>
    public int InnerDim => NumAttentionHeads * AttentionHeadDim;

    /// <summary>Channels the transformer input concatenates: <c>[latent, zeros(latent), condition]</c>.</summary>
    public int ConcatChannels => (2 * InChannels) + ConditionDim;

    /// <summary>The released checkpoint.</summary>
    public static MiniMaxMusic3DitConfig Default => new();
}
