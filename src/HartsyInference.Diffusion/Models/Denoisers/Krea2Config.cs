namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Configuration for the Krea 2 diffusion transformer (<c>Krea2Transformer2DModel</c> from
/// <c>huggingface/diffusers</c>). A single-stream MMDiT flow-matching text-to-image backbone: a text-fusion stage
/// collapses a stack of tapped Qwen3-VL-4B hidden states, the result is concatenated with patchified image latents
/// into one <c>[text, image]</c> sequence, and 28 blocks (sigmoid-output-gate GQA attention + SwiGLU, modulated by a
/// shared timestep vector plus per-block learned tables) predict the flow-matching velocity. See
/// <c>docs/Research/KREA2.md</c>.
///
/// <para>Defaults match the released <c>Krea-2-Base</c> <c>transformer/config.json</c>: hidden 6144 (head_dim 128 ×
/// 48 heads), 28 layers, GQA 12 kv-heads, SwiGLU inner 16384, 3-axis RoPE [32,48,48] θ=1000, text encoder hidden 2560
/// with a 12-layer tap fused by 2 layerwise + 2 refiner blocks (20 heads, inner 6912), Qwen-Image 16-channel VAE.</para></summary>
public sealed record Krea2Config
{
    /// <summary>Latent channels of the VAE (16, Qwen-Image).</summary>
    public int VaeChannels { get; init; } = 16;

    /// <summary>Patch size for 2×2 token packing.</summary>
    public int PatchSize { get; init; } = 2;

    /// <summary>Patchified input channels fed to <c>img_in</c> = <c>VaeChannels · PatchSize²</c> = 64.</summary>
    public int InChannels => VaeChannels * PatchSize * PatchSize;

    /// <summary>Number of transformer blocks (28).</summary>
    public int NumLayers { get; init; } = 28;

    /// <summary>Per-head dimension (128). <c>HiddenSize = AttentionHeadDim · NumAttentionHeads</c>.</summary>
    public int AttentionHeadDim { get; init; } = 128;

    /// <summary>Query heads (48).</summary>
    public int NumAttentionHeads { get; init; } = 48;

    /// <summary>Key/value heads for GQA (12; group 4).</summary>
    public int NumKvHeads { get; init; } = 12;

    /// <summary>Model hidden size = 128 · 48 = 6144.</summary>
    public int HiddenSize => AttentionHeadDim * NumAttentionHeads;

    /// <summary>SwiGLU feed-forward inner dim inside each block (16384).</summary>
    public int IntermediateSize { get; init; } = 16384;

    /// <summary>Sinusoidal timestep embedding width before its MLP (256).</summary>
    public int TimestepEmbedDim { get; init; } = 256;

    /// <summary>Hidden size of the consumed text-encoder hidden states (2560, Qwen3-VL-4B).</summary>
    public int TextHiddenDim { get; init; } = 2560;

    /// <summary>Number of tapped text-encoder hidden states stacked per token (12).</summary>
    public int NumTextLayers { get; init; } = 12;

    /// <summary>Query heads in the text-fusion blocks (20).</summary>
    public int TextNumHeads { get; init; } = 20;

    /// <summary>Key/value heads in the text-fusion blocks (20; no GQA).</summary>
    public int TextNumKvHeads { get; init; } = 20;

    /// <summary>SwiGLU inner dim inside the text-fusion blocks (6912).</summary>
    public int TextIntermediateSize { get; init; } = 6912;

    /// <summary>Text-fusion blocks across the tapped-layer axis (2).</summary>
    public int NumLayerwiseTextBlocks { get; init; } = 2;

    /// <summary>Text-fusion blocks across the token axis (2).</summary>
    public int NumRefinerTextBlocks { get; init; } = 2;

    /// <summary>Per-axis RoPE dim split (t, h, w). Sum must equal <see cref="AttentionHeadDim"/>. [32,48,48].</summary>
    public int[] AxesDimRope { get; init; } = [32, 48, 48];

    /// <summary>RoPE base frequency (1000).</summary>
    public int RopeTheta { get; init; } = 1000;

    /// <summary>RMSNorm epsilon (1e-5).</summary>
    public float NormEps { get; init; } = 1e-5f;

    /// <summary>HF <c>hidden_states</c> tap indices (0 = embeddings) for the 12 stacked text layers.</summary>
    public static readonly int[] TextEncoderSelectLayers = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35];

    /// <summary>Whether this is the distilled (Turbo/TDM) checkpoint. Only changes the scheduler shift (fixed
    /// <c>mu=1.15</c>) and sampler defaults — the transformer geometry is identical to Base.</summary>
    public bool IsDistilled { get; init; }

    /// <summary>Krea 2 Base (midtrain) preset — 28-step, CFG-4.5 sampling.</summary>
    public static Krea2Config Base => new();

    /// <summary>Krea 2 Turbo / TDM (distilled) preset — 8-step, guidance-free sampling. Same geometry as Base.</summary>
    public static Krea2Config Turbo => new() { IsDistilled = true };
}
