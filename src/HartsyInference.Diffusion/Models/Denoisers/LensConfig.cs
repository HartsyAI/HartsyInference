namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Configuration for Microsoft Lens MMDiT (Microsoft Research, MIT, 2026-05-25). 3.8B-parameter dual-stream MMDiT paired with a frozen GPT-OSS MoE text encoder (multi-layer feature concat at layers 5/11/17/23) and the Flux.2 semantic VAE. All three released variants (Lens / Lens-Turbo / Lens-Base) share the exact same transformer architecture — they differ only in checkpoint and recommended sampler defaults. Architecture details in <see href="docs/Research/MICROSOFT_LENS_ARCHITECTURE.md"/>.</summary>
public sealed record LensConfig
{
    /// <summary>Hidden dimension (1536 = num_attention_heads × attention_head_dim).</summary>
    public required int HiddenSize { get; init; }

    /// <summary>Number of joint transformer blocks (48 for the released weights).</summary>
    public required int NumLayers { get; init; }

    /// <summary>Number of attention heads (24 for the released weights).</summary>
    public required int NumHeads { get; init; }

    /// <summary>Per-head dimension (64 for the released weights; equals <c>HiddenSize / NumHeads</c>).</summary>
    public int HeadDim { get; init; } = 64;

    /// <summary>SwiGLU MLP inner dimension. Upstream computes this as <c>int(HiddenSize / 3 * 8)</c> (the standard 8/3 SwiGLU ratio) → 4096 for HiddenSize=1536.</summary>
    public int MlpDim { get; init; } = 4096;

    /// <summary>Spatial patch size for the latent-to-token embed. Always 2 (2×2 spatial pack done at the pipeline boundary; the transformer's <c>img_in</c> is a plain Linear).</summary>
    public int PatchSize { get; init; } = 2;

    /// <summary>Latent channels seen by the transformer. 128 = (Flux.2 VAE 32 channels) × (2×2 patchify on the pipeline side).</summary>
    public int InChannels { get; init; } = 128;

    /// <summary>Predicted velocity channels (per packed token, before unpatchify). 32 = Flux.2 VAE latent channels; the <c>proj_out</c> Linear outputs <c>PatchSize² * OutChannels = 128</c>, which then unpatchifies to <c>[B, 32, H/16, W/16]</c>.</summary>
    public int OutChannels { get; init; } = 32;

    /// <summary>Encoder hidden dim per captured layer (GPT-OSS hidden_size = 2880).</summary>
    public int EncoderHiddenDim { get; init; } = 2880;

    /// <summary>Indices into GPT-OSS's 24 decoder layers whose hidden states get captured and channel-concatenated as the conditioning signal. <b>Do not expose as a runtime knob</b> — these are baked into how the DiT was trained.</summary>
    public int[] SelectedEncoderLayers { get; init; } = [5, 11, 17, 23];

    /// <summary>Per-axis split for the 3-axis complex-polar RoPE (frame, height, width). Sum equals <see cref="HeadDim"/>.</summary>
    public int[] AxesDimsRope { get; init; } = [8, 28, 28];

    /// <summary>RoPE base frequency.</summary>
    public int RopeTheta { get; init; } = 10000;

    /// <summary>QK-norm and stream-norm RMSNorm epsilons. Upstream uses 1e-5 for attention head norms and 1e-6 for stream norms; the slightly larger eps on per-head RMS guards against zero variance when MoE routing makes some heads narrow.</summary>
    public float QkNormEps { get; init; } = 1e-5f;

    /// <summary>RMSNorm epsilon for stream norms (img_norm1/2, txt_norm1/2, txt_norm[i]) and the AdaLN-Continuous final layer.</summary>
    public float StreamNormEps { get; init; } = 1e-6f;

    /// <summary>Default number of inference steps for this variant.</summary>
    public required int DefaultSteps { get; init; }

    /// <summary>Default CFG scale. Lens-Turbo uses 1.0 (no CFG, distilled); Lens and Lens-Base both use 5.0.</summary>
    public required float DefaultCfgScale { get; init; }

    /// <summary><c>microsoft/Lens</c> (default RL-tuned production weights). 20 steps, CFG 5.0.</summary>
    public static LensConfig Default => new()
    {
        HiddenSize = 1536,
        NumLayers = 48,
        NumHeads = 24,
        DefaultSteps = 20,
        DefaultCfgScale = 5.0f,
    };

    /// <summary><c>microsoft/Lens-Turbo</c> (4-step distilled, no CFG). Architecturally identical to <see cref="Default"/>.</summary>
    public static LensConfig Turbo => new()
    {
        HiddenSize = 1536,
        NumLayers = 48,
        NumHeads = 24,
        DefaultSteps = 4,
        DefaultCfgScale = 1.0f,
    };

    /// <summary><c>microsoft/Lens-Base</c> (SFT-only, no RL or distillation; for fair baseline comparison). 50 steps, CFG 5.0.</summary>
    public static LensConfig Base => new()
    {
        HiddenSize = 1536,
        NumLayers = 48,
        NumHeads = 24,
        DefaultSteps = 50,
        DefaultCfgScale = 5.0f,
    };
}
