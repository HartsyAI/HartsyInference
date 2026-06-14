namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Configuration for Lance (ByteDance Research, Apache-2.0) — a single 3B-active unified multimodal model (T2I / T2V / edit / understanding). Verified against the research doc + upstream <c>Lance_3B/llm_config.json</c> and <c>modeling/lance/qwen2_navit.py</c>.
///
/// Lance's backbone is a **modified Qwen2.5-VL 3B decoder** with per-layer **MoT (Mixture-of-Tokens) dual-stream routing**: every token carries a modality role (text-und / vit-semantic / clean-vae / noisy-vae); understanding tokens flow through the standard <c>q/k/v/o/mlp/norm</c> set, generation tokens through a parallel <c>*_moe_gen</c> set. Both streams share ONE joint attention. **MaPE** then shifts only the temporal M-RoPE axis by a per-role offset so the roles don't collide in the shared position space.
///
/// This config covers the **LLM backbone** only. The frozen Qwen2.5-VL ViT and Wan2.2 3D causal VAE get their own configs (<c>Qwen25VlVitConfig</c>, <c>Wan22VaeConfig</c>) when those subsystems land. See <c>docs/Research/LANCE_ARCHITECTURE.md</c> and the Phase 4 / Phase 9 § Lance checklists for the staged build order.</summary>
public sealed record LanceConfig
{
    // ── LLM backbone (Qwen2.5-VL 3B, MoT-augmented) ──
    /// <summary>Backbone hidden dim.</summary>
    public int HiddenSize { get; init; } = 2048;

    /// <summary>Number of MoT decoder layers.</summary>
    public int NumLayers { get; init; } = 36;

    /// <summary>Query heads.</summary>
    public int NumHeads { get; init; } = 16;

    /// <summary>Per-head dim (<c>HiddenSize / NumHeads</c>).</summary>
    public int HeadDim => HiddenSize / NumHeads;

    /// <summary>Key/value heads (GQA factor 8 — the most extreme in the codebase).</summary>
    public int NumKvHeads { get; init; } = 2;

    /// <summary>SwiGLU FFN intermediate size (per stream).</summary>
    public int IntermediateSize { get; init; } = 11008;

    /// <summary>RMSNorm epsilon.</summary>
    public float RmsNormEps { get; init; } = 1e-6f;

    /// <summary>M-RoPE base frequency.</summary>
    public float RopeTheta { get; init; } = 1_000_000f;

    /// <summary>M-RoPE per-axis sections <c>(t, h, w)</c>; sums to head_dim/2 = 64.</summary>
    public (int Temporal, int Height, int Width) MropeSection { get; init; } = (16, 24, 24);

    /// <summary>Token vocabulary (Qwen2 BPE).</summary>
    public int VocabSize { get; init; } = 151_936;

    /// <summary>Whether the MoT attention applies per-head QK-RMSNorm (confirm against safetensors keys at load).</summary>
    public bool QkNorm { get; init; } = false;

    // ── Generation latent handoff (VAE → transformer) ──
    /// <summary>VAE latent channels (Wan2.2).</summary>
    public int VaeZChannels { get; init; } = 48;

    /// <summary>VAE spatial downsample factor.</summary>
    public int VaeDownsampleSpatial { get; init; } = 16;

    /// <summary>VAE temporal downsample factor.</summary>
    public int VaeDownsampleTemporal { get; init; } = 4;

    /// <summary>Latent patchify <c>(t, h, w)</c> for the transformer handoff.</summary>
    public (int T, int H, int W) LatentPatchSize { get; init; } = (1, 2, 2);

    /// <summary>Patched-latent feature dim = <c>z × patchT × patchH × patchW</c> = 48 × 1 × 2 × 2.</summary>
    public int PatchFeatureDim => VaeZChannels * LatentPatchSize.T * LatentPatchSize.H * LatentPatchSize.W;

    /// <summary>Max latent grid size (H/W in latent units) for the frozen 3D position embedding.</summary>
    public int MaxLatentSize { get; init; } = 32;

    /// <summary>ViT→LLM connector activation.</summary>
    public string ConnectorActivation { get; init; } = "gelu_pytorch_tanh";

    /// <summary>Timestep-embedder sinusoidal width.</summary>
    public int TimestepFrequencyDim { get; init; } = 256;

    // ── Sampling defaults ──
    /// <summary>Logit-normal timestep shift (image inference).</summary>
    public float ImageTimestepShift { get; init; } = 3.5f;

    /// <summary>Logit-normal timestep shift (video inference).</summary>
    public float VideoTimestepShift { get; init; } = 4.0f;

    /// <summary>Default Euler step count.</summary>
    public int NumTimesteps { get; init; } = 30;

    /// <summary>Default text branch scale for 3-way CFG.</summary>
    public float CfgTextScale { get; init; } = 4.0f;

    // ── Token sentinels (Qwen2.5-VL) ──
    public int BosTokenId { get; init; } = 151643;
    public int EosTokenId { get; init; } = 151645;
    public int VisionStartTokenId { get; init; } = 151652;
    public int VisionEndTokenId { get; init; } = 151653;
    public int VisionTokenId { get; init; } = 151654;
    public int ImageTokenId { get; init; } = 151655;
    public int VideoTokenId { get; init; } = 151656;

    /// <summary>Image specialist preset (<c>Lance_3B</c>).</summary>
    public static LanceConfig Image => new();

    /// <summary>Unified image+video preset (<c>Lance_3B_Video</c>) — same backbone, video sampling defaults.</summary>
    public static LanceConfig Video => new();
}
