namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Configuration for Lance (ByteDance Research, Apache-2.0) — a single 3B-active unified multimodal model (T2I / T2V / edit / understanding). Reconciled against the REAL released checkpoint (<c>bytedance-research/Lance</c> <c>Lance_3B/model.safetensors</c>, 2026-07) + the upstream inference command (<c>inference_lance.sh</c>): QK-RMSNorm is ON, the latent patch is <c>(1,1,1)</c> (tokens are single 48-channel latent pixels via <c>vae2llm: Linear(48→2048)</c>), and <c>max_latent_size=64</c> (the frozen <c>latent_pos_embed</c> table is 64×64 per frame).
///
/// Lance's backbone is a **modified Qwen2.5-VL 3B decoder** with per-layer **MoT (Mixture-of-Tokens) dual-stream routing**: every token carries a modality role; understanding tokens flow through the standard <c>q/k/v/o/mlp/norm</c> set, generation tokens through a parallel <c>*_moe_gen</c> set. Both streams share ONE joint attention. Position ids come from Qwen2.5-VL <c>get_rope_index</c> (text 1-D, vision block 3-D); the MaPE per-role temporal shift only applies to editing tasks (<c>full_noise</c>/<c>full</c> splits), NOT to pure T2I/T2V.
///
/// This config covers the **LLM backbone** only. The frozen Wan2.2 3D causal VAE has its own config (<c>Wan22VaeConfig</c>). See <c>docs/Research/LANCE_ARCHITECTURE.md</c>.</summary>
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

    /// <summary>Per-head QK-RMSNorm on both streams (confirmed ON in the released checkpoint: <c>q_norm</c>/<c>k_norm</c> + <c>_moe_gen</c> siblings, shape [128]).</summary>
    public bool QkNorm { get; init; } = true;

    // ── Generation latent handoff (VAE → transformer) ──
    /// <summary>VAE latent channels (Wan2.2).</summary>
    public int VaeZChannels { get; init; } = 48;

    /// <summary>VAE spatial downsample factor.</summary>
    public int VaeDownsampleSpatial { get; init; } = 16;

    /// <summary>VAE temporal downsample factor.</summary>
    public int VaeDownsampleTemporal { get; init; } = 4;

    /// <summary>Latent patchify <c>(t, h, w)</c> for the transformer handoff — <c>(1,1,1)</c> in the released checkpoint (<c>--latent_patch_size 1 1 1</c>): one token per latent pixel.</summary>
    public (int T, int H, int W) LatentPatchSize { get; init; } = (1, 1, 1);

    /// <summary>Patched-latent token feature dim = <c>z × patchT × patchH × patchW</c> (= 48; matches <c>vae2llm.weight [2048,48]</c>).</summary>
    public int PatchFeatureDim => VaeZChannels * LatentPatchSize.T * LatentPatchSize.H * LatentPatchSize.W;

    /// <summary>Max latent grid H/W for the frozen position table (<c>--max_latent_size 64</c>; the image checkpoint ships a 64·64=4096-row <c>latent_pos_embed</c>).</summary>
    public int MaxLatentSize { get; init; } = 64;

    /// <summary>Qwen2.5-VL vision temporal token rate — the per-latent-frame step of the temporal M-RoPE axis inside a video block (<c>tokens_per_second × second_per_grid_t</c> with 1.0 s/grid).</summary>
    public int TokensPerSecond { get; init; } = 2;

    /// <summary>Timestep-embedder sinusoidal width.</summary>
    public int TimestepFrequencyDim { get; init; } = 256;

    // ── Sampling defaults (inference_lance.sh + config_factory.py) ──
    /// <summary>Logit-normal timestep shift (image inference).</summary>
    public float ImageTimestepShift { get; init; } = 3.5f;

    /// <summary>Logit-normal timestep shift (video inference).</summary>
    public float VideoTimestepShift { get; init; } = 4.0f;

    /// <summary>Default Euler step count.</summary>
    public int NumTimesteps { get; init; } = 30;

    /// <summary>Default text CFG scale.</summary>
    public float CfgTextScale { get; init; } = 4.0f;

    /// <summary>CFG applies only when the (shifted) flow time is in <c>(CfgIntervalMin, 1]</c> — upstream <c>cfg_interval=[0.4, 1.0]</c>; late low-noise steps run cond-only.</summary>
    public float CfgIntervalMin { get; init; } = 0.4f;

    /// <summary>Lower clamp for the global CFG renorm scale <c>‖v_cond‖/‖v_cfg‖</c> (upstream <c>cfg_renorm_min=0</c>, <c>cfg_renorm_type="global"</c>).</summary>
    public float CfgRenormMin { get; init; } = 0f;

    // ── Token sentinels (Qwen2.5-VL) ──
    public int BosTokenId { get; init; } = 151643;
    public int EosTokenId { get; init; } = 151645;
    public int ImStartTokenId { get; init; } = 151644;
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
