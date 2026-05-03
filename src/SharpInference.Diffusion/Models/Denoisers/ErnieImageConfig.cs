namespace SharpInference.Diffusion.Models.Denoisers;

/// <summary>Configuration for ERNIE-Image (Baidu, ~8B params, Apache-2.0). Verified against
/// <c>diffusers/models/transformers/transformer_ernie_image.py</c> and the published
/// <c>baidu/ERNIE-Image</c> HuggingFace config.
///
/// **Architectural correction:** the Phase 1 roadmap previously claimed ERNIE-Image was
/// "Flux-lineage MMDiT" — that's wrong. ERNIE is **single-stream DiT** with a UNIQUE
/// **shared AdaLN modulation across all 36 layers**: one `nn.SiLU + Linear(hidden, 6*hidden)`
/// at the top level produces ONE 6-tuple of modulation params (`shift_msa, scale_msa, gate_msa,
/// shift_mlp, scale_mlp, gate_mlp`) that gets reused by every block. This saves ~300 MB of
/// per-block AdaLN linear weights vs Flux's per-block scheme.
///
/// Other notable deltas vs Flux/SD3:
/// <list type="bullet">
///   <item>**RMSNorm everywhere** (not LayerNorm). QK-norm via RMSNorm over head_dim.</item>
///   <item>**3D RoPE on `(text_pos, y, x)` axes (32, 48, 48), theta=256.** Image tokens use
///        `(text_lens_per_batch, y, x)` — image RoPE positions are **offset by the per-batch
///        text length**, which is non-standard. Text tokens use `(arange, 0, 0)`.</item>
///   <item>**Non-interleaved Megatron-style rotate-half**: `[θ0,θ0,θ1,θ1,…]` layout, NOT the
///        interleaved-pair layout that <see cref="DiTBlocks.FluxRope"/> uses. Cannot reuse FluxRope.</item>
///   <item>**GELU-gated FFN** (NOT SwiGLU): `linear_fc2(up_proj(x) * gelu(gate_proj(x)))`. All
///        FFN linears `bias=False`.</item>
///   <item>**Patch embed is 1×1 Conv2d** with bias. <c>patch_size=1</c> because patchification
///        happens in the VAE (Flux2-style 128-channel latent at lower spatial res).</item>
///   <item>`text_in = Linear(3072, 4096, bias=False)` projects text encoder output to model hidden.</item>
///   <item>Final layer: `LayerNorm(no affine) + Linear(hidden, 2*hidden) → scale, shift` then
///        `final_linear: Linear(hidden, p² * out_channels)` with out_channels=128. Reshape to
///        `[B, 128, H, W]`.</item>
///   <item>Padding-aware attention mask (image tokens always valid, text tokens valid up to
///        `text_lens`).</item>
///   <item>Internal `[S, B, H]` sequence-first tensor layout (transposed at boundaries).</item>
/// </list>
///
/// VAE: `AutoencoderKLFlux2` — Flux2-style 128-channel VAE. Reuse the existing Flux2 VAE infra.
///
/// Text encoder: **Custom Baidu encoder** (loaded via `from transformers import AutoModel`).
/// Likely an in-house ERNIE-class encoder — must inspect `text_encoder/config.json` on the
/// HF repo before implementing. Optional `pe` Prompt Enhancer LLM (`AutoModelForCausalLM`) — skip in v1.
///
/// **Implementation status:** scaffolding only. See PHASE_4_MODEL_BREADTH.md `### ERNIE-Image`.
/// </summary>
public record ErnieImageConfig
{
    /// <summary>Hidden dim (= num_attention_heads * head_dim). 4096 for `baidu/ERNIE-Image`.</summary>
    public required int HiddenSize { get; init; }

    /// <summary>Number of attention heads. 32 for v1.</summary>
    public required int NumHeads { get; init; }

    /// <summary>Per-head dimension. 128 for v1 (32 × 128 = 4096).</summary>
    public int HeadDim { get; init; } = 128;

    /// <summary>Number of identical single-stream blocks (all sharing one AdaLN). 36 for v1.</summary>
    public required int NumLayers { get; init; }

    /// <summary>FFN hidden dim. 12288 (3× hidden) for v1.</summary>
    public int FfnHiddenSize { get; init; } = 12288;

    /// <summary>Latent input channels (delivered by the Flux2-style VAE). 128 for v1.</summary>
    public int InChannels { get; init; } = 128;

    /// <summary>Output channels (back to VAE latent space). 128 for v1.</summary>
    public int OutChannels { get; init; } = 128;

    /// <summary>Patch size. 1 — patchification is done by the VAE, the patch_embed is a 1×1 Conv2d.</summary>
    public int PatchSize { get; init; } = 1;

    /// <summary>Text encoder output dim. 3072 for v1 — the custom Baidu encoder produces 3072-dim.</summary>
    public int TextInDim { get; init; } = 3072;

    /// <summary>RoPE base frequency. 256 (NOT 10000 like Flux/SD3.5).</summary>
    public int RopeTheta { get; init; } = 256;

    /// <summary>RoPE axes dims `(text_pos, y, x)`. Sums to head_dim = 128.</summary>
    public (int Text, int Y, int X) RopeAxesDim { get; init; } = (32, 48, 48);

    /// <summary>RMSNorm epsilon.</summary>
    public float RmsNormEps { get; init; } = 1e-6f;

    /// <summary>QK-norm epsilon.</summary>
    public float QkNormEps { get; init; } = 1e-6f;

    /// <summary>`baidu/ERNIE-Image` v1 preset (~8B params).</summary>
    public static ErnieImageConfig V1 => new()
    {
        HiddenSize = 4096,
        NumHeads = 32,
        HeadDim = 128,
        NumLayers = 36,
        FfnHiddenSize = 12288,
        InChannels = 128,
        OutChannels = 128,
        PatchSize = 1,
        TextInDim = 3072,
        RopeTheta = 256,
        RopeAxesDim = (32, 48, 48),
    };

    /// <summary>`baidu/ERNIE-Image-Turbo` preset — same backbone, distilled for 8-step inference.</summary>
    public static ErnieImageConfig V1Turbo => V1;
}
