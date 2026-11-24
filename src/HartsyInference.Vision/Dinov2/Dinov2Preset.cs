using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Dinov2;

/// <summary>Configuration for a DINOv2 ViT backbone (Meta, Apache-2.0) — the standard image-conditioning
/// encoder for image→3D models (Hunyuan3D, TRELLIS). Differs from CLIP/SigLIP towers by:
/// <list type="bullet">
///   <item>a prepended <b>CLS token</b> (and, for the <c>-reg</c> variants, <see cref="NumRegisterTokens"/>
///         learned register tokens) — so the token sequence is <c>1 + regs + numPatches</c>;</item>
///   <item><b>LayerScale</b>: a learned per-channel γ scales each block's attention/MLP output before the
///         residual add;</item>
///   <item><b>ImageNet</b> normalization (mean/std) in preprocessing;</item>
///   <item>feature output is the backbone's last hidden state (patch tokens), not a projected embedding.</item>
/// </list>
/// Exact position-embedding interpolation and the giant variant's SwiGLU FFN are validation-gated.</summary>
public sealed record Dinov2Preset
{
    /// <summary>Hub-style identifier (e.g. <c>facebook/dinov2-large</c>).</summary>
    public required string Name { get; init; }

    /// <summary>ViT hidden size (384 small, 768 base, 1024 large, 1536 giant).</summary>
    public required int HiddenSize { get; init; }

    /// <summary>Number of transformer layers (12 small/base, 24 large, 40 giant).</summary>
    public required int NumLayers { get; init; }

    /// <summary>Number of attention heads (head_dim = 64 for small/base/large).</summary>
    public required int NumHeads { get; init; }

    /// <summary>FFN intermediate size (4× hidden for the GELU-MLP variants).</summary>
    public required int IntermediateSize { get; init; }

    /// <summary>Input image side in pixels.</summary>
    public required int ImageSize { get; init; }

    /// <summary>Patch size (14 for all standard DINOv2 variants).</summary>
    public required int PatchSize { get; init; }

    /// <summary>Number of learned register tokens (0 for plain DINOv2; 4 for the <c>-reg</c> variants and all DINOv3 ViTs).</summary>
    public int NumRegisterTokens { get; init; }

    /// <summary>True for DINOv3 ViT variants, which replace DINOv2's absolute position embeddings with
    /// <b>rotary position embeddings (RoPE)</b> applied inside attention. Only <c>Dinov3VisionEncoder</c> accepts these
    /// presets; <see cref="Dinov2VisionEncoder"/> rejects them.</summary>
    public bool UsesRotaryPositionEmbedding { get; init; }

    /// <summary>DINOv3 only: round the RoPE periods to bfloat16, as the HairDeck and TRELLIS.2 references do. False keeps them in
    /// float32, as Hugging Face's <c>DINOv3ViTModel</c> does (Idol's reference).</summary>
    public bool RoundRopePeriodsToBf16 { get; init; } = true;

    /// <summary>DINOv3 only: the MLP is SwiGLU (<c>silu(gate_proj(x)) · up_proj(x)</c>, the ViT-H+) and not the plain GELU one. <c>Dinov3Layer</c> requires <c>mlp.gate_proj</c> in every layer when this is set and refuses a checkpoint that has it when it is not, so a preset and a file that disagree fail at load.</summary>
    public bool GatedMlp { get; init; }

    /// <summary>DINOv3 only: the query and value projections carry a bias (the HF layout and every <c>facebook/dinov3</c> checkpoint). False for a timm
    /// <c>qkv_bias=False</c> backbone such as Depth Pro 2's, whose attention has no q/k/v bias at all; <c>Dinov3Layer</c> then requires none and a file
    /// that carries one fails at load. The key projection never has a bias.</summary>
    public bool AttentionQkvBias { get; init; } = true;

    /// <summary>DINOv3 only: the encoder ends in a LayerNorm (<c>norm.weight</c>, <c>norm.bias</c>). False for a backbone whose taps are raw block outputs
    /// (Depth Pro 2): the loader then reads no final norm, and <c>Dinov3VisionEncoder.Encode</c> returns the last block's output unchanged.</summary>
    public bool HasFinalNorm { get; init; } = true;

    /// <summary>DINOv3 only: normalise both RoPE axes by the longer side of the patch grid (<c>2·(i+0.5)/max(h,w) − 1</c>, timm's <c>max</c> coordinate mode, which
    /// keeps the aspect ratio), not each axis by its own length. Identical for a square grid.</summary>
    public bool RopeNormalizesByLongSide { get; init; }

    /// <summary>LayerNorm epsilon (1e-6 for DINOv2; 1e-5 for the DINOv3 ViT-S/B/L checkpoints).</summary>
    public float LayerNormEps { get; init; } = 1e-6f;

    /// <summary>Patches along one side.</summary>
    public int PatchGrid => ImageSize / PatchSize;

    /// <summary>Number of patch tokens (excludes CLS / registers).</summary>
    public int NumPatches => PatchGrid * PatchGrid;

    /// <summary>Total token sequence length: CLS + registers + patches.</summary>
    public int SequenceLength => 1 + NumRegisterTokens + NumPatches;

    /// <summary><c>facebook/dinov2-large</c> — 300M params, the conditioning backbone Hunyuan3D-2 uses.</summary>
    public static Dinov2Preset Large => new()
    {
        Name = "facebook/dinov2-large",
        HiddenSize = 1024, NumLayers = 24, NumHeads = 16, IntermediateSize = 4096,
        ImageSize = 518, PatchSize = 14,
    };

    /// <summary><c>dinov2_vitl14_reg</c> — ViT-L/14 with 4 register tokens (the TRELLIS image conditioner).
    /// Same backbone as <see cref="Large"/> plus 4 learned registers → 1 + 4 + 37² = 1374 tokens at 518px.
    /// Native 518px (pos_embed already 37×37, no interpolation); GELU MLP (fc1/fc2). Weights = the torch.hub
    /// checkpoint remapped to HF keys (see <c>convert_dinov2_reg.py</c>).</summary>
    public static Dinov2Preset LargeReg => new()
    {
        Name = "dinov2_vitl14_reg",
        HiddenSize = 1024, NumLayers = 24, NumHeads = 16, IntermediateSize = 4096,
        ImageSize = 518, PatchSize = 14, NumRegisterTokens = 4,
    };

    /// <summary><c>facebook/dinov2-small</c> — 22M params. The Depth-Anything-V2-Small backbone.</summary>
    public static Dinov2Preset Small => new()
    {
        Name = "facebook/dinov2-small",
        HiddenSize = 384, NumLayers = 12, NumHeads = 6, IntermediateSize = 1536,
        ImageSize = 518, PatchSize = 14,
    };

    /// <summary><c>facebook/dinov2-base</c> — 86M params.</summary>
    public static Dinov2Preset Base => new()
    {
        Name = "facebook/dinov2-base",
        HiddenSize = 768, NumLayers = 12, NumHeads = 12, IntermediateSize = 3072,
        ImageSize = 518, PatchSize = 14,
    };

    /// <summary><c>facebook/dinov2-giant</c> — 1.1B params, SwiGLU FFN. The Hunyuan3D-2 shape conditioner
    /// (bundled in the DiT checkpoint). Native 518px (37×37 patches + CLS = 1370 tokens), no register tokens,
    /// so position embeddings are used directly (no interpolation). FFN is SwiGLU (<c>mlp.weights_in/out</c>),
    /// auto-detected at load; <see cref="IntermediateSize"/> is unused for the SwiGLU path.</summary>
    public static Dinov2Preset Giant => new()
    {
        Name = "facebook/dinov2-giant",
        HiddenSize = 1536, NumLayers = 40, NumHeads = 24, IntermediateSize = 4096,
        ImageSize = 518, PatchSize = 14,
    };

    // ── DINOv3 ViT variants ──────────────────────────────────────────────
    // DINOv3 (Meta, 2025) keeps DINOv2's CLS + 4 register tokens, LayerScale, and (for L and up) a gated
    // SwiGLU FFN — all already handled here — but swaps absolute position embeddings for RoPE and uses a
    // 16px patch. Encoded by Dinov3VisionEncoder; these presets are not auto-selected from weights, see FromDinov3Weights.
    // Every released DINOv3 ViT-S/B/L uses LayerNorm eps 1e-5, plain GELU MLP (up_proj/down_proj), 4 storage tokens,
    // LayerScale and a 64-wide head.

    /// <summary><c>facebook/dinov3-vits16</c> — DINOv3 ViT-S/16.</summary>
    public static Dinov2Preset V3Small16 => new()
    {
        Name = "facebook/dinov3-vits16",
        HiddenSize = 384, NumLayers = 12, NumHeads = 6, IntermediateSize = 1536,
        ImageSize = 224, PatchSize = 16, NumRegisterTokens = 4, UsesRotaryPositionEmbedding = true, LayerNormEps = 1e-5f,
    };

    /// <summary><c>facebook/dinov3-vitb16</c> — DINOv3 ViT-B/16.</summary>
    public static Dinov2Preset V3Base16 => new()
    {
        Name = "facebook/dinov3-vitb16",
        HiddenSize = 768, NumLayers = 12, NumHeads = 12, IntermediateSize = 3072,
        ImageSize = 224, PatchSize = 16, NumRegisterTokens = 4, UsesRotaryPositionEmbedding = true, LayerNormEps = 1e-5f,
    };

    /// <summary><c>facebook/dinov3-vitl16-pretrain-lvd1689m</c> — DINOv3 ViT-L/16, plain GELU MLP. The default 224 px
    /// size is the released one; HairDeck conditions on 448 px, so use <c>V3Large16 with { ImageSize = 448 }</c>
    /// (28×28 patches, 789 tokens with CLS and 4 storage tokens).</summary>
    public static Dinov2Preset V3Large16 => new()
    {
        Name = "facebook/dinov3-vitl16",
        HiddenSize = 1024, NumLayers = 24, NumHeads = 16, IntermediateSize = 4096,
        ImageSize = 224, PatchSize = 16, NumRegisterTokens = 4, UsesRotaryPositionEmbedding = true, LayerNormEps = 1e-5f,
    };

    /// <summary><c>facebook/dinov3-vith16plus-pretrain-lvd1689m</c> — DINOv3 ViT-H+/16: SwiGLU MLP (<c>mlp.gate_proj</c>,
    /// <c>up_proj</c>, <c>down_proj</c>, 5120 wide), 32 layers, 20 heads of 64. Idol conditions on 1024 px (64×64 patches, 4101
    /// tokens with CLS and 4 storage tokens). <see cref="LayerNormEps"/> is the 1e-5 of the other DINOv3 ViTs and is the HF
    /// <c>DINOv3ViTConfig</c> default, which the Idol conversion script leaves unchanged.</summary>
    public static Dinov2Preset V3HugePlus16 => new()
    {
        Name = "facebook/dinov3-vith16plus-pretrain-lvd1689m",
        HiddenSize = 1280, NumLayers = 32, NumHeads = 20, IntermediateSize = 5120,
        ImageSize = 224, PatchSize = 16, NumRegisterTokens = 4, UsesRotaryPositionEmbedding = true, LayerNormEps = 1e-5f,
        RoundRopePeriodsToBf16 = false, GatedMlp = true,
    };

    /// <summary>The DINOv2 variant an HF-layout checkpoint was trained as, read from its width
    /// (<c>embeddings.cls_token</c>), layer count, register tokens and patch size. Throws for DINOv3 (16 px patches, RoPE
    /// the encoder lacks) and for any shape no released DINOv2 variant has.</summary>
    public static Dinov2Preset FromWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix = "")
    {
        string p = prefix.Length == 0 ? "" : prefix + ".";
        if (!weights.TryGetValue($"{p}embeddings.cls_token", out Tensor? cls)
            || !weights.TryGetValue($"{p}embeddings.patch_embeddings.projection.weight", out Tensor? patch))
        {
            throw new NotSupportedException(
                $"Not an HF-layout DINOv2 checkpoint: '{p}embeddings.cls_token' or its patch projection is missing.");
        }
        int hidden = (int)cls.Shape[cls.Shape.Rank - 1];
        int patchSize = (int)patch.Shape[patch.Shape.Rank - 1];
        int registers = weights.TryGetValue($"{p}embeddings.register_tokens", out Tensor? reg) ? (int)reg.Shape[1] : 0;
        int layers = 0;
        while (weights.ContainsKey($"{p}encoder.layer.{layers}.norm1.weight"))
            layers++;

        if (patchSize != 14)
        {
            throw new NotSupportedException(
                $"DINOv2 checkpoint with {patchSize} px patches: a DINOv3 checkpoint must be read with FromDinov3Weights.");
        }
        Dinov2Preset? preset = (hidden, layers, registers) switch
        {
            (384, 12, 0) => Small,
            (768, 12, 0) => Base,
            (1024, 24, 0) => Large,
            (1024, 24, 4) => LargeReg,
            (1536, 40, 0) => Giant,
            _ => null,
        };
        return preset ?? throw new NotSupportedException(
            $"DINOv2 checkpoint with width {hidden}, {layers} layers and {registers} register tokens matches no supported "
            + "variant (Small 384×12, Base 768×12, Large 1024×24 with 0 or 4 registers, Giant 1536×40).");
    }

    /// <summary>The DINOv3 ViT-S/B/L or ViT-H+ preset an HF-layout DINOv3 checkpoint was trained as, at <paramref name="imageSize"/>.
    /// This is the explicit DINOv3 selector: <see cref="FromWeights"/> never matches a DINOv3 file, so a caller opts in by
    /// calling this and <c>Dinov3VisionEncoder</c>. Width and depth pick the variant; the caller states the input size.
    /// A gated (SwiGLU) MLP is accepted for the ViT-H+ only; the S/B/L presets are the plain GELU MLP.</summary>
    public static Dinov2Preset FromDinov3Weights(IReadOnlyDictionary<string, Tensor> weights, int imageSize, string prefix = "")
    {
        ArgumentNullException.ThrowIfNull(weights);
        if (imageSize <= 0 || imageSize % 16 != 0)
            throw new ArgumentOutOfRangeException(nameof(imageSize), imageSize, "DINOv3 input size must be a positive multiple of 16.");
        string p = prefix.Length == 0 ? "" : prefix + ".";
        if (!weights.TryGetValue($"{p}embeddings.cls_token", out Tensor? cls)
            || !weights.TryGetValue($"{p}embeddings.patch_embeddings.weight", out Tensor? patch)
            || !weights.TryGetValue($"{p}embeddings.register_tokens", out Tensor? reg)
            || !weights.ContainsKey($"{p}layer.0.mlp.up_proj.weight"))
        {
            throw new NotSupportedException(
                $"Not an HF-layout DINOv3 checkpoint: '{p}embeddings.cls_token', '{p}embeddings.patch_embeddings.weight', "
                + $"'{p}embeddings.register_tokens' or '{p}layer.0.mlp.up_proj.weight' is missing.");
        }
        bool gated = weights.ContainsKey($"{p}layer.0.mlp.gate_proj.weight");

        int hidden = (int)cls.Shape[cls.Shape.Rank - 1];
        int patchSize = (int)patch.Shape[patch.Shape.Rank - 1];
        int registers = (int)reg.Shape[1];
        int layers = 0;
        while (weights.ContainsKey($"{p}layer.{layers}.norm1.weight"))
            layers++;
        if (patchSize != 16 || registers != 4)
            throw new NotSupportedException($"DINOv3 checkpoint with {patchSize} px patches and {registers} storage tokens; expected 16 px and 4.");

        Dinov2Preset? preset = (hidden, layers, gated) switch
        {
            (384, 12, false) => V3Small16,
            (768, 12, false) => V3Base16,
            (1024, 24, false) => V3Large16,
            (1280, 32, true) => V3HugePlus16,
            _ => null,
        };
        return (preset ?? throw new NotSupportedException(
            $"DINOv3 checkpoint with width {hidden}, {layers} layers and a {(gated ? "gated (SwiGLU)" : "plain")} MLP matches no supported variant "
            + "(plain: ViT-S 384×12, ViT-B 768×12, ViT-L 1024×24; gated: ViT-H+ 1280×32)."))
            with { ImageSize = imageSize };
    }
}
