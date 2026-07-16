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

    /// <summary>Number of learned register tokens (0 for plain DINOv2; 4 for the <c>-reg</c> variants).</summary>
    public int NumRegisterTokens { get; init; }

    /// <summary>LayerNorm epsilon.</summary>
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
}
