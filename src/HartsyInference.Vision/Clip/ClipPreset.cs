using HartsyInference.Diffusion.Models.TextEncoders;

namespace HartsyInference.Vision.Clip;

/// <summary>Bundles a CLIP text encoder config + vision encoder config + preprocessing parameters into a single named preset. Used by <see cref="ClipModelLoader"/> to construct standalone CLIP pipelines for embedding and zero-shot scoring use cases.
/// <para>The configs themselves live in <see cref="ClipTextEncoderConfig"/> / <see cref="ClipVisionEncoderConfig"/> in the Diffusion package because diffusion conditioning was the first consumer. Reusing them here keeps the math in one place; this record only adds Vision-specific metadata (preset name, expected projection dim, image preprocessing size).</para></summary>
public sealed record ClipPreset
{
    /// <summary>Human-readable preset name (e.g. <c>"openai/clip-vit-large-patch14"</c>).</summary>
    public required string Name { get; init; }

    /// <summary>Text-tower config. Must have <see cref="ClipTextEncoderConfig.ProjectionDim"/> &gt; 0 for cosine-similarity scoring — the standalone CLIP scoring head requires the text_projection weight.</summary>
    public required ClipTextEncoderConfig TextConfig { get; init; }

    /// <summary>Vision-tower config. Must have <see cref="ClipVisionEncoderConfig.ProjectionDim"/> &gt; 0 for cosine-similarity scoring — the standalone CLIP scoring head requires the visual_projection weight.</summary>
    public required ClipVisionEncoderConfig VisionConfig { get; init; }

    /// <summary>Shared embedding dimension. Both text_projection and visual_projection output into this dim, so cosine similarity is well-defined.</summary>
    public int EmbeddingDim => VisionConfig.ProjectionDim;

    /// <summary>OpenAI CLIP ViT-L/14 — the de-facto standalone preset. 12 text layers (hidden=768, projection=768), 24 vision layers (hidden=1024, projection=768), 224×224 input, Quick-GELU. Matches <c>openai/clip-vit-large-patch14</c> on HuggingFace.</summary>
    public static ClipPreset OpenAiClipLarge => new()
    {
        Name = "openai/clip-vit-large-patch14",
        TextConfig = ClipTextEncoderConfig.Sd15 with { ProjectionDim = 768 },
        VisionConfig = ClipVisionEncoderConfig.ViTL14 with { ProjectionDim = 768 },
    };

    /// <summary>OpenCLIP CLIP ViT-H/14 (LAION-2B) — heavier than CLIP-L, much better retrieval. 24 text layers (hidden=1024, projection=1024), 32 vision layers (hidden=1280, projection=1024), 224×224 input, standard GELU. Matches <c>laion/CLIP-ViT-H-14-laion2B-s32B-b79K</c>.</summary>
    public static ClipPreset OpenClipHuge => new()
    {
        Name = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        TextConfig = new ClipTextEncoderConfig
        {
            HiddenSize = 1024,
            IntermediateSize = 4096,
            NumLayers = 24,
            NumHeads = 16,
            MaxPositionEmbeddings = 77,
            VocabSize = 49408,
            UseQuickGelu = false,
            ProjectionDim = 1024,
        },
        VisionConfig = ClipVisionEncoderConfig.ViTH14,
    };

    /// <summary>OpenCLIP CLIP ViT-bigG/14 (LAION-2B) — the second encoder of SDXL; also the strongest standalone scorer in this set. 32 text layers (hidden=1280, projection=1280), 48 vision layers (hidden=1664, head_dim=104, projection=1280), 224×224 input, standard GELU. Matches <c>laion/CLIP-ViT-bigG-14-laion2B-39B-b160k</c>.</summary>
    public static ClipPreset OpenClipBigG => new()
    {
        Name = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
        TextConfig = ClipTextEncoderConfig.SdxlClipG,
        VisionConfig = new ClipVisionEncoderConfig
        {
            HiddenSize = 1664,
            NumLayers = 48,
            NumHeads = 16,
            IntermediateSize = 8192,
            ImageSize = 224,
            PatchSize = 14,
            ProjectionDim = 1280,
            UseQuickGelu = false,
        },
    };

    // ── MetaCLIP (Meta, CC-licensed) ─────────────────────────────────────
    // MetaCLIP v1 (facebook/metaclip-*-fullcc2.5b) re-trains the OpenAI CLIP architecture on MetaCLIP
    // data. The HF checkpoints ship in the standard CLIPModel layout (text_model.* / vision_model.* /
    // {text,visual}_projection) with quick-GELU, so ClipModelLoader loads them with no remap — genuine
    // drop-in replacements for the OpenAI presets above.

    /// <summary><c>facebook/metaclip-b16-fullcc2.5b</c> — MetaCLIP ViT-B/16. Text hidden=512 (8 heads),
    /// vision hidden=768 patch16, shared projection=512, quick-GELU.</summary>
    public static ClipPreset MetaClipB16 => new()
    {
        Name = "facebook/metaclip-b16-fullcc2.5b",
        TextConfig = new ClipTextEncoderConfig
        {
            HiddenSize = 512,
            IntermediateSize = 2048,
            NumLayers = 12,
            NumHeads = 8,
            MaxPositionEmbeddings = 77,
            VocabSize = 49408,
            UseQuickGelu = true,
            ProjectionDim = 512,
        },
        VisionConfig = new ClipVisionEncoderConfig
        {
            HiddenSize = 768,
            NumLayers = 12,
            NumHeads = 12,
            IntermediateSize = 3072,
            ImageSize = 224,
            PatchSize = 16,
            ProjectionDim = 512,
            UseQuickGelu = true,
        },
    };

    /// <summary><c>facebook/metaclip-l14-fullcc2.5b</c> — MetaCLIP ViT-L/14. Same shapes as OpenAI CLIP-L,
    /// shared projection=768, quick-GELU.</summary>
    public static ClipPreset MetaClipL14 => new()
    {
        Name = "facebook/metaclip-l14-fullcc2.5b",
        TextConfig = ClipTextEncoderConfig.Sd15 with { ProjectionDim = 768 },
        VisionConfig = ClipVisionEncoderConfig.ViTL14 with { ProjectionDim = 768 },
    };

    /// <summary><c>facebook/metaclip-h14-fullcc2.5b</c> — MetaCLIP ViT-H/14. Text hidden=1024 (24 layers),
    /// vision hidden=1280 patch14, shared projection=1024, quick-GELU (unlike LAION's OpenCLIP-H which is standard GELU).</summary>
    public static ClipPreset MetaClipH14 => new()
    {
        Name = "facebook/metaclip-h14-fullcc2.5b",
        TextConfig = new ClipTextEncoderConfig
        {
            HiddenSize = 1024,
            IntermediateSize = 4096,
            NumLayers = 24,
            NumHeads = 16,
            MaxPositionEmbeddings = 77,
            VocabSize = 49408,
            UseQuickGelu = true,
            ProjectionDim = 1024,
        },
        VisionConfig = ClipVisionEncoderConfig.ViTH14 with { UseQuickGelu = true },
    };

    // ── EVA-CLIP (BAAI) ──────────────────────────────────────────────────
    // EVA-CLIP's TEXT tower is a standard CLIP text transformer (loads through ClipTextEncoder), but its
    // VISION tower is EVA-02: 2D RoPE, SwiGLU FFN, and extra sub-LayerNorms — NOT a vanilla CLIP ViT.
    // ClipVisionEncoder implements none of those, and the timm/OpenCLIP checkpoint keys (visual.blocks.*,
    // rope.*) differ from the CLIPModel layout ClipModelLoader expects. So this preset carries the correct
    // dimensions for the text side and documents the vision side as parity-blocked on EVA vision-tower
    // support + a key remap — it is NOT a drop-in like MetaCLIP. Included so the config exists once that
    // encoder work lands; do not treat its vision cos-sim as a passing gate until then.

    /// <summary><c>BAAI/EVA02-CLIP-B-16</c> — text tower loads as standard CLIP; the EVA-02 vision tower
    /// (RoPE + SwiGLU + sub-LN) is NOT yet supported by <see cref="ClipVisionEncoder"/>. See the section
    /// comment: vision-side parity is blocked on encoder work, not config.</summary>
    public static ClipPreset EvaClip02B16 => new()
    {
        Name = "BAAI/EVA02-CLIP-B-16",
        TextConfig = new ClipTextEncoderConfig
        {
            HiddenSize = 512,
            IntermediateSize = 2048,
            NumLayers = 12,
            NumHeads = 8,
            MaxPositionEmbeddings = 77,
            VocabSize = 49408,
            UseQuickGelu = true,
            ProjectionDim = 512,
        },
        VisionConfig = new ClipVisionEncoderConfig
        {
            HiddenSize = 768,
            NumLayers = 12,
            NumHeads = 12,
            IntermediateSize = 2048,
            ImageSize = 224,
            PatchSize = 16,
            ProjectionDim = 512,
            UseQuickGelu = false,
        },
    };
}
