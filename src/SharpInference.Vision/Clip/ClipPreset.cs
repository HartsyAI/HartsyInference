using SharpInference.Diffusion.Models.TextEncoders;

namespace SharpInference.Vision.Clip;

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
}
