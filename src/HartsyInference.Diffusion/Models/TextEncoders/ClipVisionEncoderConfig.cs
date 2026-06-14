namespace HartsyInference.Diffusion.Models.TextEncoders;

/// <summary>Configuration for CLIP vision-tower transformers used as image-prompt encoders for diffusion conditioning. The vision tower mirrors the text tower's transformer-encoder architecture but ingests image patches via a Conv2D patch embedding (stride=patch_size, no bias) plus a learned <c>class_embedding</c> CLS token, and skips the causal attention mask (vision tokens attend bidirectionally).
/// <para>The standard IP-Adapter family uses CLIP-ViT-H/14 with a 1024-dim visual_projection — the projected CLS embedding is what gets MLP'd into 4 image-prompt tokens. IP-Adapter Plus instead consumes the penultimate-layer hidden states (all 257 patch tokens, hidden_size=1280) through a Perceiver-style resampler.</para></summary>
public sealed record ClipVisionEncoderConfig
{
    /// <summary>Hidden dimension of the transformer (1280 for ViT-H/14).</summary>
    public required int HiddenSize { get; init; }

    /// <summary>Number of transformer layers (32 for ViT-H/14).</summary>
    public required int NumLayers { get; init; }

    /// <summary>Number of attention heads (16 for ViT-H/14, head_dim = hidden / heads = 80).</summary>
    public required int NumHeads { get; init; }

    /// <summary>FFN intermediate size (5120 for ViT-H/14, 4× hidden).</summary>
    public required int IntermediateSize { get; init; }

    /// <summary>Edge length of the input image (224 standard). Patch grid = (image_size / patch_size)².</summary>
    public int ImageSize { get; init; } = 224;

    /// <summary>Patch edge in pixels (14 for ViT-H/14, 16 for ViT-L). num_patches = (image_size / patch_size)².</summary>
    public int PatchSize { get; init; } = 14;

    /// <summary>Number of input image channels (3 for RGB).</summary>
    public int NumChannels { get; init; } = 3;

    /// <summary>Output dim of the optional visual_projection layer that maps the CLS token from <see cref="HiddenSize"/> down to a contrastive-aligned space. <c>1024</c> for ViT-H/14 with the SDXL IP-Adapter weights. Set to <c>0</c> if the checkpoint doesn't ship a projection.</summary>
    public int ProjectionDim { get; init; } = 1024;

    /// <summary>LayerNorm epsilon (1e-5 standard for OpenAI CLIP).</summary>
    public float LayerNormEps { get; init; } = 1e-5f;

    /// <summary>Whether MLP uses Quick-GELU (<c>x * sigmoid(1.702 * x)</c>, OpenAI's variant) instead of standard GELU. <c>true</c> for OpenAI CLIP, <c>false</c> for OpenCLIP. CLIP-ViT-H/14 ships in both flavors — IP-Adapter typically uses the OpenCLIP one (false).</summary>
    public bool UseQuickGelu { get; init; } = false;

    /// <summary>CLIP-ViT-H/14 preset matching <c>laion/CLIP-ViT-H-14-laion2B-s32B-b79K</c> (the standard image encoder for IP-Adapter SDXL). 32 layers, hidden=1280, 16 heads, head_dim=80, intermediate=5120, patch=14, projection_dim=1024. OpenCLIP weights → standard GELU.</summary>
    public static ClipVisionEncoderConfig ViTH14 => new()
    {
        HiddenSize = 1280,
        NumLayers = 32,
        NumHeads = 16,
        IntermediateSize = 5120,
        ImageSize = 224,
        PatchSize = 14,
        ProjectionDim = 1024,
        UseQuickGelu = false,
    };

    /// <summary>CLIP-ViT-L/14 preset (224×224 image, 12 layers, hidden=768). Used by some SD 1.5 IP-Adapter checkpoints. OpenAI weights → quick-GELU.</summary>
    public static ClipVisionEncoderConfig ViTL14 => new()
    {
        HiddenSize = 1024,
        NumLayers = 24,
        NumHeads = 16,
        IntermediateSize = 4096,
        ImageSize = 224,
        PatchSize = 14,
        ProjectionDim = 768,
        UseQuickGelu = true,
    };
}
