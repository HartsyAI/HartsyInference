namespace HartsyInference.Engine.Recipes.Image;

/// <summary>How a Qwen-Image-Edit build expects its references presented: how many, at what vision-tower area, and
/// whether each is labelled <c>Picture N:</c>. The two presets mirror ComfyUI's two encode nodes.</summary>
public sealed record QwenImageEditTemplate
{
    /// <summary>Qwen-Image-Edit (v1), ComfyUI <c>TextEncodeQwenImageEdit</c>: one unlabelled reference shown to the
    /// vision tower at the same ~1 MP area as the VAE copy.</summary>
    public static QwenImageEditTemplate Edit { get; } = new QwenImageEditTemplate
    {
        MaxReferences = 1,
        VisionTargetArea = 1024 * 1024,
        LabelPictures = false,
    };

    /// <summary>Qwen-Image-Edit-2509/2511 (Plus), ComfyUI <c>TextEncodeQwenImageEditPlus</c>: up to three references,
    /// each labelled <c>Picture N:</c> and shown to the vision tower at ~384².</summary>
    public static QwenImageEditTemplate EditPlus { get; } = new QwenImageEditTemplate
    {
        MaxReferences = 3,
        VisionTargetArea = 384 * 384,
        LabelPictures = true,
    };

    /// <summary>Reference slots the template was trained with.</summary>
    public required int MaxReferences { get; init; }

    /// <summary>Pixel budget for the copy fed to the Qwen2.5-VL vision tower.</summary>
    public required int VisionTargetArea { get; init; }

    /// <summary>Whether each reference is prefixed <c>Picture N: </c>.</summary>
    public required bool LabelPictures { get; init; }
}
