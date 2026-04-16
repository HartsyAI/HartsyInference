namespace SharpInference.Diffusion.Requests;

/// <summary>Request parameters for text-to-image generation.</summary>
public record TextToImageRequest
{
    /// <summary>The text prompt to generate an image from.</summary>
    public required string Prompt { get; init; }

    /// <summary>Optional negative prompt for classifier-free guidance.</summary>
    public string NegativePrompt { get; init; } = "";

    /// <summary>Number of denoising steps. Default: 20.</summary>
    public int Steps { get; init; } = 20;

    /// <summary>Classifier-free guidance scale. Higher = more prompt adherence. Default: 7.5.</summary>
    public float CfgScale { get; init; } = 7.5f;

    /// <summary>Output image width in pixels. Must be divisible by 8. Default: 512.</summary>
    public int Width { get; init; } = 512;

    /// <summary>Output image height in pixels. Must be divisible by 8. Default: 512.</summary>
    public int Height { get; init; } = 512;

    /// <summary>Random seed for reproducibility. Null = random.</summary>
    public int? Seed { get; init; }

    /// <summary>Scheduler to use. Null = default (Euler).</summary>
    public string? Scheduler { get; init; }
}
