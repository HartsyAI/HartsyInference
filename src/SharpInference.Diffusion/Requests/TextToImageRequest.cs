using SharpInference.Core.Tensors;

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

    /// <summary>"CLIP skip" — how many layers from the end of the CLIP text encoder to take hidden states from. 1 = standard last layer (default), 2 = penultimate (common for SD1.5 anime checkpoints). Null = 1. Only honored by pipelines whose text encoder is CLIP-final-layer based (SD 1.5); SDXL already uses penultimate by spec.</summary>
    public int? ClipSkip { get; init; }

    /// <summary>Optional pre-built initial noise tensor. When non-null, overrides the seed-based noise generator — used for cross-runtime parity tests where the same noise tensor must flow into both PyTorch and SharpInference (PyTorch's <c>torch.Generator.manual_seed</c> and SharpInference's <c>SeedGenerator</c> use different RNGs and don't agree bit-for-bit on the same seed). Pipeline takes ownership and disposes after use. Shape must match the pipeline's expected initial latent shape (txt2img path; for img2img use <see cref="ImageToImageRequest.SourceImage"/>).</summary>
    public Tensor? InitialNoise { get; init; }
}
