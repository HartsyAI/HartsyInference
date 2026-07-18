namespace HartsyInference.Engine.Requests;

/// <summary>Native, transport-agnostic text-to-image request. Transports (HTTP, CLI) map their own inputs onto this;
/// it is the single contract the engine's image generation accepts.</summary>
public sealed record ImageRequest
{
    /// <summary>The positive prompt.</summary>
    public required string Prompt { get; init; }

    /// <summary>The negative prompt, or null/empty for none.</summary>
    public string? NegativePrompt { get; init; }

    /// <summary>Output width in pixels.</summary>
    public int Width { get; init; } = 1024;

    /// <summary>Output height in pixels.</summary>
    public int Height { get; init; } = 1024;

    /// <summary>Number of denoising steps.</summary>
    public int Steps { get; init; } = 20;

    /// <summary>Classifier-free guidance scale.</summary>
    public float CfgScale { get; init; } = 7.5f;

    /// <summary>RNG seed; negative means a random seed is chosen per request.</summary>
    public long Seed { get; init; } = -1;

    /// <summary>CLIP-skip: number of final text-encoder layers to skip (0 = none).</summary>
    public int ClipSkip { get; init; }
}
