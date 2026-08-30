namespace HartsyInference.Engine.Requests;

/// <summary>Optional per-request overrides for a model's swappable sub-components (VAE and the various text encoders). Each value is a model id or local path; null leaves the recipe's default for that component in place.</summary>
public sealed record ComponentOverrides
{
    /// <summary>VAE override.</summary>
    public string? Vae { get; init; }

    /// <summary>Video VAE override; when null the legacy <see cref="Vae"/> value remains the video fallback.</summary>
    public string? VideoVae { get; init; }

    /// <summary>Audio VAE override, independent of <see cref="VideoVae"/>.</summary>
    public string? AudioVae { get; init; }

    /// <summary>T5-XXL text-encoder override.</summary>
    public string? T5Xxl { get; init; }

    /// <summary>CLIP-L text-encoder override.</summary>
    public string? ClipL { get; init; }

    /// <summary>CLIP-G text-encoder override.</summary>
    public string? ClipG { get; init; }

    /// <summary>CLIP vision-encoder override.</summary>
    public string? ClipVision { get; init; }

    /// <summary>Qwen text-encoder override.</summary>
    public string? Qwen { get; init; }

    /// <summary>LLaMA text-encoder override.</summary>
    public string? Llama { get; init; }

    /// <summary>Gemma text-encoder override.</summary>
    public string? Gemma { get; init; }
}
