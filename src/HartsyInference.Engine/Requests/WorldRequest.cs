namespace HartsyInference.Engine.Requests;

/// <summary>Native interactive-world request: opens a stateful session seeded by a prompt and/or an initial frame.</summary>
public sealed record WorldRequest
{
    /// <summary>The seeding prompt.</summary>
    public string Prompt { get; init; } = "";

    /// <summary>Optional initial frame the world starts from.</summary>
    public ImageData? InitImage { get; init; }

    /// <summary>Number of denoising steps per generated frame; 0 uses the model default.</summary>
    public int Steps { get; init; }

    /// <summary>RNG seed; negative means a random seed.</summary>
    public long Seed { get; init; } = -1;
}
