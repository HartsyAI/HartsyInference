namespace HartsyInference.Engine.Audio.Wake.Wyoming;

/// <summary>Who made an advertised artifact and where it came from. Home Assistant refuses to parse an <c>info</c> entry that omits this, so every program and model carries one.</summary>
public sealed record WyomingAttribution
{
    /// <summary>Default credit for artifacts served by this engine.</summary>
    public static WyomingAttribution Engine { get; } = new()
    {
        Name = "HartsyInference",
        Url = "https://github.com/HartsyAI/HartsyInference",
    };

    public required string Name { get; init; }

    public required string Url { get; init; }
}
