namespace HartsyInference.Engine.Planning;

/// <summary>A summary of the terms a model's weights are used under, with what a host must show its users.</summary>
public sealed record MeshLicence
{
    /// <summary>The licence in one line.</summary>
    public required string Summary { get; init; }

    /// <summary>Conditions that bind a user: regions, scale limits, training bans. Empty when the licence adds none.</summary>
    public IReadOnlyList<string> Restrictions { get; init; } = [];

    /// <summary>The Hugging Face repositories among the model's files that need an access token (gated); empty when none does.</summary>
    public IReadOnlyList<string> GatedRepos { get; init; } = [];
}
