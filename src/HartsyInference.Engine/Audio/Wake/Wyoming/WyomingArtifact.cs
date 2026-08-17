namespace HartsyInference.Engine.Audio.Wake.Wyoming;

/// <summary>One entry advertised in the <c>info</c> manifest — an ASR model, a TTS voice, or a wake model.
///
/// <para>Wyoming's three artifact shapes differ only in which optional field they carry (<c>phrase</c> for wake
/// models, <c>speakers</c> for TTS voices), so one record covers all three rather than three near-identical ones.
/// <see cref="ModelId"/> and <see cref="VoiceName"/> are the mapping back to this engine: Home Assistant selects
/// by <see cref="Name"/>, and that name is what the engine's model resolver would otherwise be handed.</para></summary>
public sealed record WyomingArtifact
{
    /// <summary>The name Home Assistant shows and sends back on <c>transcribe</c> / <c>synthesize</c>.</summary>
    public required string Name { get; init; }

    /// <summary>Engine model id passed to the model resolver; null means <see cref="Name"/> is already one.</summary>
    public string? ModelId { get; init; }

    /// <summary>Built-in voice passed to the speech pipeline (a Kokoro voice pack, say); null uses the model default.</summary>
    public string? VoiceName { get; init; }

    public string? Description { get; init; }

    public string? Version { get; init; }

    /// <summary>ISO language codes. Required by Wyoming on every model and voice; an empty array parses but
    /// leaves Home Assistant with nothing to match a pipeline language against.</summary>
    public IReadOnlyList<string> Languages { get; init; } = ["en"];

    /// <summary>Spoken phrase for a wake model, when it differs from the model name.</summary>
    public string? Phrase { get; init; }

    /// <summary>Named speakers inside a multi-speaker voice; null for single-speaker voices.</summary>
    public IReadOnlyList<string>? Speakers { get; init; }

    public WyomingAttribution Attribution { get; init; } = WyomingAttribution.Engine;

    /// <summary>The engine model id this artifact resolves to.</summary>
    public string ResolvedModelId => ModelId ?? Name;
}
