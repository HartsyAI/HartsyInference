using System.Globalization;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Services;

/// <summary>Maps a handler-produced <see cref="GeneratedArtifact"/> audio result onto the typed <see cref="AudioResult"/>.</summary>
internal static class AudioResults
{
    /// <summary>Builds an <see cref="AudioResult"/> from an audio artifact's encoded bytes and metadata.</summary>
    public static AudioResult From(GeneratedArtifact artifact) => new AudioResult
    {
        Data = artifact.FileBytes ?? Array.Empty<byte>(),
        Format = artifact.Extension,
        DurationSeconds = MetaDouble(artifact, "duration"),
        SampleRate = MetaInt(artifact, "sample_rate"),
        Meta = new Dictionary<string, string>(artifact.Meta),
    };

    private static double MetaDouble(GeneratedArtifact artifact, string key) =>
        artifact.Meta.TryGetValue(key, out string? value) && double.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out double parsed) ? parsed : 0.0;

    private static int MetaInt(GeneratedArtifact artifact, string key) =>
        artifact.Meta.TryGetValue(key, out string? value) && int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out int parsed) ? parsed : 0;
}
