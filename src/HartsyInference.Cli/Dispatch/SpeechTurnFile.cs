using System.Text.Json;
using System.Text.Json.Serialization;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Cli.Dispatch;

/// <summary>The JSON turn files <c>speak</c> reads for <c>--references</c> and <c>--history</c> and writes for <c>--save-turns</c>:
/// an array of <c>{ "text": …, "audio": "clip.wav" }</c> or <c>{ "text": …, "codes": [[…], …] }</c> entries, an audio path
/// relative to the file. A saved take reads back as history, so a revision is <c>--history</c> of the last <c>--save-turns</c>.</summary>
internal static class SpeechTurnFile
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNameCaseInsensitive = true,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
    };

    /// <summary>The turns in the file at <paramref name="rawPath"/>, or null when no path was given.</summary>
    /// <exception cref="FileNotFoundException">The file, or an audio file it names, does not exist.</exception>
    /// <exception cref="ArgumentException">An entry has no text, or neither audio nor codes.</exception>
    internal static IReadOnlyList<SpeechTurn>? Load(string? rawPath)
    {
        if (string.IsNullOrWhiteSpace(rawPath))
        {
            return null;
        }
        string path = Path.GetFullPath(rawPath.Trim().Trim('"'));
        if (!File.Exists(path))
        {
            throw new FileNotFoundException($"Turn file not found: {path}");
        }
        string baseDirectory = Path.GetDirectoryName(path) ?? "";
        Entry[] entries = JsonSerializer.Deserialize<Entry[]>(File.ReadAllText(path), JsonOptions)
            ?? throw new ArgumentException($"Turn file '{path}' holds no turns.");
        List<SpeechTurn> turns = new(entries.Length);
        for (int i = 0; i < entries.Length; i++)
        {
            Entry entry = entries[i];
            if (string.IsNullOrWhiteSpace(entry.Text))
            {
                throw new ArgumentException($"Turn {i} in '{path}' has no text.");
            }
            if (entry.Codes is not { Length: > 0 } && string.IsNullOrWhiteSpace(entry.Audio))
            {
                throw new ArgumentException($"Turn {i} in '{path}' has neither audio nor codes.");
            }
            turns.Add(new SpeechTurn
            {
                Text = entry.Text,
                Codes = entry.Codes is { Length: > 0 } codes ? codes : null,
                Audio = entry.Codes is { Length: > 0 } ? null : GenerationDispatch.LoadAudioClip(Path.Combine(baseDirectory, entry.Audio!)),
            });
        }
        return turns;
    }

    /// <summary>Writes <paramref name="turns"/> as text and codes, the form <see cref="Load"/> reads back as history.</summary>
    internal static void Save(string rawPath, IReadOnlyList<SpeechTurn> turns)
    {
        ArgumentNullException.ThrowIfNull(turns);
        Entry[] entries = [.. turns.Select(static t => new Entry(t.Text, null, t.Codes?.ToArray()))];
        File.WriteAllText(Path.GetFullPath(rawPath.Trim().Trim('"')), JsonSerializer.Serialize(entries, JsonOptions));
    }

    /// <summary>One turn as the file holds it.</summary>
    /// <param name="Text">The transcript or the turn's text.</param>
    /// <param name="Audio">An audio file, relative to the turn file.</param>
    /// <param name="Codes">Codec codes, one row per codebook.</param>
    private sealed record Entry(string Text, string? Audio, int[][]? Codes);
}
