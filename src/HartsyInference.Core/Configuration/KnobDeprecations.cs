namespace HartsyInference.Core.Configuration;

/// <summary>Setting ids that were renamed or removed. A settings file, a profile or <c>--set</c> naming one is accepted
/// with a warning — the settings file is shared with the engine inside SwarmUI, and an unknown id there would stop every
/// load — until the release in <see cref="Entry.AcceptedThrough"/>, after which a test fails until the entry is deleted.</summary>
public static class KnobDeprecations
{
    /// <summary>One retired id.</summary>
    /// <param name="OldId">The retired id.</param>
    /// <param name="NewId">The id that replaces it, or null when the setting was removed outright (its value is ignored).</param>
    /// <param name="MapValue">Converts the old setting's value, as text, to the new one's (an inverted switch, a renamed
    /// mode), which the new setting then parses; null keeps the text.</param>
    /// <param name="AcceptedThrough">The last <c>alpha.N</c> that still accepts the old id.</param>
    /// <param name="Note">What changed, shown with the warning.</param>
    public sealed record Entry(string OldId, string? NewId, Func<string, string>? MapValue, int AcceptedThrough, string Note);

    private static readonly System.Collections.Concurrent.ConcurrentDictionary<string, Entry> _entries = new(StringComparer.Ordinal);

    /// <summary>Every retired id still accepted.</summary>
    public static IReadOnlyCollection<Entry> All => [.. _entries.Values];

    /// <summary>The entry for <paramref name="id"/>, or null when it is not retired.</summary>
    public static Entry? Find(string id) => _entries.TryGetValue(id, out Entry? e) ? e : null;

    /// <summary>The id <paramref name="id"/> stands for now: its replacement when it was renamed, else itself.</summary>
    public static string Resolve(string id) =>
        KnobRegistry.Find(id) is null && Find(id) is { NewId: { } renamed } ? renamed : id;

    /// <summary>The text a value of a retired setting means for its replacement.</summary>
    public static string MapText(Entry entry, string text) => entry.MapValue is null ? text : entry.MapValue(text);

    /// <summary>The warning a use of <paramref name="entry"/> prints.</summary>
    public static string Warning(Entry entry) => entry.NewId is null
        ? $"Setting '{entry.OldId}' was removed and is ignored: {entry.Note}"
        : $"Setting '{entry.OldId}' is now '{entry.NewId}': {entry.Note}";

    internal static void Add(Entry entry)
    {
        if (!_entries.TryAdd(entry.OldId, entry)) throw new ArgumentException($"'{entry.OldId}' is already retired.");
    }

    internal static void Remove(string oldId) => _entries.TryRemove(oldId, out _);
}
