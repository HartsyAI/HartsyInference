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

    /// <summary>Releases a retired id stays accepted after the release that retires it.</summary>
    public const int Window = 100;

    private static readonly System.Collections.Concurrent.ConcurrentDictionary<string, Entry> _entries =
        new(Retired().Select(e => KeyValuePair.Create(e.OldId, e)), StringComparer.Ordinal);

    /// <summary>Every retired id still accepted (a snapshot).</summary>
    public static IReadOnlyCollection<Entry> All => [.. _entries.Values];

    /// <summary>The entry for <paramref name="id"/>, or null when it is not retired.</summary>
    public static Entry? Find(string id) => _entries.TryGetValue(id, out Entry? e) ? e : null;

    /// <summary>The id <paramref name="id"/> stands for now: its replacement when it was renamed, else itself.</summary>
    public static string Resolve(string id) =>
        KnobRegistry.Find(id) is null && Find(id) is { NewId: { } renamed } ? renamed : id;

    /// <summary>The text a value of a retired setting means for its replacement; a value the mapping rejects throws
    /// <see cref="InvalidOperationException"/> naming the setting and the value, like any other bad setting.</summary>
    public static string MapText(Entry entry, string text)
    {
        if (entry.MapValue is null) return text;
        try
        {
            return entry.MapValue(text);
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException(
                $"Setting '{entry.OldId}' (now '{entry.NewId}') has a value '{text}' that does not carry over: {ex.Message}", ex);
        }
    }

    /// <summary>The warning a use of <paramref name="entry"/> prints.</summary>
    public static string Warning(Entry entry) => entry.NewId is null
        ? $"Setting '{entry.OldId}' was removed and is ignored: {entry.Note}"
        : $"Setting '{entry.OldId}' is now '{entry.NewId}': {entry.Note}";

    private static Entry[] Retired()
    {
        // The attention kernels and switches retired in alpha.386.
        const int Attention386 = 386 + Window;
        return
        [
            new("numerics.sdpaV2", null, null, Attention386,
                "the TF32 FlashAttention-2 kernel it opted into is gone; F32 attention runs on cuDNN, the tiled core or the materialized path."),
            new("numerics.sdpaForceFlash", null, null, Attention386, "attention no longer has a forced per-row flash route."),
            new("numerics.sageV0", null, null, Attention386, "the v0 SageAttention kernel is gone; numerics.sageAttn governs the one that remains."),
            new("numerics.sageAttnExplicit", null, null, Attention386,
                "F32 operands no longer reach SageAttention; numerics.sageAttn governs the native-F16 path."),
            new("numerics.sageUnsafeF32VNarrow", null, null, Attention386,
                "F32 operands no longer reach SageAttention, so there is no V narrowing to accept."),
            new("numerics.sagePv", null, null, Attention386, "the native-F16 SageAttention path always accumulates PV in F16 per key step."),
        ];
    }

    internal static void Add(Entry entry)
    {
        if (!_entries.TryAdd(entry.OldId, entry)) throw new ArgumentException($"'{entry.OldId}' is already retired.");
    }

    internal static void Remove(string oldId) => _entries.TryRemove(oldId, out _);
}
