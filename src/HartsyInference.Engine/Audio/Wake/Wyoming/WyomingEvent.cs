using System.Text.Json;

namespace HartsyInference.Engine.Audio.Wake.Wyoming;

/// <summary>One decoded Wyoming event: its type, the merged <c>data</c> object, and an optional binary payload.
///
/// <para>Owns the <see cref="JsonDocument"/> backing <see cref="Data"/>, so every event read off the wire must be
/// disposed once its fields have been consumed.</para></summary>
public sealed class WyomingEvent : IDisposable
{
    private readonly JsonDocument? _data;

    /// <summary>Wyoming event name, e.g. <c>describe</c>, <c>audio-chunk</c>, <c>synthesize</c>.</summary>
    public string Type { get; }

    /// <summary>Raw binary payload, or null when the event carried none.</summary>
    public byte[]? Payload { get; }

    /// <summary>Valid byte count in <see cref="Payload"/>.</summary>
    public int PayloadLength { get; }

    internal WyomingEvent(string type, JsonDocument? data, byte[]? payload, int payloadLength)
    {
        Type = type;
        _data = data;
        Payload = payload;
        PayloadLength = payloadLength;
    }

    /// <summary>The event's <c>data</c> object; <see cref="JsonValueKind.Undefined"/> when it carried none.</summary>
    public JsonElement Data => _data?.RootElement ?? default;

    /// <summary>A string field of <c>data</c>, or null when absent or not a string.</summary>
    public string? GetString(string key) =>
        TryGet(key, out JsonElement value) && value.ValueKind == JsonValueKind.String ? value.GetString() : null;

    /// <summary>An integer field of <c>data</c>, or null when absent or not a number.</summary>
    public int? GetInt32(string key) =>
        TryGet(key, out JsonElement value) && value.ValueKind == JsonValueKind.Number && value.TryGetInt32(out int i) ? i : null;

    /// <summary>A string-array field of <c>data</c> with non-string entries skipped, or null when absent.</summary>
    public IReadOnlyList<string>? GetStringArray(string key)
    {
        if (!TryGet(key, out JsonElement value) || value.ValueKind != JsonValueKind.Array) return null;
        List<string> items = [];
        foreach (JsonElement item in value.EnumerateArray())
        {
            if (item.ValueKind == JsonValueKind.String && item.GetString() is string text) items.Add(text);
        }
        return items;
    }

    /// <summary>A nested object field of <c>data</c> (Wyoming's <c>synthesize.voice</c>), or null when absent.</summary>
    public JsonElement? GetObject(string key) =>
        TryGet(key, out JsonElement value) && value.ValueKind == JsonValueKind.Object ? value : null;

    private bool TryGet(string key, out JsonElement value)
    {
        JsonElement data = Data;
        if (data.ValueKind != JsonValueKind.Object)
        {
            value = default;
            return false;
        }
        return data.TryGetProperty(key, out value);
    }

    public void Dispose() => _data?.Dispose();
}
