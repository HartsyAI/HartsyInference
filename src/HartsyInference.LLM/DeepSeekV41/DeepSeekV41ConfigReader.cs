using System.Text.Json;
using HartsyInference.Core.Exceptions;

namespace HartsyInference.LLM.DeepSeekV41;

/// <summary>Typed reads of one config JSON object, each failing with the field name so a bad config is fixed from the message alone.</summary>
internal readonly struct DeepSeekV41ConfigReader(JsonElement element, string scope)
{
    public bool Has(string name) => element.TryGetProperty(name, out JsonElement value) && value.ValueKind != JsonValueKind.Null;

    public int Int(string name) => Int(name, out int value) ? value : throw Missing(name, "an integer");

    public int Int(string name, int fallback) => Int(name, out int value) ? value : fallback;

    public long Long(string name) =>
        Find(name) is { ValueKind: JsonValueKind.Number } n && n.TryGetInt64(out long value) ? value : throw Missing(name, "an integer");

    public double Double(string name) =>
        Find(name) is { ValueKind: JsonValueKind.Number } n ? n.GetDouble() : throw Missing(name, "a number");

    public double Double(string name, double fallback) =>
        Find(name) is { ValueKind: JsonValueKind.Number } n ? n.GetDouble() : fallback;

    public bool Bool(string name, bool fallback) =>
        Find(name) is { ValueKind: JsonValueKind.True or JsonValueKind.False } b ? b.GetBoolean() : fallback;

    public string String(string name) =>
        Find(name) is { ValueKind: JsonValueKind.String } s ? s.GetString()! : throw Missing(name, "a string");

    public string? OptionalString(string name) => Find(name) is { ValueKind: JsonValueKind.String } s ? s.GetString() : null;

    public int[] IntArray(string name)
    {
        if (Find(name) is not { ValueKind: JsonValueKind.Array } array)
            throw Missing(name, "an integer array");
        List<int> values = new List<int>();
        foreach (JsonElement item in array.EnumerateArray())
        {
            if (item.ValueKind != JsonValueKind.Number || !item.TryGetInt32(out int value))
                throw Missing(name, "an integer array");
            values.Add(value);
        }
        return values.ToArray();
    }

    public long[] LongArray(string name)
    {
        if (Find(name) is not { ValueKind: JsonValueKind.Array } array)
            throw Missing(name, "an integer array");
        List<long> values = new List<long>();
        foreach (JsonElement item in array.EnumerateArray())
        {
            if (item.ValueKind != JsonValueKind.Number || !item.TryGetInt64(out long value))
                throw Missing(name, "an integer array");
            values.Add(value);
        }
        return values.ToArray();
    }

    public DeepSeekV41ConfigReader? Object(string name) =>
        Find(name) is { ValueKind: JsonValueKind.Object } o ? new DeepSeekV41ConfigReader(o, $"{scope}.{name}") : null;

    private bool Int(string name, out int value)
    {
        value = 0;
        return Find(name) is { ValueKind: JsonValueKind.Number } n && n.TryGetInt32(out value);
    }

    private JsonElement? Find(string name) =>
        element.TryGetProperty(name, out JsonElement value) && value.ValueKind != JsonValueKind.Null ? value : null;

    private HartsyInferenceException Missing(string name, string expected) =>
        new($"DeepSeek-V4.1 config: '{scope}.{name}' must be {expected}.");
}
