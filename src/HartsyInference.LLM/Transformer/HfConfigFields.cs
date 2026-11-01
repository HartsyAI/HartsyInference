using System.Text.Json;
using HartsyInference.Core.Exceptions;

namespace HartsyInference.LLM.Transformer;

/// <summary>Typed reads of one <c>config.json</c> object, each failing with the field name so a bad config is fixed from the message alone. Mirrors <c>DeepSeekV41ConfigReader</c>'s shape.</summary>
internal readonly struct HfConfigFields(JsonElement element, string scope, string model)
{
    public int Int(string name) => Int(name, out int value) ? value : throw Missing(name, "an integer");

    public int Int(string name, int fallback) => Int(name, out int value) ? value : fallback;

    public double Double(string name, double fallback) =>
        Find(name) is { ValueKind: JsonValueKind.Number } n ? n.GetDouble() : fallback;

    public double Double(string name) =>
        Find(name) is { ValueKind: JsonValueKind.Number } n ? n.GetDouble() : throw Missing(name, "a number");

    public bool Bool(string name, bool fallback) =>
        Find(name) is { ValueKind: JsonValueKind.True or JsonValueKind.False } b ? b.GetBoolean() : fallback;

    public string? OptionalString(string name) => Find(name) is { ValueKind: JsonValueKind.String } s ? s.GetString() : null;

    /// <summary>The array <paramref name="name"/> as integers.</summary>
    public int[] Ints(string name) =>
        Find(name) is { ValueKind: JsonValueKind.Array } a ? [.. a.EnumerateArray().Select(static e => e.GetInt32())] : throw Missing(name, "an array of integers");

    /// <summary>The nested object <paramref name="name"/>, scoped under this one.</summary>
    public HfConfigFields Object(string name) =>
        Find(name) is { ValueKind: JsonValueKind.Object } o ? new HfConfigFields(o, $"{scope}.{name}", model) : throw Missing(name, "an object");

    private bool Int(string name, out int value)
    {
        value = 0;
        return Find(name) is { ValueKind: JsonValueKind.Number } n && n.TryGetInt32(out value);
    }

    private JsonElement? Find(string name) =>
        element.TryGetProperty(name, out JsonElement value) && value.ValueKind != JsonValueKind.Null ? value : null;

    private HartsyInferenceException Missing(string name, string expected) =>
        new($"{model} config: '{scope}.{name}' must be {expected}.");
}
