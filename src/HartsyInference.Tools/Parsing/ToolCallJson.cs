using System.Text.Json;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Tools.Parsing;

/// <summary>Turns a completed JSON call span into <see cref="NativeToolCall"/>s: one object, an array of objects, or (for the <c>name{…}</c> forms) a bare arguments object under a name the parser already read. OpenAI's <c>{"function": {…}}</c> nesting is unwrapped.</summary>
internal static class ToolCallJson
{
    /// <summary>Id prefix; the suffix is the zero-based call index within the request.</summary>
    public const string IdPrefix = "call_";

    private const int MaxDepth = 64;

    /// <summary>Parses <paramref name="json"/> into <paramref name="into"/>, numbering ids from <paramref name="firstIndex"/> under <paramref name="idPrefix"/>; false when the text is not valid JSON or is not shaped like a call.</summary>
    public static bool TryParse(string json, IReadOnlyList<string> argumentKeys, string? presetName, List<NativeToolCall> into, int firstIndex, string idPrefix)
    {
        try
        {
            using JsonDocument doc = JsonDocument.Parse(json, new JsonDocumentOptions { MaxDepth = MaxDepth });
            JsonElement root = doc.RootElement;
            if (presetName is not null)
            {
                if (root.ValueKind != JsonValueKind.Object) return false;
                into.Add(Make(idPrefix, firstIndex, presetName, root.GetRawText()));
                return true;
            }
            if (root.ValueKind == JsonValueKind.Object) return TryAdd(root, argumentKeys, idPrefix, firstIndex, into);
            if (root.ValueKind != JsonValueKind.Array) return false;
            int count = 0;
            foreach (JsonElement element in root.EnumerateArray())
            {
                if (!TryAdd(element, argumentKeys, idPrefix, firstIndex + count, into)) return false;
                count++;
            }
            return count > 0;
        }
        catch (JsonException)
        {
            return false;
        }
    }

    /// <summary>The id for the call at <paramref name="index"/> under <paramref name="idPrefix"/>.</summary>
    public static string IdFor(string idPrefix, int index) => idPrefix + index.ToString(System.Globalization.CultureInfo.InvariantCulture);

    private static bool TryAdd(JsonElement element, IReadOnlyList<string> argumentKeys, string idPrefix, int index, List<NativeToolCall> into)
    {
        if (element.ValueKind != JsonValueKind.Object) return false;
        JsonElement call = element.TryGetProperty("function", out JsonElement function) && function.ValueKind == JsonValueKind.Object
            ? function : element;
        if (!call.TryGetProperty("name", out JsonElement name) || name.ValueKind != JsonValueKind.String) return false;
        string toolName = name.GetString() ?? "";
        if (toolName.Length == 0) return false;
        string arguments = "{}";
        for (int i = 0; i < argumentKeys.Count; i++)
        {
            if (!call.TryGetProperty(argumentKeys[i], out JsonElement value)) continue;
            arguments = value.ValueKind switch
            {
                JsonValueKind.Null or JsonValueKind.Undefined => "{}",
                // Some models pre-serialise the arguments into a string; hand the string through as the JSON text.
                JsonValueKind.String => value.GetString() ?? "{}",
                _ => value.GetRawText(),
            };
            break;
        }
        into.Add(Make(idPrefix, index, toolName, arguments));
        return true;
    }

    private static NativeToolCall Make(string idPrefix, int index, string name, string arguments)
        => new() { Id = IdFor(idPrefix, index), Name = name, Arguments = arguments };
}
