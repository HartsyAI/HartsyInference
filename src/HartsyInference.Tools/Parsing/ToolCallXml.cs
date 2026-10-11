using System.Globalization;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Tools.Parsing;

/// <summary>Converters for the markup dialects: Qwen3.5 and Qwen3-Coder's <c>&lt;function=…&gt;&lt;parameter=…&gt;</c>, GLM-4.5's <c>&lt;arg_key&gt;/&lt;arg_value&gt;</c> pairs and DeepSeek-R1's fenced JSON blocks. Each takes the span the parser read up to its closing marker and returns the calls it holds; values are typed by the offered tool's JSON schema.</summary>
internal static partial class ToolCallXml
{
    private const string GlmOpen = "<tool_call>";
    private const string DeepSeekCallBegin = "<｜tool▁call▁begin｜>";

    [GeneratedRegex(@"<function=(?<name>[^>\s]+)\s*>(?<body>.*?)</function>", RegexOptions.Singleline)]
    private static partial Regex FunctionBlock();

    [GeneratedRegex(@"<parameter=(?<key>[^>\s]+)\s*>(?<value>.*?)</parameter>", RegexOptions.Singleline)]
    private static partial Regex Parameter();

    [GeneratedRegex(@"<arg_key>(?<key>.*?)</arg_key>\s*<arg_value>(?<value>.*?)</arg_value>", RegexOptions.Singleline)]
    private static partial Regex ArgPair();

    [GeneratedRegex("<｜tool▁call▁begin｜>[^<]*?<｜tool▁sep｜>(?<name>[^\\n]+)\\n```json\\n(?<args>.*?)\\n```<｜tool▁call▁end｜>", RegexOptions.Singleline)]
    private static partial Regex DeepSeekCall();

    /// <summary>Each offered tool's parameter types, by tool name then parameter name, read from its JSON schema (<c>properties.&lt;key&gt;.type</c>).</summary>
    public static IReadOnlyDictionary<string, IReadOnlyDictionary<string, string>> IndexSchemas(IReadOnlyList<ToolDefinition> tools)
    {
        Dictionary<string, IReadOnlyDictionary<string, string>> index = new(StringComparer.Ordinal);
        foreach (ToolDefinition tool in tools)
        {
            Dictionary<string, string> types = new(StringComparer.Ordinal);
            try
            {
                using JsonDocument doc = JsonDocument.Parse(string.IsNullOrWhiteSpace(tool.JsonSchema) ? "{}" : tool.JsonSchema);
                if (doc.RootElement.TryGetProperty("properties", out JsonElement props) && props.ValueKind == JsonValueKind.Object)
                {
                    foreach (JsonProperty prop in props.EnumerateObject())
                    {
                        if (prop.Value.ValueKind == JsonValueKind.Object && prop.Value.TryGetProperty("type", out JsonElement type) && type.ValueKind == JsonValueKind.String)
                            types[prop.Name] = type.GetString()!;
                    }
                }
            }
            catch (JsonException)
            {
                // A schema that does not parse types every parameter as a string.
            }
            index[tool.Name] = types;
        }
        return index;
    }

    /// <summary>Qwen's <c>&lt;function=NAME&gt;&lt;parameter=KEY&gt;value&lt;/parameter&gt;&lt;/function&gt;</c> blocks; every block in the span is one call.</summary>
    public static bool TryConvertFunctionBlocks(string span, IReadOnlyDictionary<string, IReadOnlyDictionary<string, string>>? schemas, string idPrefix, int firstIndex, List<NativeToolCall> into)
    {
        int before = into.Count;
        foreach (Match block in FunctionBlock().Matches(span))
        {
            string name = block.Groups["name"].Value;
            IReadOnlyDictionary<string, string>? types = schemas is not null && schemas.TryGetValue(name, out IReadOnlyDictionary<string, string>? known) ? known : null;
            List<(string Key, string Value)> parameters = [];
            foreach (Match parameter in Parameter().Matches(block.Groups["body"].Value))
                parameters.Add((parameter.Groups["key"].Value, TrimOneNewline(parameter.Groups["value"].Value)));
            into.Add(new NativeToolCall
            {
                Id = ToolCallJson.IdFor(idPrefix, firstIndex + into.Count - before),
                Name = name,
                Arguments = ObjectJson(parameters, types),
            });
        }
        return into.Count > before;
    }

    /// <summary>GLM-4.5's <c>name\n&lt;arg_key&gt;k&lt;/arg_key&gt;\n&lt;arg_value&gt;v&lt;/arg_value&gt;…</c> inside its tool-call tag; one call per span.</summary>
    public static bool TryConvertArgKey(string span, IReadOnlyDictionary<string, IReadOnlyDictionary<string, string>>? schemas, string idPrefix, int index, List<NativeToolCall> into)
    {
        string body = span.TrimStart();
        if (body.StartsWith(GlmOpen, StringComparison.Ordinal)) body = body[GlmOpen.Length..].TrimStart();
        int firstKey = body.IndexOf("<arg_key>", StringComparison.Ordinal);
        string name = (firstKey < 0 ? body : body[..firstKey]).Trim();
        if (name.Length == 0 || name.Contains('<') || name.Contains('>')) return false;
        IReadOnlyDictionary<string, string>? types = schemas is not null && schemas.TryGetValue(name, out IReadOnlyDictionary<string, string>? known) ? known : null;
        List<(string Key, string Value)> pairs = [];
        foreach (Match pair in ArgPair().Matches(body)) pairs.Add((pair.Groups["key"].Value.Trim(), pair.Groups["value"].Value));
        into.Add(new NativeToolCall { Id = ToolCallJson.IdFor(idPrefix, index), Name = name, Arguments = ObjectJson(pairs, types) });
        return true;
    }

    /// <summary>DeepSeek-R1's fenced-JSON calls: one or more <c>tool_call_begin … tool_call_end</c> blocks, each naming a function and carrying its arguments as a JSON object.</summary>
    public static bool TryConvertDeepSeek(string span, string idPrefix, int firstIndex, List<NativeToolCall> into)
    {
        int before = into.Count;
        foreach (Match call in DeepSeekCall().Matches(span))
        {
            string name = call.Groups["name"].Value.Trim();
            string args = call.Groups["args"].Value.Trim();
            if (name.Length == 0 || !IsJsonObject(args)) return false;
            into.Add(new NativeToolCall { Id = ToolCallJson.IdFor(idPrefix, firstIndex + into.Count - before), Name = name, Arguments = args });
        }
        return into.Count > before;
    }

    /// <summary>A JSON object from raw key/value text, each value typed by <paramref name="types"/> (string when unknown).</summary>
    private static string ObjectJson(IReadOnlyList<(string Key, string Value)> pairs, IReadOnlyDictionary<string, string>? types)
    {
        using MemoryStream stream = new();
        using (Utf8JsonWriter writer = new(stream))
        {
            writer.WriteStartObject();
            foreach ((string key, string value) in pairs)
            {
                writer.WritePropertyName(key);
                string? type = types is not null && types.TryGetValue(key, out string? known) ? known : null;
                WriteTyped(writer, value, type);
            }
            writer.WriteEndObject();
        }
        return Encoding.UTF8.GetString(stream.ToArray());
    }

    private static void WriteTyped(Utf8JsonWriter writer, string raw, string? type)
    {
        string trimmed = raw.Trim();
        switch (type)
        {
            case "integer" when long.TryParse(trimmed, NumberStyles.Integer, CultureInfo.InvariantCulture, out long whole):
                writer.WriteNumberValue(whole);
                return;
            case "number" when double.TryParse(trimmed, NumberStyles.Float, CultureInfo.InvariantCulture, out double real) && double.IsFinite(real):
                writer.WriteNumberValue(real);
                return;
            case "boolean" when bool.TryParse(trimmed, out bool flag):
                writer.WriteBooleanValue(flag);
                return;
            case "object" or "array" when IsJson(trimmed):
                writer.WriteRawValue(trimmed);
                return;
            default:
                writer.WriteStringValue(raw);
                return;
        }
    }

    /// <summary>A parameter value loses exactly one leading and one trailing newline: the markup puts one on each side of a value, and any others belong to the value.</summary>
    private static string TrimOneNewline(string value)
    {
        if (value.StartsWith('\n')) value = value[1..];
        if (value.EndsWith('\n')) value = value[..^1];
        return value;
    }

    private static bool IsJson(string text)
    {
        try
        {
            using JsonDocument _ = JsonDocument.Parse(text);
            return true;
        }
        catch (JsonException)
        {
            return false;
        }
    }

    private static bool IsJsonObject(string text)
    {
        try
        {
            using JsonDocument doc = JsonDocument.Parse(text);
            return doc.RootElement.ValueKind == JsonValueKind.Object;
        }
        catch (JsonException)
        {
            return false;
        }
    }
}
