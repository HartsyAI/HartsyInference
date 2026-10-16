using System.Globalization;
using System.Text.Json;

namespace HartsyInference.LLM.ChatTemplates;

/// <summary>JSON Schema facts that tool-call parsers read from OpenAI-format <see cref="ToolSpec"/>s.</summary>
public static class ToolSchema
{
    /// <summary>For each offered function, whether each of its parameters takes raw text. Formats that write argument values bare
    /// (Muse Spark's <c>&lt;atem:parameter&gt;</c>) keep a value verbatim when its schema admits a string and read it as JSON otherwise, the rule
    /// llama.cpp applies. A spec that does not parse or has no function name is skipped.</summary>
    public static IReadOnlyDictionary<string, IReadOnlyDictionary<string, bool>> StringParameters(IReadOnlyList<ToolSpec>? tools)
    {
        Dictionary<string, IReadOnlyDictionary<string, bool>> byTool = new(StringComparer.Ordinal);
        if (tools is null) return byTool;
        foreach (ToolSpec tool in tools)
        {
            JsonDocument doc;
            try
            {
                doc = JsonDocument.Parse(tool.Json);
            }
            catch (JsonException)
            {
                continue;
            }
            using (doc)
            {
                if (!doc.RootElement.TryGetProperty("function", out JsonElement function) || function.ValueKind != JsonValueKind.Object) continue;
                if (!function.TryGetProperty("name", out JsonElement name) || name.ValueKind != JsonValueKind.String) continue;
                Dictionary<string, bool> parameters = new(StringComparer.Ordinal);
                if (function.TryGetProperty("parameters", out JsonElement schema) && schema.ValueKind == JsonValueKind.Object
                    && schema.TryGetProperty("properties", out JsonElement properties) && properties.ValueKind == JsonValueKind.Object)
                {
                    foreach (JsonProperty p in properties.EnumerateObject()) parameters[p.Name] = MayBeString(p.Value, schema);
                }
                byTool[name.GetString()!] = parameters;
            }
        }
        return byTool;
    }

    /// <summary>True when some value valid under <paramref name="schema"/> is a JSON string. Mirrors llama.cpp's schema reading
    /// (<c>common/json-schema.cpp</c>, <c>may_be_string</c>) keyword by keyword and in its order: <c>$ref</c> (local, resolved against
    /// <paramref name="root"/>; a cycle is not a string), <c>oneOf</c> over <c>anyOf</c> (any branch), a <c>type</c> array (any listed type), <c>const</c>,
    /// <c>enum</c> (any string member), then the type; an untyped schema is a string only through string keywords, and <c>allOf</c> needs every
    /// constraining part to admit one. An unconstrained schema is not a string.</summary>
    public static bool MayBeString(JsonElement schema, JsonElement root) => Classify(schema, root, [], null) == Admits.String;

    /// <summary>What a schema node admits, as far as the string question goes; <see cref="Admits.Anything"/> is llama.cpp's unconstrained node,
    /// which <c>allOf</c> skips.</summary>
    private enum Admits { Anything, String, NoString }

    private static readonly string[] StringFormats = ["date", "time", "date-time", "uuid", "uuid1", "uuid2", "uuid3", "uuid4", "uuid5"];

    private static Admits Classify(JsonElement schema, JsonElement root, HashSet<string> visiting, string? typeOverride)
    {
        if (schema.ValueKind != JsonValueKind.Object) return Admits.NoString;
        if (schema.TryGetProperty("$ref", out JsonElement reference))
        {
            string path = reference.ValueKind == JsonValueKind.String ? reference.GetString()! : "";
            if (!TryResolve(root, path, out JsonElement target) || !visiting.Add(path)) return Admits.NoString;
            Admits resolved = Classify(target, root, visiting, null);
            visiting.Remove(path);
            return resolved == Admits.String ? Admits.String : Admits.NoString;
        }
        if (schema.TryGetProperty("oneOf", out JsonElement branches) || schema.TryGetProperty("anyOf", out branches))
            return AnyString(branches, root, visiting);
        string typeName = typeOverride ?? "";
        if (typeOverride is null && schema.TryGetProperty("type", out JsonElement type))
        {
            if (type.ValueKind == JsonValueKind.Array)
            {
                foreach (JsonElement t in type.EnumerateArray())
                {
                    if (t.ValueKind == JsonValueKind.String && Classify(schema, root, visiting, t.GetString()) == Admits.String) return Admits.String;
                }
                return Admits.NoString;
            }
            if (type.ValueKind != JsonValueKind.String) return Admits.NoString;
            typeName = type.GetString()!;
        }
        if (schema.TryGetProperty("const", out JsonElement constant))
            return constant.ValueKind == JsonValueKind.String ? Admits.String : Admits.NoString;
        if (schema.TryGetProperty("enum", out JsonElement members))
        {
            if (members.ValueKind != JsonValueKind.Array) return Admits.NoString;
            foreach (JsonElement member in members.EnumerateArray())
            {
                if (member.ValueKind == JsonValueKind.String) return Admits.String;
            }
            return Admits.NoString;
        }
        bool hasProperties = schema.TryGetProperty("properties", out _)
            || (schema.TryGetProperty("additionalProperties", out JsonElement additional) && additional.ValueKind != JsonValueKind.True);
        bool hasAllOf = schema.TryGetProperty("allOf", out JsonElement parts);
        switch (typeName)
        {
            case "":
                if (hasProperties) return Admits.NoString;
                if (hasAllOf) return AllString(parts, root, visiting);
                if (schema.TryGetProperty("items", out _) || schema.TryGetProperty("prefixItems", out _)) return Admits.NoString;
                if (schema.TryGetProperty("pattern", out _) || schema.TryGetProperty("minLength", out _) || schema.TryGetProperty("maxLength", out _))
                    return Admits.String;
                return schema.TryGetProperty("format", out JsonElement format) && format.ValueKind == JsonValueKind.String
                    && StringFormats.Contains(format.GetString(), StringComparer.Ordinal) ? Admits.String : Admits.Anything;
            case "object":
                return !hasProperties && hasAllOf ? AllString(parts, root, visiting) : Admits.NoString;
            case "string":
                return hasAllOf ? AllString(parts, root, visiting) : Admits.String;
            default:
                return Admits.NoString;
        }
    }

    private static Admits AnyString(JsonElement branches, JsonElement root, HashSet<string> visiting)
    {
        if (branches.ValueKind != JsonValueKind.Array) return Admits.NoString;
        foreach (JsonElement branch in branches.EnumerateArray())
        {
            if (Classify(branch, root, visiting, null) == Admits.String) return Admits.String;
        }
        return Admits.NoString;
    }

    private static Admits AllString(JsonElement parts, JsonElement root, HashSet<string> visiting)
    {
        if (parts.ValueKind != JsonValueKind.Array) return Admits.NoString;
        bool constrained = false;
        foreach (JsonElement part in parts.EnumerateArray())
        {
            Admits admits = Classify(part, root, visiting, null);
            if (admits == Admits.Anything) continue;
            if (admits == Admits.NoString) return Admits.NoString;
            constrained = true;
        }
        return constrained ? Admits.String : Admits.NoString;
    }

    /// <summary>Follows a local JSON pointer (<c>#/$defs/Name</c>, <c>#/anyOf/1</c>) from <paramref name="root"/> through object keys and array
    /// indices, as llama.cpp's <c>resolve_ref</c> does; remote references do not resolve.</summary>
    private static bool TryResolve(JsonElement root, string reference, out JsonElement target)
    {
        target = root;
        if (!reference.StartsWith('#')) return false;
        foreach (string raw in reference[1..].Split('/', StringSplitOptions.RemoveEmptyEntries))
        {
            string segment = raw.Replace("~1", "/", StringComparison.Ordinal).Replace("~0", "~", StringComparison.Ordinal);
            if (target.ValueKind == JsonValueKind.Object && target.TryGetProperty(segment, out JsonElement child))
            {
                target = child;
            }
            else if (target.ValueKind == JsonValueKind.Array && int.TryParse(segment, NumberStyles.None, CultureInfo.InvariantCulture, out int index)
                && index < target.GetArrayLength())
            {
                target = target[index];
            }
            else
            {
                return false;
            }
        }
        return true;
    }
}
