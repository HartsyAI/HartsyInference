using System.Text;

namespace HartsyInference.LLM.ChatTemplates;

/// <summary>Tool-schema and tool-call rendering of the V4.1 prompt (namespaces, DSML parameters, schema JSON).</summary>
internal static class DeepSeekV41Tools
{
    public const string DsmlToken = "｜DSML｜";
    public const string ThinkStart = "<think>";
    public const string ThinkEnd = "</think>";
    public const string ToolCallsBlockName = " calls";
    public const string ToolCallTagName = " invoke";
    public const string ToolParameterTagName = " parameter";

    /// <summary>Renders the tools section appended to a system message.</summary>
    public static string RenderTools(IReadOnlyList<ToolSpec> tools)
    {
        StringBuilder schemas = new();
        for (int i = 0; i < tools.Count; i++)
        {
            if (i > 0) schemas.Append('\n');
            schemas.Append(PyJson.Dump(ToFunctionSchema(tools[i])));
        }
        return DeepSeekV41ToolsTemplate.Text
            .Replace("{dsml_token}", DsmlToken, StringComparison.Ordinal)
            .Replace("{tc_block_name}", ToolCallsBlockName, StringComparison.Ordinal)
            .Replace("{tool_call_tag_name}", ToolCallTagName, StringComparison.Ordinal)
            .Replace("{tool_parameter_tag_name}", ToolParameterTagName, StringComparison.Ordinal)
            .Replace("{thinking_start_token}", ThinkStart, StringComparison.Ordinal)
            .Replace("{thinking_end_token}", ThinkEnd, StringComparison.Ordinal)
            .Replace("{tool_schemas}", schemas.ToString(), StringComparison.Ordinal);
    }

    /// <summary>Renders the DSML calls block appended to an assistant turn (without the leading blank line).</summary>
    public static string RenderToolCalls(IReadOnlyList<ChatToolCall> calls)
    {
        StringBuilder invokes = new();
        for (int i = 0; i < calls.Count; i++)
        {
            if (i > 0) invokes.Append('\n');
            string name = QualifiedName(calls[i].Name, calls[i].Namespace is { Length: > 0 } ns ? ns : null);
            invokes.Append('<').Append(DsmlToken).Append(ToolCallTagName).Append(" name=\"").Append(name).Append("\">\n")
                .Append(EncodeArguments(calls[i].ArgumentsJson))
                .Append("\n</").Append(DsmlToken).Append(ToolCallTagName).Append('>');
        }
        return "<" + DsmlToken + ToolCallsBlockName + ">\n" + invokes + "\n</" + DsmlToken + ToolCallsBlockName + ">";
    }

    private static PyJsonObject ToFunctionSchema(ToolSpec spec)
    {
        if (PyJson.Parse(spec.Json) is not PyJsonObject tool || tool.GetOrNull("function") is not PyJsonObject source)
            throw new ArgumentException("Tool JSON must be an object with a 'function' object.");
        PyJsonObject function = source.Clone();
        object? toolNamespace = tool.GetOrNull("namespace");
        if (toolNamespace is not null) function["namespace"] = toolNamespace;

        object? ns = function.GetOrNull("namespace");
        object? nsName = ns is PyJsonObject nsObject ? nsObject["name"] : ns;
        if (function.GetOrNull("name") is not string rawName)
            throw new ArgumentException("Tool function has no string 'name'.");
        function["name"] = QualifiedName(rawName, nsName as string);
        function.Remove("namespace");

        if (ns is PyJsonObject nsDescribed && nsDescribed.GetOrNull("description") is string { Length: > 0 } prefix)
        {
            string own = function.GetOrNull("description") as string ?? string.Empty;
            function["description"] = prefix + "\n" + own;
        }
        return function;
    }

    // Mirrors _split_tool_name: a qualified name may repeat its namespace but never contradict it.
    private static string QualifiedName(string name, string? ns)
    {
        int sep = name.IndexOf("::", StringComparison.Ordinal);
        if (sep >= 0)
        {
            string prefix = name[..sep];
            if (ns is not null && ns != prefix) throw new ArgumentException($"Conflicting tool namespaces: {ns} != {prefix}");
            ns = prefix;
            name = name[(sep + 2)..];
        }
        if (name.Contains("::", StringComparison.Ordinal)) throw new ArgumentException($"Tool name must not contain '::': {name}");
        if (ns is not null && ns.Contains("::", StringComparison.Ordinal))
            throw new ArgumentException($"Tool namespace must not contain '::': {ns}");
        return ns is null ? name : ns + "::" + name;
    }

    private static string EncodeArguments(string argumentsJson)
    {
        PyJsonObject arguments = DecodeArguments(argumentsJson);
        StringBuilder sb = new();
        for (int i = 0; i < arguments.Keys.Count; i++)
        {
            string key = arguments.Keys[i];
            object? value = arguments[key];
            if (i > 0) sb.Append('\n');
            sb.Append('<').Append(DsmlToken).Append(ToolParameterTagName).Append(" name=\"").Append(key)
                .Append("\" string=\"").Append(value is string ? "true" : "false").Append("\">")
                .Append(value as string ?? PyJson.Dump(value))
                .Append("</").Append(DsmlToken).Append(ToolParameterTagName).Append('>');
        }
        return sb.ToString();
    }

    // Tolerates JSON strings, including double-encoded ones; anything else is wrapped as {"arguments": <original>}.
    private static PyJsonObject DecodeArguments(string argumentsJson)
    {
        object? current = argumentsJson;
        for (int attempt = 0; attempt < 2 && current is string text; attempt++)
        {
            try { current = PyJson.Parse(text); }
            catch (System.Text.Json.JsonException) { break; }
        }
        if (current is PyJsonObject parsed) return parsed;
        PyJsonObject wrapped = new();
        wrapped["arguments"] = argumentsJson;
        return wrapped;
    }
}
