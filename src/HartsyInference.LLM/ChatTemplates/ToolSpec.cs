namespace HartsyInference.LLM.ChatTemplates;

/// <summary>A tool schema in OpenAI form (<c>{"type":"function","function":{...}}</c>, optional <c>namespace</c>), kept as JSON text so key order survives into the prompt.</summary>
public sealed record ToolSpec(string Json)
{
    /// <summary>Wraps OpenAI-format tool JSON, validating that it parses and carries a <c>function</c> object.</summary>
    public static ToolSpec FromJson(string json)
    {
        ArgumentNullException.ThrowIfNull(json);
        if (PyJson.Parse(json) is not PyJsonObject tool || tool.GetOrNull("function") is not PyJsonObject)
            throw new ArgumentException("Tool JSON must be an object with a 'function' object.", nameof(json));
        return new ToolSpec(json);
    }
}
