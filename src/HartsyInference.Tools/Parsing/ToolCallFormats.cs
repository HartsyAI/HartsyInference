namespace HartsyInference.Tools.Parsing;

/// <summary>The per-format rule tables and the model-id / template heuristic that picks a <see cref="ToolCallFormat"/>. Hermes is the default: it is what Qwen, GLM, DeepSeek and the Hermes fine-tunes emit, and its bare-object rule also reads the plain <c>{"name": …}</c> object most other instruct models fall back to.</summary>
public static class ToolCallFormats
{
    private static readonly string[] HermesArgumentKeys = ["arguments", "parameters"];
    private static readonly string[] LlamaArgumentKeys = ["parameters", "arguments"];

    private static readonly ToolCallFormatRules HermesRules = new()
    {
        Format = ToolCallFormat.Hermes,
        Markers =
        [
            new ToolCallMarker("<tool_call>", ToolCallPayload.JsonObject),
            new ToolCallMarker("{\"name\"", ToolCallPayload.JsonObject, TextIsPayload: true, Strict: true),
            new ToolCallMarker("{", ToolCallPayload.JsonObject, TextIsPayload: true, LineStartOnly: true, Strict: true),
        ],
        CloseMarker = "</tool_call>",
        ArgumentKeys = HermesArgumentKeys,
    };

    private static readonly ToolCallFormatRules Llama3Rules = new()
    {
        Format = ToolCallFormat.Llama3,
        Markers =
        [
            new ToolCallMarker("<|python_tag|>", ToolCallPayload.JsonObject),
            new ToolCallMarker("{\"name\"", ToolCallPayload.JsonObject, TextIsPayload: true, Strict: true),
            new ToolCallMarker("{", ToolCallPayload.JsonObject, TextIsPayload: true, LineStartOnly: true, Strict: true),
        ],
        ArgumentKeys = LlamaArgumentKeys,
    };

    private static readonly ToolCallFormatRules GemmaRules = new()
    {
        Format = ToolCallFormat.Gemma,
        Markers =
        [
            new ToolCallMarker("<|tool_call>", ToolCallPayload.GemmaCall),
            new ToolCallMarker("call:", ToolCallPayload.GemmaCall, Strict: true),
        ],
        CloseMarker = "<tool_call|>",
        ArgumentKeys = HermesArgumentKeys,
    };

    private static readonly ToolCallFormatRules MistralRules = new()
    {
        Format = ToolCallFormat.Mistral,
        Markers =
        [
            new ToolCallMarker("[TOOL_CALLS]", ToolCallPayload.JsonAny),
            new ToolCallMarker("[", ToolCallPayload.JsonArray, TextIsPayload: true, LineStartOnly: true, Strict: true),
            new ToolCallMarker("{\"name\"", ToolCallPayload.JsonObject, TextIsPayload: true, Strict: true),
            new ToolCallMarker("{", ToolCallPayload.JsonObject, TextIsPayload: true, LineStartOnly: true, Strict: true),
        ],
        NamedFormAtLineStart = true,
        ArgumentKeys = HermesArgumentKeys,
    };

    /// <summary>The rule table for <paramref name="format"/>.</summary>
    public static ToolCallFormatRules RulesFor(ToolCallFormat format) => format switch
    {
        ToolCallFormat.Hermes => HermesRules,
        ToolCallFormat.Llama3 => Llama3Rules,
        ToolCallFormat.Gemma => GemmaRules,
        ToolCallFormat.Mistral => MistralRules,
        _ => throw new ArgumentOutOfRangeException(nameof(format), format, "Unknown tool-call format."),
    };

    /// <summary>Picks a format from a model id, checkpoint name or the model's own chat template text; anything unrecognised is <see cref="ToolCallFormat.Hermes"/>.</summary>
    public static ToolCallFormat Detect(string? hint)
    {
        if (string.IsNullOrWhiteSpace(hint)) return ToolCallFormat.Hermes;
        // A template carries its own literal markers, which beat any family name it happens to mention.
        if (Has(hint, "<tool_call>")) return ToolCallFormat.Hermes;
        if (Has(hint, "<|python_tag|>")) return ToolCallFormat.Llama3;
        if (Has(hint, "[TOOL_CALLS]")) return ToolCallFormat.Mistral;
        if (Has(hint, "<|tool_call>")) return ToolCallFormat.Gemma;
        if (Has(hint, "hermes") || Has(hint, "qwen") || Has(hint, "glm") || Has(hint, "deepseek")) return ToolCallFormat.Hermes;
        if (Has(hint, "mistral") || Has(hint, "mixtral") || Has(hint, "ministral") || Has(hint, "magistral")
            || Has(hint, "devstral") || Has(hint, "codestral"))
            return ToolCallFormat.Mistral;
        if (Has(hint, "gemma")) return ToolCallFormat.Gemma;
        if (Has(hint, "llama")) return ToolCallFormat.Llama3;
        return ToolCallFormat.Hermes;
    }

    private static bool Has(string hint, string needle) => hint.Contains(needle, StringComparison.OrdinalIgnoreCase);
}
