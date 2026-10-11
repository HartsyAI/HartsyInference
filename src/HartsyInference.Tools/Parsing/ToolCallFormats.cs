using System.Text.RegularExpressions;

namespace HartsyInference.Tools.Parsing;

/// <summary>The per-format rule tables, the model-id / template-text heuristic (<see cref="Detect"/>) and the
/// template-content detector (<see cref="TryDetectFromTemplate"/>) that pick a <see cref="ToolCallFormat"/>.
/// Hermes is the default: it is what Qwen and the Hermes fine-tunes emit, and its bare-object rule also reads
/// the plain <c>{"name": …}</c> object most other instruct models fall back to — including families (GLM,
/// DeepSeek) whose real wire format these rules don't actually match; <see cref="Detect"/> falls back to it
/// for anything unrecognized because it is the most permissive guess, not because it is correct for every
/// family. Prefer <see cref="TryDetectFromTemplate"/> over a name guess whenever the model's own chat
/// template is available.</summary>
public static partial class ToolCallFormats
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
        ExtraLiterals = ["<|\"|>"],
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

    /// <summary>Picks a format from a model id, checkpoint name or the model's own chat template text; anything
    /// unrecognised falls through to the unconditional <see cref="ToolCallFormat.Hermes"/> return below — a
    /// permissive default, not a claim that the family actually emits Hermes. Prefer
    /// <see cref="TryDetectFromTemplate"/> over this name guess whenever the real chat template is available.</summary>
    public static ToolCallFormat Detect(string? hint)
    {
        if (string.IsNullOrWhiteSpace(hint)) return ToolCallFormat.Hermes;
        // A template carries its own literal markers, which beat any family name it happens to mention.
        if (Has(hint, "<tool_call>")) return ToolCallFormat.Hermes;
        if (Has(hint, "<|python_tag|>")) return ToolCallFormat.Llama3;
        if (Has(hint, "[TOOL_CALLS]")) return ToolCallFormat.Mistral;
        if (Has(hint, "<|tool_call>")) return ToolCallFormat.Gemma;
        if (Has(hint, "hermes") || Has(hint, "qwen")) return ToolCallFormat.Hermes;
        if (Has(hint, "mistral") || Has(hint, "mixtral") || Has(hint, "ministral") || Has(hint, "magistral")
            || Has(hint, "devstral") || Has(hint, "codestral"))
            return ToolCallFormat.Mistral;
        if (Has(hint, "gemma")) return ToolCallFormat.Gemma;
        if (Has(hint, "llama")) return ToolCallFormat.Llama3;
        // GLM and DeepSeek land here too: their real formats are XML-argument and marker-based respectively,
        // not Hermes (see TryDetectFromTemplate), but neither has a rule table of its own yet, and Hermes'
        // bare-object rule is the most permissive fallback among the four we do support.
        return ToolCallFormat.Hermes;
    }

    private static bool Has(string hint, string needle) => hint.Contains(needle, StringComparison.OrdinalIgnoreCase);

    /// <summary>Reads the markers a model's own GGUF <c>tokenizer.chat_template</c> actually renders, instead of
    /// guessing from a name. The general-purpose equivalent of the heuristic the SwarmUI LLMAssistant extension's
    /// <c>HartsyLocalLLMProvider</c> applies locally (read the template, require it to reference <c>tools</c> and
    /// instruct Hermes JSON). Normalizes <c>\"</c> → <c>"</c> first: chat templates are Jinja string literals, so
    /// a JSON example embedded in one (e.g. Qwen's <c>"...\"name\": ...\"arguments\": ..."</c>) carries that escape
    /// in the raw GGUF metadata string. Returns <see langword="true"/> only when the template references the
    /// word <c>tools</c> (a plain substring match — prose that merely mentions "tools" passes this half, but
    /// still needs a literal envelope marker below it to return true) AND literally instructs one of four
    /// supported envelopes: Hermes JSON
    /// (<c>&lt;tool_call&gt;</c> followed by a JSON object naming <c>"name"</c> and <c>"arguments"</c>), Llama-3
    /// (<c>&lt;|python_tag|&gt;</c>), Mistral (<c>[TOOL_CALLS]</c>) or Gemma (<c>&lt;|tool_call&gt;</c>). An
    /// XML-argument dialect (GLM's <c>&lt;tool_call&gt;name\n&lt;arg_key&gt;…</c>, Qwen3.5/Qwen3-Coder's
    /// <c>&lt;tool_call&gt;&lt;function=name&gt;&lt;parameter=…&gt;</c>), DeepSeek-R1's
    /// <c>&lt;｜tool▁calls▁begin｜&gt;</c> markers, a template that never renders <c>tools</c>, or no template at
    /// all all return <see langword="false"/> — none of them has a rule table to detect into.</summary>
    public static bool TryDetectFromTemplate(string? chatTemplate, out ToolCallFormat format)
    {
        format = ToolCallFormat.Hermes;
        if (string.IsNullOrEmpty(chatTemplate)) return false;
        string t = chatTemplate.Replace("\\\"", "\"", StringComparison.Ordinal);
        if (!ToolsVariableRegex().IsMatch(t)) return false;
        if (HasHermesJsonInstruction(t)) { format = ToolCallFormat.Hermes; return true; }
        if (t.Contains("<|python_tag|>", StringComparison.Ordinal)) { format = ToolCallFormat.Llama3; return true; }
        if (t.Contains("[TOOL_CALLS]", StringComparison.Ordinal)) { format = ToolCallFormat.Mistral; return true; }
        if (t.Contains("<|tool_call>", StringComparison.Ordinal)) { format = ToolCallFormat.Gemma; return true; }
        return false;
    }

    /// <summary>Scan window (characters) after an unclosed <c>&lt;tool_call&gt;</c> marker — generous for the
    /// real instruction blocks this reads (a few hundred characters), small enough to stay local to one call.</summary>
    private const int HermesInstructionWindow = 300;

    /// <summary>True when a <c>&lt;tool_call&gt;</c> marker is followed — before its matching
    /// <c>&lt;/tool_call&gt;</c>, or within <see cref="HermesInstructionWindow"/> characters when there is no
    /// close marker — by <c>"name"</c> and then <c>"arguments"</c>, in order. That ordered pair is what
    /// distinguishes the Hermes JSON envelope from the XML-argument dialects, which open with the identical
    /// <c>&lt;tool_call&gt;</c> tag but never render either literal key.</summary>
    private static bool HasHermesJsonInstruction(string t)
    {
        int from = 0;
        int tagIdx;
        while ((tagIdx = t.IndexOf("<tool_call>", from, StringComparison.Ordinal)) >= 0)
        {
            int closeIdx = t.IndexOf("</tool_call>", tagIdx, StringComparison.Ordinal);
            int bound = closeIdx >= 0 ? closeIdx : Math.Min(t.Length, tagIdx + HermesInstructionWindow);
            int nameIdx = t.IndexOf("\"name\"", tagIdx, StringComparison.Ordinal);
            if (nameIdx >= 0 && nameIdx < bound)
            {
                int argsIdx = t.IndexOf("\"arguments\"", nameIdx, StringComparison.Ordinal);
                if (argsIdx >= 0 && argsIdx < bound) return true;
            }
            from = tagIdx + 1;
        }
        return false;
    }

    /// <summary>Matches the Jinja <c>tools</c> variable as a whole identifier, not a prefix/suffix of a longer
    /// one (Llama-3.2's own template defines <c>tools_in_user_message</c>, which is not the tools list itself).</summary>
    [GeneratedRegex(@"\btools\b")]
    private static partial Regex ToolsVariableRegex();
}
