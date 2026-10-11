namespace HartsyInference.Tools.Parsing;

/// <summary>What follows a <see cref="ToolCallMarker"/>: the grammar the parser reads the call span with.</summary>
public enum ToolCallPayload
{
    /// <summary>One JSON object carrying <c>name</c> and the arguments.</summary>
    JsonObject,

    /// <summary>A JSON array of call objects.</summary>
    JsonArray,

    /// <summary>A JSON array, a JSON object, or an identifier immediately followed by a JSON arguments object (<c>name{…}</c>).</summary>
    JsonAny,

    /// <summary>Gemma's <c>[call:]name{key:value,…}</c> block with unquoted keys and <c>&lt;|"|&gt;</c> string delimiters.</summary>
    GemmaCall,

    /// <summary>Qwen's <c>&lt;function=NAME&gt;&lt;parameter=KEY&gt;value&lt;/parameter&gt;&lt;/function&gt;</c> markup; the span completes at its closing marker, not at a balanced value.</summary>
    XmlFunction,

    /// <summary>GLM's <c>name&lt;arg_key&gt;k&lt;/arg_key&gt;&lt;arg_value&gt;v&lt;/arg_value&gt;</c> pairs; completes at its closing marker.</summary>
    XmlArgKey,

    /// <summary>DeepSeek-R1's fenced-JSON call blocks; completes at its closing marker.</summary>
    DeepSeekR1Block,
}
