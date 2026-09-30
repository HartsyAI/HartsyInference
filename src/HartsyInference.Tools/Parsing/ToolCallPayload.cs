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
}
