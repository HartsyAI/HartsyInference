namespace HartsyInference.Tools.Parsing;

/// <summary>The wire format a model family uses to emit a tool call in its decoded text. <see cref="ToolCallFormats.RulesFor"/> maps each value to its markers and payload grammar; <see cref="ToolCallFormats.Detect"/> picks one from a model id or template hint.</summary>
public enum ToolCallFormat
{
    /// <summary>Hermes / Qwen: <c>&lt;tool_call&gt;{"name": …, "arguments": {…}}&lt;/tool_call&gt;</c>. The closing tag is optional (the call ends at the balanced object), and the tags are often special tokens the detokenizer drops, so a bare <c>{"name": …}</c> object also counts.</summary>
    Hermes,

    /// <summary>Llama 3.x: <c>&lt;|python_tag|&gt;</c> followed by JSON, or the bare <c>{"name": …, "parameters": {…}}</c> object the instruct models emit for user-defined tools.</summary>
    Llama3,

    /// <summary>Gemma 4 function-call block: <c>&lt;|tool_call&gt;call:name{key:value,text:&lt;|"|&gt;string&lt;|"|&gt;}&lt;tool_call|&gt;</c>, with unquoted keys and <c>&lt;|"|&gt;</c> string delimiters; the three markers are user-defined tokens the detokenizer may drop.</summary>
    Gemma,

    /// <summary>Mistral: <c>[TOOL_CALLS][{"name": …, "arguments": {…}}, …]</c>, a single object, or the <c>name{…}</c> form of Mistral Small 3.x.</summary>
    Mistral,
}
