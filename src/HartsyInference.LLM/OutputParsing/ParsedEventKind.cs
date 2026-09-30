namespace HartsyInference.LLM.OutputParsing;

/// <summary>The event vocabulary an <see cref="IOutputParser"/> streams to its sink.</summary>
public enum ParsedEventKind
{
    /// <summary>Incremental reasoning text (inside the model's think block).</summary>
    ReasoningDelta,

    /// <summary>Incremental user-visible answer text.</summary>
    ContentDelta,

    /// <summary>A tool call opened; <see cref="ParsedEvent.Text"/> is its bare name.</summary>
    ToolCallBegin,

    /// <summary>A fragment of the open tool call's JSON arguments; fragments concatenate to the final arguments.</summary>
    ToolCallArgsDelta,

    /// <summary>The open tool call is complete.</summary>
    ToolCallEnd,

    /// <summary>The open tool call was dropped because the format broke; its argument fragments must be discarded.</summary>
    ToolCallAbort,

    /// <summary>The completion ended (end-of-sequence seen or the stream finished).</summary>
    Stop,

    /// <summary>The completion violated the format; <see cref="ParsedEvent.Text"/> says how. Parsing continues without throwing.</summary>
    Malformed,
}
