namespace HartsyInference.LLM.OutputParsing;

/// <summary>Which part of an assistant turn a parser is currently inside.</summary>
public enum OutputParserSection
{
    /// <summary>Inside the think block; the next think-end closes it.</summary>
    Reasoning,

    /// <summary>Plain answer text.</summary>
    Content,

    /// <summary>Inside a tool-calls block.</summary>
    ToolCalls,

    /// <summary>The turn has ended; further input is ignored.</summary>
    Done,
}
