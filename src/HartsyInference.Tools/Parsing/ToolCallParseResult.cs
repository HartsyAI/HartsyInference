using HartsyInference.Engine.Requests;

namespace HartsyInference.Tools.Parsing;

/// <summary>What one <see cref="ToolCallParser.Push"/> or <see cref="ToolCallParser.Flush"/> produced: the plain text to forward (empty forwards nothing) and the calls that completed, oldest first (null when none; more than one only when a single delta closed several, as a Mistral array does).</summary>
public readonly record struct ToolCallParseResult(string ForwardText, IReadOnlyList<NativeToolCall>? Calls)
{
    /// <summary>The first completed call, or null.</summary>
    public NativeToolCall? Call => Calls is { Count: > 0 } ? Calls[0] : null;
}
