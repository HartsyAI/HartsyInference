using HartsyInference.Engine.Requests;

namespace HartsyInference.Tools;

/// <summary>What a tool loop needs from its host: the definitions to offer the model and the code that runs a call. <see cref="ToolRegistry"/> is one implementation; a host with its own executor (permissions, rate limits, audit) implements this and keeps the loop.</summary>
public interface IToolDispatcher
{
    /// <summary>The definitions offered when a request names none of its own.</summary>
    IReadOnlyList<ToolDefinition> Definitions { get; }

    /// <summary>Runs one call and returns the result text fed back to the model.</summary>
    Task<string> InvokeAsync(NativeToolCall call, CancellationToken cancel);
}
