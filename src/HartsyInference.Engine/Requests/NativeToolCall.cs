namespace HartsyInference.Engine.Requests;

/// <summary>A tool call the model emitted: an id, the tool name, and the raw JSON arguments.</summary>
public sealed record NativeToolCall
{
    /// <summary>Call id (for correlating the eventual tool result).</summary>
    public string Id { get; init; } = "";

    /// <summary>The invoked tool's name.</summary>
    public required string Name { get; init; }

    /// <summary>Raw JSON arguments string.</summary>
    public string Arguments { get; init; } = "{}";
}
