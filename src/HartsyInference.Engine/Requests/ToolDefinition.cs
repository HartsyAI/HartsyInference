namespace HartsyInference.Engine.Requests;

/// <summary>A tool the model may call: a name, a description, and a JSON-schema string for its arguments. Drives the
/// grammar-constrained tool-call decoding path.</summary>
public sealed record ToolDefinition
{
    /// <summary>The tool name.</summary>
    public required string Name { get; init; }

    /// <summary>Human/model-readable description of what the tool does.</summary>
    public string Description { get; init; } = "";

    /// <summary>JSON-schema string describing the tool's argument object.</summary>
    public string JsonSchema { get; init; } = "{}";
}
