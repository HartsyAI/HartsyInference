namespace HartsyInference.Tools;

/// <summary>A callable tool: the definition the model sees (name, description, JSON schema of the arguments object) and the code that runs when the model calls it.</summary>
public interface IToolHandler
{
    /// <summary>The tool name the model calls, unique within a <see cref="ToolRegistry"/>.</summary>
    string Name { get; }

    /// <summary>What the tool does, rendered into the prompt for the model.</summary>
    string Description { get; }

    /// <summary>JSON-schema text for the arguments object.</summary>
    string JsonSchema { get; }

    /// <summary>Runs the tool with the model's raw JSON arguments and returns the text fed back to the model as the tool result.</summary>
    Task<string> InvokeAsync(string argumentsJson, CancellationToken cancel);
}
