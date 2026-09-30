using System.Text;
using System.Text.Json;
using HartsyInference.Core.Logging;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Tools;

/// <summary>The tools a host offers to the model: handlers keyed by name, their <see cref="ToolDefinition"/>s for <see cref="TextRequest.Tools"/>, and dispatch of a <see cref="NativeToolCall"/>. Dispatch never throws for an ordinary failure: an unknown tool or a throwing handler yields an <c>{"error": …}</c> result the model can read and recover from; cancellation propagates.</summary>
public sealed class ToolRegistry
{
    private readonly Dictionary<string, IToolHandler> _handlers = new(StringComparer.Ordinal);
    private readonly List<ToolDefinition> _definitions = [];

    /// <summary>Registered tool count.</summary>
    public int Count => _handlers.Count;

    /// <summary>Definitions in registration order, ready for <see cref="TextRequest.Tools"/>.</summary>
    public IReadOnlyList<ToolDefinition> Definitions => _definitions;

    /// <summary>Registered names in registration order.</summary>
    public IEnumerable<string> Names => _definitions.Select(d => d.Name);

    /// <summary>Registers <paramref name="handler"/>; a duplicate name is an error.</summary>
    public ToolRegistry Add(IToolHandler handler)
    {
        ArgumentNullException.ThrowIfNull(handler);
        ArgumentException.ThrowIfNullOrWhiteSpace(handler.Name);
        if (!_handlers.TryAdd(handler.Name, handler))
            throw new ArgumentException($"A tool named '{handler.Name}' is already registered.", nameof(handler));
        _definitions.Add(handler is DelegateToolHandler bound ? bound.Definition : new ToolDefinition
        {
            Name = handler.Name,
            Description = handler.Description ?? "",
            JsonSchema = string.IsNullOrWhiteSpace(handler.JsonSchema) ? "{}" : handler.JsonSchema,
        });
        return this;
    }

    /// <summary>Registers a tool from its definition parts and an invoke function taking the raw JSON arguments.</summary>
    public ToolRegistry Add(string name, string description, string jsonSchema, Func<string, CancellationToken, Task<string>> invoke)
        => Add(new FunctionToolHandler(name, description, jsonSchema, invoke));

    /// <summary>Registers a C# delegate as a tool; its parameters become the JSON schema (see <see cref="ToolSchema.FromDelegate"/>) and the model's arguments are bound to them by name.</summary>
    public ToolRegistry Add(string name, Delegate method, string? description = null)
        => Add(new DelegateToolHandler(name, method, description));

    /// <summary>Looks up a handler by exact name.</summary>
    public bool TryGet(string name, out IToolHandler handler) => _handlers.TryGetValue(name, out handler!);

    /// <summary>Runs the handler for <paramref name="call"/> and returns the tool result text; unknown tools and handler failures come back as an error result.</summary>
    public async Task<string> InvokeAsync(NativeToolCall call, CancellationToken cancel = default)
    {
        ArgumentNullException.ThrowIfNull(call);
        if (!_handlers.TryGetValue(call.Name, out IToolHandler? handler))
            return ErrorJson($"Unknown tool '{call.Name}'. Available tools: {string.Join(", ", Names)}.");
        try
        {
            return await handler.InvokeAsync(call.Arguments ?? "{}", cancel).ConfigureAwait(false);
        }
        catch (OperationCanceledException) when (cancel.IsCancellationRequested)
        {
            throw;
        }
        catch (Exception ex)
        {
            Logs.Error($"Tool '{call.Name}' failed: {ex.Message}", ex);
            return ErrorJson($"{ex.GetType().Name}: {ex.Message}");
        }
    }

    /// <summary>An <c>{"error": message}</c> JSON result.</summary>
    public static string ErrorJson(string message)
    {
        using MemoryStream buffer = new();
        using (Utf8JsonWriter writer = new(buffer))
        {
            writer.WriteStartObject();
            writer.WriteString("error", message);
            writer.WriteEndObject();
        }
        return Encoding.UTF8.GetString(buffer.GetBuffer(), 0, (int)buffer.Length);
    }

    private sealed class FunctionToolHandler(string name, string description, string jsonSchema, Func<string, CancellationToken, Task<string>> invoke) : IToolHandler
    {
        public string Name { get; } = name;

        public string Description { get; } = description;

        public string JsonSchema { get; } = jsonSchema;

        public Task<string> InvokeAsync(string argumentsJson, CancellationToken cancel) => invoke(argumentsJson, cancel);
    }
}
