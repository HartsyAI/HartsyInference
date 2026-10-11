using System.Text.Json;
using HartsyInference.Core.Logging;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Tools;

/// <summary>The tools a host offers to the model: handlers keyed by name, their <see cref="ToolDefinition"/>s for <see cref="TextRequest.Tools"/>, and dispatch of a <see cref="NativeToolCall"/>. Dispatch never throws for an ordinary failure: an unknown tool, a throwing handler or a handler that outlives its timeout yields an <c>{"error": …}</c> result the model can read and recover from; cancellation of the caller's token propagates.</summary>
/// <remarks>Thread-safe: <see cref="Add(IToolHandler)"/>, <see cref="Remove"/>, lookup and enumeration may run concurrently. Writers swap in a new immutable snapshot, so a <see cref="Definitions"/> list or <see cref="Names"/> sequence already handed out never changes underneath a request in flight. Handlers have no timeout unless <see cref="DefaultTimeout"/> or a per-tool timeout is set.</remarks>
public sealed class ToolRegistry : IToolDispatcher
{
    private readonly object _writeLock = new();
    private volatile Snapshot _snapshot = Snapshot.Empty;
    private long _defaultTimeoutTicks;

    /// <summary>Registered tool count.</summary>
    public int Count => _snapshot.Handlers.Count;

    /// <summary>Definitions in registration order, ready for <see cref="TextRequest.Tools"/>. The returned list is an immutable snapshot: later <c>Add</c>/<c>Remove</c> calls do not change it.</summary>
    public IReadOnlyList<ToolDefinition> Definitions => _snapshot.Definitions;

    /// <summary>Registered names in registration order (a snapshot, like <see cref="Definitions"/>).</summary>
    public IEnumerable<string> Names => _snapshot.Names;

    /// <summary>Timeout applied to every tool that has no per-tool timeout; null (the default) means handlers run until they finish or the caller's token is cancelled. Must be positive when set.</summary>
    public TimeSpan? DefaultTimeout
    {
        get
        {
            long ticks = Interlocked.Read(ref _defaultTimeoutTicks);
            return ticks == 0 ? null : TimeSpan.FromTicks(ticks);
        }
        set
        {
            if (value is { } timeout) ValidateTimeout(timeout);
            Interlocked.Exchange(ref _defaultTimeoutTicks, value?.Ticks ?? 0);
        }
    }

    /// <summary>Registers <paramref name="handler"/>; a duplicate name is an error.</summary>
    public ToolRegistry Add(IToolHandler handler) => Add(handler, null);

    /// <summary>Registers <paramref name="handler"/> with a per-tool <paramref name="timeout"/> that overrides <see cref="DefaultTimeout"/>; null uses the default. On expiry the handler's token is cancelled and the call yields an error result.</summary>
    public ToolRegistry Add(IToolHandler handler, TimeSpan? timeout)
    {
        ArgumentNullException.ThrowIfNull(handler);
        ArgumentException.ThrowIfNullOrWhiteSpace(handler.Name);
        if (timeout is { } limit) ValidateTimeout(limit);
        ToolDefinition definition = handler is DelegateToolHandler bound ? bound.Definition : new ToolDefinition
        {
            Name = handler.Name,
            Description = handler.Description ?? "",
            JsonSchema = string.IsNullOrWhiteSpace(handler.JsonSchema) ? "{}" : handler.JsonSchema,
        };
        lock (_writeLock)
        {
            Snapshot current = _snapshot;
            if (current.Handlers.ContainsKey(handler.Name))
                throw new ArgumentException($"A tool named '{handler.Name}' is already registered.", nameof(handler));
            Dictionary<string, Entry> handlers = new(current.Handlers, StringComparer.Ordinal) { [handler.Name] = new Entry(handler, timeout) };
            _snapshot = new Snapshot(handlers, [.. current.DefinitionArray, definition]);
        }
        return this;
    }

    /// <summary>Registers a tool from its definition parts and an invoke function taking the raw JSON arguments.</summary>
    public ToolRegistry Add(string name, string description, string jsonSchema, Func<string, CancellationToken, Task<string>> invoke)
        => Add(new FunctionToolHandler(name, description, jsonSchema, invoke), null);

    /// <summary>Registers a tool from its definition parts and an invoke function, with a per-tool timeout (see <see cref="Add(IToolHandler, TimeSpan?)"/>).</summary>
    public ToolRegistry Add(string name, string description, string jsonSchema, Func<string, CancellationToken, Task<string>> invoke, TimeSpan? timeout)
        => Add(new FunctionToolHandler(name, description, jsonSchema, invoke), timeout);

    /// <summary>Registers a C# delegate as a tool; its parameters become the JSON schema (see <see cref="ToolSchema.FromDelegate"/>) and the model's arguments are bound to them by name.</summary>
    public ToolRegistry Add(string name, Delegate method, string? description = null)
        => Add(new DelegateToolHandler(name, method, description), null);

    /// <summary>Registers a C# delegate as a tool with a per-tool timeout (see <see cref="Add(IToolHandler, TimeSpan?)"/>).</summary>
    public ToolRegistry Add(string name, Delegate method, string? description, TimeSpan? timeout)
        => Add(new DelegateToolHandler(name, method, description), timeout);

    /// <summary>Unregisters the tool named <paramref name="name"/>; false when there is none. Calls already running keep their handler; the remaining tools keep their order.</summary>
    public bool Remove(string name)
    {
        ArgumentNullException.ThrowIfNull(name);
        lock (_writeLock)
        {
            Snapshot current = _snapshot;
            if (!current.Handlers.ContainsKey(name)) return false;
            Dictionary<string, Entry> handlers = new(current.Handlers, StringComparer.Ordinal);
            handlers.Remove(name);
            _snapshot = new Snapshot(handlers, current.DefinitionArray.Where(d => d.Name != name).ToArray());
            return true;
        }
    }

    /// <summary>Looks up a handler by exact name.</summary>
    public bool TryGet(string name, out IToolHandler handler)
    {
        if (_snapshot.Handlers.TryGetValue(name, out Entry? entry))
        {
            handler = entry.Handler;
            return true;
        }
        handler = null!;
        return false;
    }

    /// <summary>Runs the handler for <paramref name="call"/> and returns the tool result text; unknown tools, handler failures and timeouts come back as an error result carrying the exception type and message (or the timeout), which is meant for the model and the host, not for a remote end user. Cancelling <paramref name="cancel"/> still throws.</summary>
    public async Task<string> InvokeAsync(NativeToolCall call, CancellationToken cancel = default)
    {
        ArgumentNullException.ThrowIfNull(call);
        Snapshot snapshot = _snapshot;
        if (!snapshot.Handlers.TryGetValue(call.Name, out Entry? entry))
            return ErrorJson($"Unknown tool '{call.Name}'. Available tools: {string.Join(", ", snapshot.Names)}.");
        IToolHandler handler = entry.Handler;
        TimeSpan? timeout = entry.Timeout ?? DefaultTimeout;
        try
        {
            return timeout is { } limit
                ? await InvokeWithTimeoutAsync(handler, call.Arguments ?? "{}", limit, cancel).ConfigureAwait(false)
                : await handler.InvokeAsync(call.Arguments ?? "{}", cancel).ConfigureAwait(false);
        }
        catch (OperationCanceledException) when (cancel.IsCancellationRequested)
        {
            throw;
        }
        catch (ToolTimeoutException)
        {
            Logs.Error($"Tool '{call.Name}' timed out after {timeout!.Value.TotalSeconds:0.###}s.");
            return ErrorJson($"Tool '{call.Name}' timed out after {timeout.Value.TotalSeconds:0.###} seconds.");
        }
        catch (Exception ex)
        {
            Logs.Error($"Tool '{call.Name}' failed: {ex.Message}", ex);
            return ErrorJson($"{ex.GetType().Name}: {ex.Message}");
        }
    }

    /// <summary>Runs <paramref name="handler"/> off the caller's thread with a token that fires on timeout or caller cancellation, and stops waiting on either even if the handler ignores its token. Throws <see cref="ToolTimeoutException"/> on expiry; a handler that outlives it is abandoned with its fault observed.</summary>
    private static async Task<string> InvokeWithTimeoutAsync(IToolHandler handler, string argumentsJson, TimeSpan timeout, CancellationToken cancel)
    {
        using CancellationTokenSource timeoutSource = new(timeout);
        using CancellationTokenSource linked = CancellationTokenSource.CreateLinkedTokenSource(cancel, timeoutSource.Token);
        // An abandoned handler may outlive the disposal below; the timeout already cancelled its token, so that is harmless.
        Task<string> running = Task.Run(() => handler.InvokeAsync(argumentsJson, linked.Token), CancellationToken.None);
        try
        {
            return await running.WaitAsync(linked.Token).ConfigureAwait(false);
        }
        catch (OperationCanceledException) when (!cancel.IsCancellationRequested && timeoutSource.IsCancellationRequested)
        {
            _ = running.ContinueWith(static t => _ = t.Exception, CancellationToken.None, TaskContinuationOptions.OnlyOnFaulted, TaskScheduler.Default);
            throw new ToolTimeoutException();
        }
    }

    private static void ValidateTimeout(TimeSpan timeout)
    {
        if (timeout <= TimeSpan.Zero || timeout > TimeSpan.FromMilliseconds(uint.MaxValue - 1))
            throw new ArgumentOutOfRangeException(nameof(timeout), timeout, "A tool timeout must be positive and finite.");
    }

    /// <summary>An <c>{"error": message}</c> JSON result.</summary>
    public static string ErrorJson(string message) => JsonText.Write(writer =>
    {
        writer.WriteStartObject();
        writer.WriteString("error", message);
        writer.WriteEndObject();
    });

    private sealed class ToolTimeoutException : Exception;

    private sealed record Entry(IToolHandler Handler, TimeSpan? Timeout);

    /// <summary>One immutable registry state: handlers by name plus the definitions in insertion order.</summary>
    private sealed class Snapshot
    {
        public static readonly Snapshot Empty = new(new Dictionary<string, Entry>(StringComparer.Ordinal), []);

        public Snapshot(Dictionary<string, Entry> handlers, ToolDefinition[] definitions)
        {
            Handlers = handlers;
            DefinitionArray = definitions;
            Definitions = Array.AsReadOnly(definitions);
            Names = Array.AsReadOnly(definitions.Select(d => d.Name).ToArray());
        }

        public Dictionary<string, Entry> Handlers { get; }

        public ToolDefinition[] DefinitionArray { get; }

        public IReadOnlyList<ToolDefinition> Definitions { get; }

        public IReadOnlyList<string> Names { get; }
    }

    private sealed class FunctionToolHandler(string name, string description, string jsonSchema, Func<string, CancellationToken, Task<string>> invoke) : IToolHandler
    {
        public string Name { get; } = name;

        public string Description { get; } = description;

        public string JsonSchema { get; } = jsonSchema;

        public Task<string> InvokeAsync(string argumentsJson, CancellationToken cancel) => invoke(argumentsJson, cancel);
    }
}
