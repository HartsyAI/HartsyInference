using System.Runtime.CompilerServices;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;

namespace HartsyInference.Tools.Tests;

/// <summary>An <see cref="ITextService"/> that replays scripted chunk sequences, one per <see cref="StreamAsync"/> call, and records every request it received.</summary>
internal sealed class ScriptedTextService : ITextService
{
    private readonly Queue<IReadOnlyList<TextChunk>> _rounds = new();

    public List<TextRequest> Requests { get; } = [];

    public ScriptedTextService Round(params TextChunk[] chunks)
    {
        _rounds.Enqueue(chunks);
        return this;
    }

    public static TextChunk Text(string text) => new() { Kind = TextChunkKind.Chunk, Text = text };

    public static TextChunk Call(string id, string name, string arguments = "{}")
        => new() { Kind = TextChunkKind.NativeToolCall, ToolCallIndex = 0, ToolCall = new NativeToolCall { Id = id, Name = name, Arguments = arguments } };

    public static TextChunk Result(string text) => new() { Kind = TextChunkKind.Result, Text = text };

    public static TextChunk Stop(StopReason reason, string? text = null) => new() { Kind = TextChunkKind.StopReason, Stop = reason, Text = text };

    public async IAsyncEnumerable<TextChunk> StreamAsync(ModelSpec spec, TextRequest request, [EnumeratorCancellation] CancellationToken cancel = default)
    {
        Requests.Add(request);
        if (_rounds.Count == 0) throw new InvalidOperationException("No scripted round left.");
        foreach (TextChunk chunk in _rounds.Dequeue())
        {
            cancel.ThrowIfCancellationRequested();
            await Task.Yield();
            yield return chunk;
        }
    }

    public Task<TextResult> GenerateAsync(ModelSpec spec, TextRequest request, CancellationToken cancel = default) => throw new NotSupportedException();

    public int CountTokens(ModelSpec spec, string text) => text.Length;

    public bool Unload(string? device = null) => false;
}
