using System.Collections.Concurrent;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;

namespace HartsyInference.Voice.Tests.Fakes;

/// <summary>An <see cref="ITextService"/> that replays scripted rounds, one per <see cref="StreamAsync"/> call, and records
/// every request, streamed or not. Token counts are one per whitespace-separated word unless overridden.</summary>
internal sealed class ScriptedTextService : ITextService
{
    private readonly ConcurrentQueue<IReadOnlyList<TextChunk>> _rounds = new();

    /// <summary>Every request received, warm-up generations included, in arrival order.</summary>
    public ConcurrentQueue<TextRequest> Requests { get; } = new();

    public Func<string, int> TokenCounter { get; init; } = CountWords;

    public ScriptedTextService Round(params TextChunk[] chunks)
    {
        _rounds.Enqueue(chunks);
        return this;
    }

    /// <summary>Queues a round that streams <paramref name="text"/> word by word and stops normally.</summary>
    public ScriptedTextService Reply(string text)
    {
        List<TextChunk> chunks = [];
        foreach (string piece in SplitKeepingSpaces(text))
        {
            chunks.Add(Text(piece));
        }
        chunks.Add(new TextChunk { Kind = TextChunkKind.Result, Text = text });
        chunks.Add(Stop(StopReason.Stop));
        return Round([.. chunks]);
    }

    public static TextChunk Text(string text) => new() { Kind = TextChunkKind.Chunk, Text = text };

    public static TextChunk Call(string id, string name, string arguments = "{}") =>
        new() { Kind = TextChunkKind.NativeToolCall, ToolCallIndex = 0, ToolCall = new NativeToolCall { Id = id, Name = name, Arguments = arguments } };

    public static TextChunk Stop(StopReason reason, string? text = null) => new() { Kind = TextChunkKind.StopReason, Stop = reason, Text = text };

    public async IAsyncEnumerable<TextChunk> StreamAsync(ModelSpec spec, TextRequest request, [EnumeratorCancellation] CancellationToken cancel = default)
    {
        Requests.Enqueue(request);
        if (!_rounds.TryDequeue(out IReadOnlyList<TextChunk>? round))
        {
            throw new InvalidOperationException("No scripted round left.");
        }
        foreach (TextChunk chunk in round)
        {
            cancel.ThrowIfCancellationRequested();
            await Task.Yield();
            yield return chunk;
        }
    }

    public Task<TextResult> GenerateAsync(ModelSpec spec, TextRequest request, CancellationToken cancel = default)
    {
        Requests.Enqueue(request);
        return Task.FromResult(new TextResult { Text = "" });
    }

    public int CountTokens(ModelSpec spec, string text) => TokenCounter(text);

    public bool Unload(string? device = null) => false;

    private static int CountWords(string text) => text.Split(' ', StringSplitOptions.RemoveEmptyEntries).Length;

    private static IEnumerable<string> SplitKeepingSpaces(string text)
    {
        int start = 0;
        for (int i = 1; i <= text.Length; i++)
        {
            if (i == text.Length || text[i] == ' ')
            {
                yield return text[start..i];
                start = i;
            }
        }
    }
}
