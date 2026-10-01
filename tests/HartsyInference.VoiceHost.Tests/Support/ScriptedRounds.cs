using System.Collections.Concurrent;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;

namespace HartsyInference.VoiceHost.Tests.Support;

/// <summary>A language model that replays scripted rounds, one per streamed request, and records each request; with no
/// round left it streams <see cref="DefaultReply"/>. One-shot requests (the warm-up) answer with nothing.</summary>
internal sealed class ScriptedRounds : ITextService
{
    private readonly ConcurrentQueue<IReadOnlyList<TextChunk>> _rounds = new();

    public ConcurrentQueue<TextRequest> Requests { get; } = new();

    public string DefaultReply { get; init; } = "Okay.";

    /// <summary>Queues a round that streams <paramref name="text"/> word by word and stops normally.</summary>
    public ScriptedRounds Reply(string text)
    {
        _rounds.Enqueue([.. Words(text), new TextChunk { Kind = TextChunkKind.Result, Text = text }, Stop(StopReason.Stop)]);
        return this;
    }

    /// <summary>Queues a round that says <paramref name="text"/> and then calls <paramref name="tool"/>.</summary>
    public ScriptedRounds ReplyThenCall(string text, string tool, string arguments = "{}")
    {
        _rounds.Enqueue(
        [
            .. Words(text),
            new TextChunk { Kind = TextChunkKind.NativeToolCall, ToolCallIndex = 0, ToolCall = new NativeToolCall { Id = "call_0", Name = tool, Arguments = arguments } },
            Stop(StopReason.ToolCall),
        ]);
        return this;
    }

    public async IAsyncEnumerable<TextChunk> StreamAsync(ModelSpec spec, TextRequest request, [EnumeratorCancellation] CancellationToken cancel = default)
    {
        Requests.Enqueue(request);
        if (!_rounds.TryDequeue(out IReadOnlyList<TextChunk>? round))
        {
            round = [.. Words(DefaultReply), new TextChunk { Kind = TextChunkKind.Result, Text = DefaultReply }, Stop(StopReason.Stop)];
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

    public int CountTokens(ModelSpec spec, string text) => text.Split(' ', StringSplitOptions.RemoveEmptyEntries).Length;

    public bool Unload(string? device = null) => false;

    private static TextChunk Stop(StopReason reason) => new() { Kind = TextChunkKind.StopReason, Stop = reason };

    private static IEnumerable<TextChunk> Words(string text)
    {
        int start = 0;
        for (int i = 1; i <= text.Length; i++)
        {
            if (i == text.Length || text[i] == ' ')
            {
                yield return new TextChunk { Kind = TextChunkKind.Chunk, Text = text[start..i] };
                start = i;
            }
        }
    }
}
