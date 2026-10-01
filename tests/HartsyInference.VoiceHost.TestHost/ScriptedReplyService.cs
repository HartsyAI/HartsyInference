using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;

namespace HartsyInference.VoiceHost.TestHost;

/// <summary>A language model that answers every streamed request with the same reply, word by word, and every one-shot
/// request with nothing; it loads nothing and touches no device.</summary>
public sealed class ScriptedReplyService(string reply) : ITextService
{
    public Task<TextResult> GenerateAsync(ModelSpec spec, TextRequest request, CancellationToken cancel = default) =>
        Task.FromResult(new TextResult { Text = "" });

    public async IAsyncEnumerable<TextChunk> StreamAsync(ModelSpec spec, TextRequest request, [EnumeratorCancellation] CancellationToken cancel = default)
    {
        int start = 0;
        for (int i = 1; i <= reply.Length; i++)
        {
            if (i == reply.Length || reply[i] == ' ')
            {
                cancel.ThrowIfCancellationRequested();
                await Task.Yield();
                yield return new TextChunk { Kind = TextChunkKind.Chunk, Text = reply[start..i] };
                start = i;
            }
        }
        yield return new TextChunk { Kind = TextChunkKind.Result, Text = reply };
        yield return new TextChunk { Kind = TextChunkKind.StopReason, Stop = StopReason.Stop };
    }

    public int CountTokens(ModelSpec spec, string text) => text.Split(' ', StringSplitOptions.RemoveEmptyEntries).Length;

    public bool Unload(string? device = null) => false;
}
