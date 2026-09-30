using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.LLM.OutputParsing;
using Xunit;

namespace HartsyInference.LLM.Tests.OutputParsing;

/// <summary>Golden chunk sequence for a normal completion through the same wiring <c>TextService.StreamAsync</c> uses (parser → translator → sink → pump → Result + StopReason): the lifetime rewrite must not add, drop or reorder anything.</summary>
public sealed class StreamChunkSequenceTests
{
    [Fact]
    public async Task NormalCompletionYieldsContentChunksThenResultThenStop()
    {
        PieceTokenizer tok = new();
        // Byte offsets 5, 8, 9, 14 and 17 cut "ö" and "😀" across tokens, so held-back partial characters are part of the sequence.
        int[] ids = tok.SplitBytes("Hello wörld 😀!", 5, 8, 9, 14, 17);
        Task<IReadOnlyList<TextChunk>> Produce(Action<TextChunk> sink, CancellationToken ct)
        {
            PassthroughOutputParser parser = new(tok);
            ParsedEventTranslator translator = new(sink, 1);
            foreach (int id in ids) parser.Push(id, translator.Handle);
            parser.Finish(translator.Handle);
            return Task.FromResult<IReadOnlyList<TextChunk>>(
            [
                new TextChunk { Kind = TextChunkKind.Result, Text = parser.Result.Content },
                new TextChunk { Kind = TextChunkKind.StopReason, Stop = StopReason.Stop },
            ]);
        }
        List<(TextChunkKind Kind, string? Text, StopReason? Stop)> got = [];
        await foreach (TextChunk c in TextStreamPump.Run(Produce)) got.Add((c.Kind, c.Text, c.Stop));
        List<(TextChunkKind Kind, string? Text, StopReason? Stop)> expected =
        [
            (TextChunkKind.Chunk, "Hello", null),
            (TextChunkKind.Chunk, " w", null),
            (TextChunkKind.Chunk, "ö", null),
            (TextChunkKind.Chunk, "rld ", null),
            (TextChunkKind.Chunk, "😀", null),
            (TextChunkKind.Chunk, "!", null),
            (TextChunkKind.Result, "Hello wörld 😀!", null),
            (TextChunkKind.StopReason, null, StopReason.Stop),
        ];
        Assert.Equal(expected, got);
    }
}
