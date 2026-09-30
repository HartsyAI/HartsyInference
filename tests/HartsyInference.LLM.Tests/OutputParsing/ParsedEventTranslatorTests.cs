using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.LLM.OutputParsing;
using Xunit;

namespace HartsyInference.LLM.Tests.OutputParsing;

public sealed class ParsedEventTranslatorTests
{
    private static List<TextChunk> Translate(string completion, bool thinking, long requestId)
    {
        PieceTokenizer tok = new();
        DeepSeekV41OutputParser parser = ParserTestHelpers.NewParser(tok, thinking);
        List<TextChunk> chunks = [];
        ParsedEventTranslator translator = new(chunks.Add, requestId);
        foreach (int id in tok.RandomWithSpecials(completion, new Random(9), 4)) parser.Push(id, translator.Handle);
        parser.Finish(translator.Handle);
        return chunks;
    }

    [Fact]
    public void ReasoningContentAndCallsMapToTheirChunkKinds()
    {
        string text = "why</think>ok\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"w::f\">\n" +
            "<｜DSML｜ parameter name=\"a\" string=\"false\">1</｜DSML｜ parameter>\n" +
            "</｜DSML｜ invoke>\n<｜DSML｜ invoke name=\"g\">\n</｜DSML｜ invoke>\n</｜DSML｜ calls>" + PieceTokenizer.Eos;
        List<TextChunk> chunks = Translate(text, true, 42);
        Assert.Equal("why", string.Concat(chunks.Where(c => c.Kind == TextChunkKind.Reasoning).Select(c => c.Text)));
        Assert.Equal("ok", string.Concat(chunks.Where(c => c.Kind == TextChunkKind.Chunk).Select(c => c.Text)));
        List<TextChunk> complete = chunks.Where(c => c.Kind == TextChunkKind.NativeToolCall).ToList();
        Assert.Equal(2, complete.Count);
        NativeToolCall first = complete[0].ToolCall!;
        NativeToolCall second = complete[1].ToolCall!;
        Assert.Equal(("call_42_0", "f", "w", "{\"a\": 1}"), (first.Id, first.Name, first.Namespace, first.Arguments));
        Assert.Equal(("call_42_1", "g", null, "{}"), (second.Id, second.Name, second.Namespace, second.Arguments));
        Assert.Equal([0, 1], complete.Select(c => c.ToolCallIndex!.Value));
        IEnumerable<TextChunk> deltas = chunks.Where(c => c.Kind == TextChunkKind.ToolCallDelta && c.ToolCallIndex == 0);
        Assert.Equal("{\"a\": 1}", string.Concat(deltas.Select(c => c.Text)));
        Assert.Equal("call_42_0", deltas.First().ToolCall!.Id);
    }

    [Fact]
    public void MalformedEventsProduceNoChunks()
    {
        List<TextChunk> chunks = Translate("a ｜DSML｜ b" + PieceTokenizer.Eos, false, 1);
        Assert.All(chunks, c => Assert.Equal(TextChunkKind.Chunk, c.Kind));
        Assert.Equal("a ｜DSML｜ b", string.Concat(chunks.Select(c => c.Text)));
    }

    [Fact]
    public void UnfinishedCallNeverProducesACompleteToolCallChunk()
    {
        const string open = "\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"f\">\n<｜DSML｜ parameter name=\"a\" string=\"true\">x";
        List<TextChunk> chunks = Translate(open, false, 1);
        Assert.DoesNotContain(chunks, c => c.Kind == TextChunkKind.NativeToolCall);
        Assert.Equal(TextChunkKind.ToolCallAbort, chunks[^1].Kind);
        Assert.Equal(0, chunks[^1].ToolCallIndex);
    }
}
