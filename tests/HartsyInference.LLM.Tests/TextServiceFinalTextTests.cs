using HartsyInference.Engine.Services;
using HartsyInference.LLM.OutputParsing;
using HartsyInference.LLM.Tests.OutputParsing;
using Xunit;

namespace HartsyInference.LLM.Tests;

/// <summary>The terminal Result of a streamed request equals the concatenated chunks, including the reasoning tags the passthrough parser shows.</summary>
public sealed class TextServiceFinalTextTests
{
    [Fact]
    public void PassthroughResultKeepsTheStreamedReasoningTags()
    {
        PieceTokenizer tok = new();
        int think = tok.SpecialIdOf(PieceTokenizer.Think);
        int thinkEnd = tok.SpecialIdOf(PieceTokenizer.ThinkEnd);
        int[] ids = [think, .. tok.SplitBytes("plan", 2), thinkEnd, .. tok.SplitBytes("answer", 3)];
        PassthroughOutputParser parser = new(tok, new HashSet<int> { think, thinkEnd });
        System.Text.StringBuilder streamed = new();
        void Sink(ParsedEvent e) { if (e.Kind == ParsedEventKind.ContentDelta) streamed.Append(e.Text); }
        foreach (int id in ids) parser.Push(id, Sink);
        parser.Finish(Sink);

        string decoded = tok.Decode(ids);
        Assert.Equal("plananswer", decoded);
        string final = TextService.FinalText(parser, callCount: 0, decoded);
        Assert.Equal("<think>plan</think>answer", final);
        Assert.Equal(streamed.ToString(), final);
    }

    [Fact]
    public void NoParserKeepsTheDecodedText()
    {
        Assert.Equal("x", TextService.FinalText(null, callCount: 0, "x"));
    }
}
