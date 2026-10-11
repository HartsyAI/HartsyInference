using HartsyInference.LLM.OutputParsing;
using Xunit;

namespace HartsyInference.LLM.Tests.OutputParsing;

/// <summary>Control tokens a filter lists as marker literals are decoded as text; every other control token is still skipped.</summary>
public sealed class LiteralSpecialIdsTests
{
    [Fact]
    public void ListedLiteralsAreDecodedAsTextAndOtherSpecialsStillSkipped()
    {
        PieceTokenizer tok = new();
        int think = tok.SpecialIdOf(PieceTokenizer.Think);
        int thinkEnd = tok.SpecialIdOf(PieceTokenizer.ThinkEnd);
        int[] ids = tok.SplitBytes("a", 1);
        int[] stream = [.. ids, think, .. tok.SplitBytes("b"), thinkEnd, tok.SpecialIdOf(PieceTokenizer.Eos)];
        IncrementalDetokenizer detok = new(tok, includeSpecial: false, new HashSet<int> { think });
        string output = string.Concat(stream.Select(detok.Push)) + detok.Flush();
        // <think> is listed so it surfaces; </think> and <eos> are not, so they stay dropped.
        Assert.Equal("a<think>b", output);
    }

    [Fact]
    public void PassthroughWithLiteralIdsStreamsTheMarkersAsContent()
    {
        PieceTokenizer tok = new();
        string text = "Hello <think>inner</think> done";
        int[] ids = tok.RandomWithSpecials(text, new Random(3), 4);
        HashSet<int> literals = [tok.SpecialIdOf(PieceTokenizer.Think), tok.SpecialIdOf(PieceTokenizer.ThinkEnd)];
        PassthroughOutputParser parser = new(tok, literals);
        List<ParsedEvent> events = [];
        foreach (int id in ids) parser.Push(id, events.Add);
        parser.Finish(events.Add);
        Assert.Equal(text, parser.Result.Content);
    }

    [Fact]
    public void PassthroughWithoutLiteralIdsIsUnchanged()
    {
        PieceTokenizer tok = new();
        string text = "Hello <think>inner</think> wörld 😀";
        int[] ids = tok.RandomWithSpecials(text, new Random(9), 3);
        PassthroughOutputParser plain = new(tok);
        PassthroughOutputParser nullSet = new(tok, null);
        foreach (int id in ids)
        {
            plain.Push(id, _ => { });
            nullSet.Push(id, _ => { });
        }
        plain.Finish(_ => { });
        nullSet.Finish(_ => { });
        Assert.Equal(plain.Result.Content, nullSet.Result.Content);
        Assert.Equal("Hello inner wörld 😀", plain.Result.Content);
    }
}
