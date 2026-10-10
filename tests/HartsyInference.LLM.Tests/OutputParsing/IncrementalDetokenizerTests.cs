using System.Text;
using HartsyInference.LLM.OutputParsing;
using Xunit;

namespace HartsyInference.LLM.Tests.OutputParsing;

public sealed class IncrementalDetokenizerTests
{
    [Fact]
    public void MultibyteCharacterSplitAcrossTokensIsHeldBackThenEmittedWhole()
    {
        PieceTokenizer tok = new();
        int[] ids = tok.SplitBytes("é😀", 1, 2, 3, 4, 5);
        IncrementalDetokenizer d = new(tok, includeSpecial: true);
        Assert.Equal("", d.Push(ids[0]));
        Assert.Equal("é", d.Push(ids[1]));
        Assert.Equal("", d.Push(ids[2]));
        Assert.Equal("", d.Push(ids[3]));
        Assert.Equal("", d.Push(ids[4]));
        Assert.Equal("😀", d.Push(ids[5]));
        Assert.Equal("", d.Flush());
    }

    [Fact]
    public void ControlTokensAreSkippedOrKeptAsLiterals()
    {
        PieceTokenizer tok = new();
        int[] ids = [tok.Add("a"u8.ToArray()), tok.SpecialIdOf(PieceTokenizer.ThinkEnd), tok.Add("b"u8.ToArray())];
        Assert.Equal("ab", Run(new IncrementalDetokenizer(tok, includeSpecial: false), ids));
        Assert.Equal("a</think>b", Run(new IncrementalDetokenizer(tok, includeSpecial: true), ids));
    }

    [Fact]
    public void TokenizerWithoutTokenBytesFallsBackToTheFullDecodeDelta()
    {
        FallbackTokenizer tok = new();
        IncrementalDetokenizer d = new(tok, includeSpecial: false);
        Assert.Equal("he", d.Push(0) + d.Push(1));
        Assert.Equal("llo", d.Push(2));
        Assert.Equal("", d.Flush());
    }

    private static string Run(IncrementalDetokenizer d, int[] ids)
    {
        StringBuilder sb = new();
        foreach (int id in ids) sb.Append(d.Push(id));
        sb.Append(d.Flush());
        return sb.ToString();
    }

    private sealed class FallbackTokenizer : NoBytesTokenizer
    {
        private static readonly string[] Words = ["h", "e", "llo"];

        public override string Decode(IReadOnlyList<int> ids) => string.Concat(ids.Select(i => Words[i]));
    }
}
