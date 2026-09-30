using HartsyInference.LLM.OutputParsing;
using HartsyInference.ModelAssets.Tokenizers;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.LLM.Tests.OutputParsing;

/// <summary>The parser on the real DeepSeek-V4.1 vocabulary (tokenizer.json is not committed): pinned special ids and BPE tokens that split characters and markers.</summary>
[Trait("Category", "Integration")]
public sealed class DeepSeekV41OutputParserRealTokenizerTests
{
    private readonly ITestOutputHelper _output;

    public DeepSeekV41OutputParserRealTokenizerTests(ITestOutputHelper output) => _output = output;

    private GgufTokenizer? LoadTokenizer()
    {
        string? path = DeepSeekV41ReferenceFiles.FindTokenizerJson();
        if (path is null)
        {
            _output.WriteLine("SKIPPED: DeepSeek-V4.1 tokenizer.json not found (set DSV41_TOKENIZER_JSON).");
            return null;
        }
        using FileStream stream = File.OpenRead(path);
        return HfTokenizerJson.LoadByteLevelBpe(stream, bosToken: "<｜begin▁of▁sentence｜>", eosToken: "<｜end▁of▁sentence｜>");
    }

    [Fact]
    public void PinnedControlTokenIdsAreSingleTokens()
    {
        GgufTokenizer? tok = LoadTokenizer();
        if (tok is null) return;
        Assert.Equal(0, tok.SpecialId(PieceTokenizer.Bos));
        Assert.Equal(1, tok.SpecialId(PieceTokenizer.Eos));
        Assert.Equal(128821, tok.SpecialId(PieceTokenizer.Think));
        Assert.Equal(128822, tok.SpecialId(PieceTokenizer.ThinkEnd));
        Assert.Equal(128825, tok.SpecialId(PieceTokenizer.Dsml));
        Assert.Equal([128822], tok.Encode(PieceTokenizer.ThinkEnd, true));
        Assert.Equal([1], tok.Encode(PieceTokenizer.Eos, true));
    }

    [Fact]
    public void EveryReferenceCompletionParsesToTheReferenceOnRealBpeTokens()
    {
        GgufTokenizer? tok = LoadTokenizer();
        if (tok is null) return;
        int checkedCases = 0;
        foreach (ParserCase c in ParserCase.All.Where(x => x.Ok))
        {
            int[] ids = tok.Encode(c.Text, true);
            DeepSeekV41OutputParser parser = new(tok, c.Thinking ? OutputParserState.Reasoning : OutputParserState.Content);
            ParsedAssistant got = ReplayedTurn.Run(parser, ids, out ReplayedTurn replay);
            ParserTestHelpers.AssertMatches(c, got, replay, "real-bpe");
            checkedCases++;
        }
        Assert.True(checkedCases >= 20);
        _output.WriteLine($"Parsed {checkedCases} reference completions on real BPE tokens.");
    }

    [Fact]
    public void MalformedReferenceCompletionsAreFlaggedOnRealBpeTokens()
    {
        GgufTokenizer? tok = LoadTokenizer();
        if (tok is null) return;
        foreach (ParserCase c in ParserCase.All.Where(x => !x.Ok))
        {
            DeepSeekV41OutputParser parser = new(tok, c.Thinking ? OutputParserState.Reasoning : OutputParserState.Content);
            ParsedAssistant got = ReplayedTurn.Run(parser, tok.Encode(c.Text, true), out _);
            Assert.True(got.Malformed, $"{c.Name} not flagged");
        }
    }
}
