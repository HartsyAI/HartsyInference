using HartsyInference.Audio.Models.SheetSage2;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>SheetSage2's vocabulary layout and decode grammar. Both are checkpoint facts: the decoder's output
/// projection is 31,678 wide and every id's meaning comes from which range it lands in, so an off-by-one here
/// mis-reads every transcription rather than failing loudly.
/// <para>Pure — no checkpoint, backend or GPU.</para></summary>
public sealed class SheetSage2TokenizerTests
{
    private static readonly ScoreTokenizer Tok = new();

    [Theory]
    [InlineData(0, "pad")]
    [InlineData(31_677, "duration")]
    public void TokenType_ReadsTheRange(int token, string expected) => Assert.Equal(expected, Tok.TokenType(token));

    /// <summary>The majmin range is 25 ids the grammar never opens — it exists in the layout but the released
    /// decode only ever writes full chords. Worth pinning: deleting it would shift every later range.</summary>
    [Fact]
    public void MajMinChords_OccupyIdsTheGrammarNeverAllows()
    {
        bool[] allowed = new PromptGrammar(Tok).Allowed();
        for (int t = Tok.MajMinChordStart; t < Tok.MajMinChordEnd; t++) Assert.False(allowed[t]);
        Assert.Equal(25, Tok.MajMinChordEnd - Tok.MajMinChordStart);
    }

    [Fact]
    public void Grammar_RefusesToEndBeforeAnEventIsWritten()
    {
        PromptGrammar g = new(Tok);
        Assert.False(g.Allowed()[ScoreTokenizer.EosToken]);
        g.Update(Tok.TimeStart + 10);
        Assert.True(g.Allowed()[ScoreTokenizer.EosToken]);
    }

    /// <summary>The run limit is read before the shift is counted, so four may be emitted and the fifth is
    /// refused — that is what stops a decode walking forward forever without writing anything.</summary>
    [Fact]
    public void Grammar_StopsAFifthConsecutiveShift()
    {
        PromptGrammar g = new(Tok);
        for (int i = 0; i < 4; i++)
        {
            Assert.True(g.Allowed()[Tok.SubbeatShiftStart], $"shift {i + 1} should be allowed");
            g.Update(Tok.SubbeatShiftStart);
        }
        Assert.False(g.Allowed()[Tok.SubbeatShiftStart]);
    }

    [Fact]
    public void Grammar_ShiftClosesAnEventAndReopensEveryField()
    {
        PromptGrammar g = new(Tok);
        g.Update(Tok.KeyStart + 1);
        Assert.Equal(0, g.EventCount);
        g.Update(Tok.SubbeatShiftStart + 1);
        Assert.Equal(1, g.EventCount);
        bool[] allowed = g.Allowed();
        Assert.True(allowed[Tok.TimeStart]);
        Assert.True(allowed[Tok.MeterStart]);
        Assert.True(allowed[Tok.StructureStart]);
    }

}
