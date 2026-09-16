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

    [Fact]
    public void Vocabulary_IsTheWidthTheCheckpointWasTrainedWith()
    {
        Assert.Equal(31_678, Tok.TokenCount);
        // 1 no-chord, then 5 qualities carrying 3 inversions plus root position, and 10 in root position only.
        Assert.Equal(1 + (5 * 12 * 4) + (10 * 12 * 1), Tok.FullChordLabels.Count);
        Assert.Equal(361, Tok.FullChordLabels.Count);
        Assert.Equal(192, Tok.MeterPairs.Count);
        Assert.Equal(23, ScoreTokenizer.StructureLabels.Length);
        Assert.Equal(24, ScoreTokenizer.DurationTemplates.Length);
    }

    [Fact]
    public void Ranges_AreContiguousFrom260_InLayoutOrder()
    {
        (string Name, int Start, int End)[] expected =
        [
            ("subbeat_shift", 260, 517),
            ("time", 517, 30_517),
            ("meter", 30_517, 30_709),
            ("eighth_position", 30_709, 30_965),
            ("structure", 30_965, 30_988),
            ("key", 30_988, 31_012),
            ("chord_majmin", 31_012, 31_037),
            ("chord_full", 31_037, 31_398),
            ("pitch", 31_398, 31_654),
            ("duration", 31_654, 31_678),
        ];
        Assert.Equal(expected.Length, Tok.Ranges.Count);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i], Tok.Ranges[i]);
            if (i > 0) Assert.Equal(Tok.Ranges[i - 1].End, Tok.Ranges[i].Start);
        }
    }

    [Fact]
    public void PromptPrefix_IsTheReleaseTaskSelection()
        => Assert.Equal([1, 4, 5, 6, 7, 9, 11, 3], ScoreTokenizer.PromptPrefix());

    [Theory]
    [InlineData(0, "pad")]
    [InlineData(1, "sos")]
    [InlineData(2, "eos")]
    [InlineData(3, "out")]
    [InlineData(7, "prompt")]
    [InlineData(260, "subbeat_shift")]
    [InlineData(517, "time")]
    [InlineData(30_516, "time")]
    [InlineData(31_012, "chord_majmin")]
    [InlineData(31_037, "chord_full")]
    [InlineData(31_677, "duration")]
    public void TokenType_ReadsTheRange(int token, string expected) => Assert.Equal(expected, Tok.TokenType(token));

    [Fact]
    public void Timestamps_CountAtOneHundredHertz()
    {
        Assert.Equal(0, Tok.TokenToTimeId(Tok.TimeStart));
        Assert.Equal(1.5, Tok.TokenToSeconds(Tok.TimeStart + 150), 6);
        // The range spans five minutes, which is the window the encoder attends over.
        Assert.Equal(300.0, (Tok.TimeEnd - Tok.TimeStart) / (double)ScoreTokenizer.TimeHz, 6);
    }

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

    /// <summary>A half-written field admits only its partner among the FIELDS — but ending and shifting are
    /// decided before that rule applies, so they stay legal. Subtle, and getting it backwards would refuse
    /// every sequence that ends on a note.</summary>
    [Fact]
    public void Grammar_DemandsAPositionAfterAMeter_ButStillPermitsEnding()
    {
        PromptGrammar g = new(Tok);
        g.Update(Tok.MeterStart + 3);
        bool[] allowed = g.Allowed();
        Assert.True(allowed[Tok.EighthPositionStart]);
        Assert.False(allowed[Tok.StructureStart]);
        Assert.False(allowed[Tok.PitchStart]);
        Assert.False(allowed[Tok.TimeStart]);
        Assert.True(allowed[ScoreTokenizer.EosToken]);
        Assert.True(allowed[Tok.SubbeatShiftStart]);
    }

    [Fact]
    public void Grammar_DemandsADurationOrAnotherPitchAfterAPitch_ButStillPermitsEnding()
    {
        PromptGrammar g = new(Tok);
        g.Update(Tok.PitchStart + 60);
        bool[] allowed = g.Allowed();
        Assert.True(allowed[Tok.DurationStart]);
        Assert.True(allowed[Tok.PitchStart]);
        Assert.False(allowed[Tok.StructureStart]);
        Assert.False(allowed[Tok.FullChordStart]);
        Assert.True(allowed[ScoreTokenizer.EosToken]);
    }

    [Fact]
    public void Grammar_KeepsFieldsInOrder_ButLetsMelodyRepeat()
    {
        PromptGrammar g = new(Tok);
        g.Update(Tok.KeyStart + 1);
        bool[] allowed = g.Allowed();
        // Earlier fields have closed.
        Assert.False(allowed[Tok.TimeStart]);
        Assert.False(allowed[Tok.MeterStart]);
        Assert.False(allowed[Tok.StructureStart]);
        // Later ones are still open.
        Assert.True(allowed[Tok.FullChordStart]);
        Assert.True(allowed[Tok.PitchStart]);

        // Melody is the one field that stays open at its own index.
        g.Update(Tok.PitchStart + 60);
        g.Update(Tok.DurationStart + 2);
        Assert.True(g.Allowed()[Tok.PitchStart]);
        Assert.False(g.Allowed()[Tok.FullChordStart]);
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

    [Fact]
    public void Grammar_ReportsTheEndToken() => Assert.True(new PromptGrammar(Tok).Update(ScoreTokenizer.EosToken));
}
