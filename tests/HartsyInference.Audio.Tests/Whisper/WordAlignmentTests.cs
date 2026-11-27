using System.Text.Json;
using HartsyInference.Audio.Dsp.Alignment;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Audio.Tests.Whisper;

/// <summary>The word-timing math behind Whisper word timestamps, checked without weights: the DTW and median filter
/// on small hand-made matrices, the word grouping on a fake vocabulary, and the whole attention-to-word-times chain
/// on attention built to peak where each token is "spoken".</summary>
public sealed unsafe class WordAlignmentTests
{
    [Fact]
    public void Dtw_FollowsTheCheapDiagonal()
    {
        float[] cost = [0f, 1f, 1f, 1f, 0f, 1f, 1f, 1f, 0f];

        (int[] rows, int[] cols) = DynamicTimeWarping.Align(cost, 3, 3);

        Assert.Equal([0, 1, 2], rows);
        Assert.Equal([0, 1, 2], cols);
    }

    [Fact]
    public void Dtw_HoldsARowAcrossAllTheFramesItIsCheapOn()
    {
        // Row 0 is cheap on frames 0-2, row 1 on frames 3-5.
        float[] cost =
        [
            0f, 0f, 0f, 5f, 5f, 5f,
            5f, 5f, 5f, 0f, 0f, 0f,
        ];

        (int[] rows, int[] cols) = DynamicTimeWarping.Align(cost, 2, 6);

        Assert.Equal([0, 0, 0, 1, 1, 1], rows);
        Assert.Equal([0, 1, 2, 3, 4, 5], cols);
    }

    [Fact]
    public void Dtw_PathIsMonotoneAndCoversEveryRow()
    {
        Random rng = new(7);
        float[] cost = new float[9 * 40];
        for (int i = 0; i < cost.Length; i++)
        {
            cost[i] = (float)rng.NextDouble();
        }

        (int[] rows, int[] cols) = DynamicTimeWarping.Align(cost, 9, 40);

        Assert.Equal((0, 0), (rows[0], cols[0]));
        Assert.Equal((8, 39), (rows[^1], cols[^1]));
        for (int k = 1; k < rows.Length; k++)
        {
            Assert.InRange(rows[k] - rows[k - 1], 0, 1);
            Assert.InRange(cols[k] - cols[k - 1], 0, 1);
            Assert.True(rows[k] != rows[k - 1] || cols[k] != cols[k - 1]);
        }
    }

    [Fact]
    public void Dtw_BreaksATieTowardTheRightwardStep()
    {
        // On a flat matrix every move costs the same, and OpenAI's aligner advances in time before it advances to the next row.
        (int[] rows, int[] cols) = DynamicTimeWarping.Align(new float[4], 2, 2);

        Assert.Equal([0, 1, 1], rows);
        Assert.Equal([0, 0, 1], cols);
    }

    [Fact]
    public void Dtw_RejectsAMismatchedMatrix()
    {
        Assert.Throws<ArgumentException>(() => DynamicTimeWarping.Align(new float[5], 2, 3));
        Assert.Throws<ArgumentOutOfRangeException>(() => DynamicTimeWarping.Align(new float[0], 0, 3));
    }

    [Fact]
    public void MedianFilter_RemovesASingleSpikeAndKeepsAStep()
    {
        float[] spike = [0f, 0f, 0f, 9f, 0f, 0f, 0f, 0f, 0f];
        MedianFilter.Rows(spike, 1, spike.Length, 5);
        Assert.All(spike, value => Assert.Equal(0f, value));

        float[] step = [0f, 0f, 0f, 0f, 1f, 1f, 1f, 1f];
        MedianFilter.Rows(step, 1, step.Length, 3);
        Assert.Equal([0f, 0f, 0f, 0f, 1f, 1f, 1f, 1f], step);
    }

    [Fact]
    public void MedianFilter_MirrorsTheEdgesAndFiltersEachRowAlone()
    {
        // Mirrored, the first sample's window is [2, 1, 2] so it becomes 2; the second row is independent.
        float[] data = [1f, 2f, 3f, 10f, 10f, 10f];

        MedianFilter.Rows(data, 2, 3, 3);

        Assert.Equal([2f, 2f, 2f, 10f, 10f, 10f], data);
    }

    [Fact]
    public void MedianFilter_RejectsAnEvenWindow()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => MedianFilter.Rows(new float[4], 1, 4, 2));
    }

    [Fact]
    public void AlignmentHeads_ReadsTheCheckpointList_AndDropsPairsThatDoNotFit()
    {
        WhisperConfig cfg = WhisperConfig.Tiny;
        using JsonDocument doc = JsonDocument.Parse($$"""{"alignment_heads": [[2, 2], [3, 0], [{{cfg.DecoderLayers}}, 0], [1, {{cfg.NumHeads}}]]}""");

        IReadOnlyList<WhisperAlignmentHead> heads = WhisperAlignmentHeads.Parse(doc.RootElement, cfg);

        Assert.Equal([new WhisperAlignmentHead(2, 2), new WhisperAlignmentHead(3, 0)], heads);
    }

    [Fact]
    public void AlignmentHeads_IgnoreAPairThatIsNotTwoNumbers()
    {
        WhisperConfig cfg = WhisperConfig.Tiny;
        using JsonDocument doc = JsonDocument.Parse("""{"alignment_heads": [["a", 1], [2, 1], [1]]}""");

        IReadOnlyList<WhisperAlignmentHead> heads = WhisperAlignmentHeads.Parse(doc.RootElement, cfg);

        Assert.Equal([new WhisperAlignmentHead(2, 1)], heads);
    }

    [Fact]
    public void AlignmentHeads_FallBackToTheUpperHalfOfTheDecoder()
    {
        WhisperConfig cfg = WhisperConfig.Tiny;
        using JsonDocument doc = JsonDocument.Parse("{}");

        IReadOnlyList<WhisperAlignmentHead> heads = WhisperAlignmentHeads.Parse(doc.RootElement, cfg);

        Assert.Equal((cfg.DecoderLayers - cfg.DecoderLayers / 2) * cfg.NumHeads, heads.Count);
        Assert.All(heads, head => Assert.True(head.Layer >= cfg.DecoderLayers / 2));
    }

    // A word is a leading-space token plus any continuation; a token id maps straight to its text here.
    private static Func<ReadOnlyMemory<int>, string> Decoder(string[] vocabulary) =>
        ids => string.Concat(ids.ToArray().Select(id => vocabulary[id]));

    [Fact]
    public void Splitter_StartsAWordAtEachLeadingSpace_AndJoinsPunctuationToItsNeighbour()
    {
        string[] vocabulary = [" Hel", "lo", ",", " wor", "ld", "!"];

        IReadOnlyList<WhisperWordTokens> words = WhisperWordSplitter.Split(Decoder(vocabulary), new[] { 0, 1, 2, 3, 4, 5 }, "en");

        Assert.Equal(
            [new WhisperWordTokens(" Hello,", 3, TrailTokens: 1), new WhisperWordTokens(" world!", 3, TrailTokens: 1)],
            words);
    }

    [Fact]
    public void Splitter_GivesASpacelessScriptOneWordPerCharacter_AndHoldsPartialCharactersTogether()
    {
        // Token 1 and 2 are the two halves of one character: alone each decodes to a replacement mark.
        string replacement = ((char)0xFFFD).ToString();
        string[] vocabulary = ["你", replacement, replacement, "好"];
        Func<ReadOnlyMemory<int>, string> decode = ids =>
        {
            int[] run = ids.ToArray();
            System.Text.StringBuilder text = new();
            for (int i = 0; i < run.Length; i++)
            {
                if (run[i] == 1 && i + 1 < run.Length && run[i + 1] == 2)
                {
                    text.Append('吗');
                    i++;
                }
                else
                {
                    text.Append(vocabulary[run[i]]);
                }
            }
            return text.ToString();
        };

        IReadOnlyList<WhisperWordTokens> words = WhisperWordSplitter.Split(decode, new[] { 0, 1, 2, 3 }, "zh");

        Assert.Equal(
            [new WhisperWordTokens("你", 1), new WhisperWordTokens("吗", 2), new WhisperWordTokens("好", 1)],
            words);
    }

    [Fact]
    public void Splitter_KeepsAMarkWithNoSpaceInItsWord_AndTimesPunctuationOutOfAWord()
    {
        // "..." is not one of the single marks the rule treats as stray punctuation, so it stays on the word before it.
        string[] vocabulary = [" wait", "...", " \"", " Hello", "."];

        Assert.Equal([new WhisperWordTokens(" wait...", 2)], WhisperWordSplitter.Split(Decoder(vocabulary), new[] { 0, 1 }, "en"));
        Assert.Equal(
            [new WhisperWordTokens(" \" Hello.", 3, LeadTokens: 1, TrailTokens: 1)],
            WhisperWordSplitter.Split(Decoder(vocabulary), new[] { 2, 3, 4 }, "en"));
    }

    [Theory]
    [InlineData("zh-CN", 2)]
    [InlineData("ZH", 2)]
    [InlineData("ja_JP", 2)]
    [InlineData("en", 1)]
    [InlineData(null, 2)]
    public void Splitter_ReadsLocaleCodes_AndFallsBackToTheTextsScript(string? language, int expectedWords)
    {
        string[] vocabulary = ["你", "好"];

        Assert.Equal(expectedWords, WhisperWordSplitter.Split(Decoder(vocabulary), new[] { 0, 1 }, language).Count);
    }

    [Fact]
    public void Splitter_TreatsUnlabelledLatinTextAsSpaced()
    {
        string[] vocabulary = [" Hel", "lo", " world"];

        Assert.Equal(2, WhisperWordSplitter.Split(Decoder(vocabulary), new[] { 0, 1, 2 }, null).Count);
    }

    [Fact]
    public void Splitter_DoesNotLetARealReplacementMarkSwallowTheTextAfterIt()
    {
        string replacement = ((char)0xFFFD).ToString();
        string[] vocabulary = [" a", replacement, " b"];

        IReadOnlyList<WhisperWordTokens> words = WhisperWordSplitter.Split(Decoder(vocabulary), new[] { 0, 1, 2 }, "en");

        Assert.Equal([new WhisperWordTokens(" a" + replacement, 2), new WhisperWordTokens(" b", 1)], words);
    }

    /// <summary>Attention that peaks on frames <c>[10r, 10r + 10)</c> for the row at absolute decoder position <c>1 +
    /// r</c>: queries and keys are one-hot over eight slots, so the cross-attention softmax is flat except on the
    /// frames a row is meant to read.</summary>
    private static WhisperAttentionProbe PeakedProbe(int rows, int framesPerRow, int frames)
    {
        const int headDim = 8;
        IBackend backend = new CpuBackend();
        Tensor query = new(new TensorShape(1, 1, rows, headDim), DType.F32);
        Tensor keys = new(new TensorShape(1, 1, frames, headDim), DType.F32);
        float* q = (float*)query.DataPointer;
        float* k = (float*)keys.DataPointer;
        new Span<float>(q, rows * headDim).Clear();
        new Span<float>(k, frames * headDim).Clear();
        for (int r = 0; r < rows; r++)
        {
            // Row 0 (the start token) and the last row (end of text) match no key, so they read evenly.
            int slot = r >= 1 && r <= rows - 2 ? r - 1 : 7;
            q[r * headDim + slot] = 1f;
        }
        for (int f = 0; f < frames; f++)
        {
            k[f * headDim + Math.Min(f / framesPerRow, 6)] = 10f;
        }

        WhisperAttentionProbe probe = new([new WhisperAlignmentHead(0, 0)], frames);
        probe.Capture(backend, layer: 0, query, keys, rows, scale: 1f);
        query.Dispose();
        keys.Dispose();
        return probe;
    }

    [Fact]
    public void Probe_MeanAttentionPeaksWhereARowReads_AndEachFrameIsZeroMeanAcrossRows()
    {
        WhisperAttentionProbe probe = PeakedProbe(rows: 6, framesPerRow: 10, frames: 50);

        float[] mean = probe.MeanAttention();

        Assert.Equal(6 * 50, mean.Length);
        // Row 2 reads frames 10-19, so it stands above its own reading of frame 45 after the per-frame normalization.
        Assert.True(mean[2 * 50 + 15] > mean[2 * 50 + 45] + 1f);
        for (int f = 0; f < 50; f++)
        {
            float sum = 0f;
            for (int r = 0; r < 6; r++)
            {
                sum += mean[r * 50 + f];
            }
            // The median filter moves a column's sum only a little, so it stays near zero.
            Assert.InRange(sum, -1f, 1f);
        }
    }

    [Fact]
    public void Probe_RefusesAMeanBeforeEveryHeadWasCaptured()
    {
        WhisperAttentionProbe probe = new([new WhisperAlignmentHead(0, 0), new WhisperAlignmentHead(1, 0)], 50);

        Assert.Throws<InvalidOperationException>(() => probe.MeanAttention());
    }

    [Theory]
    [InlineData(null, null)]
    [InlineData("", null)]
    [InlineData("  \n\t ", null)]
    [InlineData("hold on\nlet go", "hold on let go")]
    [InlineData("  a\t\tb   c\r\n\r\nd ", "a b c d")]
    public void KnownText_IsCollapsedToSingleSpaces_OrDroppedWhenBlank(string? text, string? expected)
    {
        Assert.Equal(expected, WhisperWordSplitter.NormalizeKnownText(text));
    }

    [Fact]
    public void Timings_PlaceEachWordWhereItsTokensAreRead()
    {
        // Sequence: [start, no-timestamps, t0, t1, t2, end]. Used row r introduces text token r, and reads frames
        // [10r, 10r + 10), so the words {t0, t1} and {t2} should span 0.0-0.4 s and 0.4-0.6 s.
        WhisperAttentionProbe probe = PeakedProbe(rows: 6, framesPerRow: 10, frames: 50);
        WhisperWordTokens[] words = [new(" ab", 2), new(" c", 1)];

        IReadOnlyList<WhisperWord> timed = WhisperWordTimings.Compute(probe, promptLength: 2, words);

        Assert.Equal(2, timed.Count);
        Assert.Equal(" ab", timed[0].Text);
        Assert.Equal(0.0, timed[0].Start, 2);
        Assert.Equal(0.4, timed[0].End, 1);
        Assert.Equal(timed[0].End, timed[1].Start, 5);
        Assert.Equal(0.6, timed[1].End, 1);
    }

    [Fact]
    public void Timings_LeaveAttachedPunctuationOutOfAWordsSpan()
    {
        // Sequence: [start, no-timestamps, t0, t1, comma, t3, end]: the comma belongs to the first word's text but not to its span.
        WhisperAttentionProbe probe = PeakedProbe(rows: 7, framesPerRow: 10, frames: 70);
        WhisperWordTokens[] words = [new(" ab,", 3, TrailTokens: 1), new(" c", 1)];

        IReadOnlyList<WhisperWord> timed = WhisperWordTimings.Compute(probe, promptLength: 2, words);

        Assert.Equal(0.0, timed[0].Start, 2);
        Assert.Equal(0.4, timed[0].End, 1);
        Assert.Equal(0.6, timed[1].Start, 1);
        Assert.Equal(0.8, timed[1].End, 1);
    }

    [Fact]
    public void Timings_RefuseAProbeFromADifferentSequence()
    {
        WhisperAttentionProbe probe = PeakedProbe(rows: 6, framesPerRow: 10, frames: 50);

        Assert.Throws<ArgumentException>(() => WhisperWordTimings.Compute(probe, promptLength: 2, [new WhisperWordTokens(" ab", 2)]));
    }
}
