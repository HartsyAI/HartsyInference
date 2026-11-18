using System.Text.Json;
using HartsyInference.Audio.Dsp.Alignment;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Audio.Tests.Whisper;

/// <summary>The word-timing math behind Whisper word timestamps, checked without weights: the DTW and median filter on small hand-made matrices, the word grouping on a fake vocabulary, and the whole attention-to-word-times chain on attention built to peak where each token is "spoken".</summary>
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
            [new WhisperWordTokens(" Hello,", 3), new WhisperWordTokens(" world!", 3)],
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
            if (run.SequenceEqual([1, 2]))
            {
                return "吗";
            }
            return string.Concat(run.Select(id => vocabulary[id]));
        };

        IReadOnlyList<WhisperWordTokens> words = WhisperWordSplitter.Split(decode, new[] { 0, 1, 2, 3 }, "zh");

        Assert.Equal(
            [new WhisperWordTokens("你", 1), new WhisperWordTokens("吗", 2), new WhisperWordTokens("好", 1)],
            words);
    }

    /// <summary>Attention that peaks on frames <c>[10r, 10r + 10)</c> for the row at absolute decoder position <c>1 + r</c>: queries and keys are one-hot over eight slots, so the cross-attention softmax is flat except on the frames a row is meant to read.</summary>
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
    public void Probe_RecordsASoftmaxOverTheCroppedFrames()
    {
        WhisperAttentionProbe probe = PeakedProbe(rows: 6, framesPerRow: 10, frames: 50);

        ReadOnlySpan<float> weights = probe.HeadWeights(0);

        Assert.Equal(6 * 50, weights.Length);
        for (int r = 0; r < 6; r++)
        {
            float sum = 0f;
            for (int f = 0; f < 50; f++)
            {
                sum += weights[r * 50 + f];
            }
            Assert.Equal(1f, sum, 4);
        }
        // Row 2 reads frames 10-19 (the keys marked 1) far more than frame 45.
        Assert.True(weights[2 * 50 + 15] > 50f * weights[2 * 50 + 45]);
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
    public void Timings_RefuseAProbeFromADifferentSequence()
    {
        WhisperAttentionProbe probe = PeakedProbe(rows: 6, framesPerRow: 10, frames: 50);

        Assert.Throws<ArgumentException>(() => WhisperWordTimings.Compute(probe, promptLength: 2, [new WhisperWordTokens(" ab", 2)]));
    }
}
