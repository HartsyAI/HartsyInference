using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.ModelAssets.Tokenizers;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>CPU capability facts for the MiniMax-H3-Image text-to-image path: the sigma grid and output-head selection,
/// the packed layout and its rotary positions, the timestep rows, latent packing, the prompt presentation and the
/// single-frame VAE configuration. Expected values are derived from the dossier formulas
/// (docs/Research/MINIMAX_H3_IMAGE_ARCHITECTURE.md), not copied from the code under test. No weights are read.</summary>
public sealed class MiniMaxH3ImageCapabilityTests
{
    private static readonly double[] SixStepGrid =
    [
        1.0, 0.9675236940383911, 0.8875485062599182, 0.75, 0.5327739119529724, 0.23200887441635132, 0.0,
    ];

    // ---- Sigma grid and output-head selection -------------------------------------------------------------------

    [Fact]
    public void SixStepGridGivesSixHeadsOverSevenPoints()
    {
        StackedOutputHeads heads = new(SixStepGrid);

        Assert.Equal(6, heads.HeadCount);
        Assert.Equal(7, MiniMaxH3Schedule.ImageSigmas(SixStepGrid).Length);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    public void MidSchedule_StepNSelectsOneIntervalAndBlockN(int n)
    {
        StackedOutputHeads heads = new(SixStepGrid);

        OutputHeadStep step = heads.Resolve(SixStepGrid[n], SixStepGrid[n + 1]);

        Assert.Equal(n, step.Start);
        Assert.Equal(n + 1, step.Stop);
        Assert.Equal(n, step.FirstOffset);
        for (int k = 0; k < heads.HeadCount; k++)
        {
            Assert.Equal(k == n ? 1.0 : 0.0, step.Weights[k], 12);
        }
    }

    [Fact]
    public void FirstStep_UsesBlockZeroAloneBecauseItsOffsetRangeIsEmpty()
    {
        StackedOutputHeads heads = new(SixStepGrid);

        OutputHeadStep step = heads.Resolve(1.0, SixStepGrid[1]);

        Assert.Equal(0, step.Start);
        Assert.Equal(1, step.Stop);
        Assert.Equal(1, step.FirstOffset);
        Assert.All(step.Weights, w => Assert.Equal(0.0, w));
    }

    [Fact]
    public void SpanOverThreeIntervals_WeightsEachOffsetByItsWidth_AndTheOffsetsDoNotSumToOne()
    {
        StackedOutputHeads heads = new(SixStepGrid);

        // sigma 1.0 -> 0.75 covers intervals 0, 1 and 2. Block 0's own width is in the total, so the offsets stop at 0.87.
        OutputHeadStep step = heads.Resolve(1.0, 0.75);
        double total = 1.0 - 0.75;

        Assert.Equal(0, step.Start);
        Assert.Equal(3, step.Stop);
        Assert.Equal(1, step.FirstOffset);
        Assert.Equal((SixStepGrid[1] - SixStepGrid[2]) / total, step.Weights[1], 12);
        Assert.Equal((SixStepGrid[2] - SixStepGrid[3]) / total, step.Weights[2], 12);
        double offsetSum = step.Weights[1] + step.Weights[2];
        Assert.Equal((1.0 - 0.75 - (1.0 - SixStepGrid[1])) / total, offsetSum, 12);
        Assert.True(offsetSum < 1.0);
    }

    [Fact]
    public void SpanStartingAboveBlockZero_NormalisesItsOffsetsToOne()
    {
        StackedOutputHeads heads = new(SixStepGrid);

        OutputHeadStep step = heads.Resolve(SixStepGrid[2], SixStepGrid[4]);

        Assert.Equal(2, step.Start);
        Assert.Equal(4, step.Stop);
        Assert.Equal(2, step.FirstOffset);
        Assert.Equal(1.0, step.Weights[2] + step.Weights[3], 12);
    }

    [Fact]
    public void EqualDistanceTie_PicksTheLowerGridIndex()
    {
        // sigma 0.75 is equidistant from grid points 1.0 and 0.5; the first minimum (index 0) wins, as in ComfyUI.
        StackedOutputHeads heads = new([1.0, 0.5, 0.0]);

        OutputHeadStep step = heads.Resolve(0.75, 0.25);

        Assert.Equal(0, step.Start);
        Assert.Equal(1, step.Stop);
    }

    [Fact]
    public void HeadGrid_MustStrictlyDecrease()
    {
        Assert.Throws<ArgumentException>(() => new StackedOutputHeads([1.0, 0.5, 0.6, 0.0]));
    }

    // ---- Effective head weights (numerical) --------------------------------------------------------------------

    [Fact]
    public void Combine_FirstStep_ReturnsBlockZeroRowsExactly()
    {
        const int outRows = 3, heads = 6, inDim = 4;
        using CpuBackend backend = new();
        (Tensor stackedW, Tensor stackedB) = StackedHead(outRows, heads, inDim);
        using (stackedW)
        using (stackedB)
        {
            OutputHeadStep step = new StackedOutputHeads(SixStepGrid).Resolve(1.0, SixStepGrid[1]);

            (Tensor w, Tensor b) = StackedOutputHeadWeights.Combine(backend, stackedW, stackedB, outRows, step);
            using (w)
            using (b)
            {
                Assert.Equal(stackedW.AsSpan<float>()[..(outRows * inDim)].ToArray(), w.AsSpan<float>().ToArray());
                Assert.Equal(stackedB.AsSpan<float>()[..outRows].ToArray(), b.AsSpan<float>().ToArray());
            }
        }
    }

    [Fact]
    public void Combine_SpanOverThreeIntervals_AddsEachOffsetBlockScaledByItsWeight()
    {
        const int outRows = 3, heads = 6, inDim = 4;
        using CpuBackend backend = new();
        (Tensor stackedW, Tensor stackedB) = StackedHead(outRows, heads, inDim);
        using (stackedW)
        using (stackedB)
        {
            OutputHeadStep step = new StackedOutputHeads(SixStepGrid).Resolve(1.0, 0.75);
            float[] sw = stackedW.AsSpan<float>().ToArray();
            float[] sb = stackedB.AsSpan<float>().ToArray();

            (Tensor w, Tensor b) = StackedOutputHeadWeights.Combine(backend, stackedW, stackedB, outRows, step);
            using (w)
            using (b)
            {
                float[] actualW = w.AsSpan<float>().ToArray();
                float[] actualB = b.AsSpan<float>().ToArray();
                for (int i = 0; i < outRows * inDim; i++)
                {
                    double expected = sw[i] + step.Weights[1] * sw[outRows * inDim + i] + step.Weights[2] * sw[2 * outRows * inDim + i];
                    Assert.Equal(expected, actualW[i], 4);
                }
                for (int i = 0; i < outRows; i++)
                {
                    double expected = sb[i] + step.Weights[1] * sb[outRows + i] + step.Weights[2] * sb[2 * outRows + i];
                    Assert.Equal(expected, actualB[i], 4);
                }
            }
        }
    }

    // ---- Packed layout and rotary positions ---------------------------------------------------------------------

    [Fact]
    public void TextToImageLayout_IsTextThenOneTargetVideoSegment()
    {
        MiniMaxH3PackedLayout layout = new(textLen: 7, latentT: 1, latentH: 4, latentW: 6, audioT: 0);

        Assert.Equal(13, layout.SequenceLength);
        Assert.Contains(layout.Segments, s => s.Kind == MiniMaxH3SegmentKind.Text && s.Start == 0 && s.Stop == 7);
        Assert.Contains(layout.Segments, s => s.Kind == MiniMaxH3SegmentKind.Video && s.Start == 7 && s.Stop == 13);
        Assert.DoesNotContain(layout.Segments, s => s.Kind is MiniMaxH3SegmentKind.Cond or MiniMaxH3SegmentKind.RefImage);
    }

    [Fact]
    public void TextRows_CountTimeAndKeepSpatialCoordinatesAtZero()
    {
        MiniMaxH3PackedLayout layout = new(textLen: 7, latentT: 1, latentH: 4, latentW: 6, audioT: 0);

        for (int i = 0; i < 7; i++)
        {
            Assert.Equal(i, layout.PositionIds[i * 3], 12);
            Assert.Equal(0.0, layout.PositionIds[i * 3 + 1], 12);
            Assert.Equal(0.0, layout.PositionIds[i * 3 + 2], 12);
        }
    }

    [Fact]
    public void TargetRows_SitAtTheTextCursorInTime_AndFollowTheAreaNormalisedAxes()
    {
        const int textLen = 7, latentH = 4, latentW = 6;
        MiniMaxH3PackedLayout layout = new(textLen, 1, latentH, latentW, 0);

        // Reference: linspace((1-r)/2, (1+r)/2, n, endpoint=False) * 32 with r = dim / sqrt(h*w) and n = dim / 2.
        double area = Math.Sqrt((double)latentH * latentW);
        double[] hAxis = Axis(latentH, area);
        double[] wAxis = Axis(latentW, area);
        int k = 0;
        for (int h = 0; h < hAxis.Length; h++)
        {
            for (int w = 0; w < wAxis.Length; w++, k++)
            {
                int row = textLen + k;
                Assert.Equal(textLen, layout.PositionIds[row * 3], 12);
                Assert.Equal(hAxis[h], layout.PositionIds[row * 3 + 1], 9);
                Assert.Equal(wAxis[w], layout.PositionIds[row * 3 + 2], 9);
            }
        }
        Assert.Equal(hAxis.Length * wAxis.Length, k);
    }

    [Fact]
    public void ImageUpdate_IsIndexedByVideoRowsAndEveryVideoRowIsADenoiseTarget()
    {
        MiniMaxH3PackedLayout layout = new(textLen: 7, latentT: 1, latentH: 4, latentW: 6, audioT: 0);

        // The flag array holds one entry per video row (text rows are not listed), and all of them are targets.
        Assert.Equal(6, layout.ImageUpdate.Length);
        Assert.All(layout.ImageUpdate, u => Assert.True(u));
    }

    [Fact]
    public void ThreeAxisRope_BlocksTimeThenHeightThenWidth_AndDuplicatesEachPairHalf()
    {
        // One token at t=2, h=0.5, w=1.5 with two frequencies per axis: rotary width 3 x 2 x 2 = 12 in a head of 12.
        float[] invFreq = [1f, 0.5f];
        (Tensor cos, Tensor sin) = MiniMaxH3Rope.BuildTables([2.0, 0.5, 1.5], invFreq, headDim: 12);
        using (cos)
        using (sin)
        {
            double[] expectedAngles = [2.0 * 1.0, 2.0 * 0.5, 0.5 * 1.0, 0.5 * 0.5, 1.5 * 1.0, 1.5 * 0.5];
            Span<float> c = cos.AsSpan<float>();
            Span<float> s = sin.AsSpan<float>();
            for (int slot = 0; slot < 6; slot++)
            {
                Assert.Equal(Math.Cos(expectedAngles[slot]), c[slot], 5);
                Assert.Equal(Math.Sin(expectedAngles[slot]), s[slot], 5);
                Assert.Equal(c[slot], c[slot + 6], 6);
                Assert.Equal(s[slot], s[slot + 6], 6);
            }
        }
    }

    [Fact]
    public void RotaryTable_LeavesDimsBeyondTheRotaryWidthAtZero()
    {
        (Tensor cos, Tensor sin) = MiniMaxH3Rope.BuildTables([3.0, 1.0, 1.0], [1f, 0.5f], headDim: 16);
        using (cos)
        using (sin)
        {
            Span<float> c = cos.AsSpan<float>();
            Span<float> s = sin.AsSpan<float>();
            for (int slot = 12; slot < 16; slot++)
            {
                Assert.Equal(0f, c[slot]);
                Assert.Equal(0f, s[slot]);
            }
        }
    }

    // ---- Timestep rows ------------------------------------------------------------------------------------------

    [Fact]
    public void TextToImage_TextAndVideoShareOneStreamTimestepRow()
    {
        MiniMaxH3PackedLayout layout = new(7, 1, 4, 6, 0);

        (float[] timesteps, IReadOnlyDictionary<MiniMaxH3SegmentKind, int> rowOf) =
            MiniMaxH3Conditioning.BuildTimestepRows(layout, 0.25f, 0f, MiniMaxH3Schedule.VisualCondTimestep,
                MiniMaxH3Schedule.AudioCondTimestep);

        Assert.Equal(rowOf[MiniMaxH3SegmentKind.Video], rowOf[MiniMaxH3SegmentKind.Text]);
        Assert.Equal(0.25f, timesteps[rowOf[MiniMaxH3SegmentKind.Video]]);
        Assert.Subset(new HashSet<MiniMaxH3SegmentKind> { MiniMaxH3SegmentKind.Text, MiniMaxH3SegmentKind.Video, MiniMaxH3SegmentKind.Audio },
            rowOf.Keys.ToHashSet());
    }

    [Fact]
    public void FirstStep_StreamTimestepIsZeroAndSitsAtRowZero()
    {
        MiniMaxH3PackedLayout layout = new(7, 1, 4, 6, 0);

        (float[] timesteps, IReadOnlyDictionary<MiniMaxH3SegmentKind, int> rowOf) =
            MiniMaxH3Conditioning.BuildTimestepRows(layout, 0f, 0f, MiniMaxH3Schedule.VisualCondTimestep,
                MiniMaxH3Schedule.AudioCondTimestep);

        Assert.Equal(0, rowOf[MiniMaxH3SegmentKind.Video]);
        Assert.Equal(0f, timesteps[0]);
    }

    // ---- Latent packing -----------------------------------------------------------------------------------------

    [Fact]
    public void PackThenUnpack_RoundTripsATextToImageLatentExactly()
    {
        MiniMaxH3Config config = new();
        Tensor latent = RampLatent(channels: config.LatentsDim, height: 8, width: 6);
        using (latent)
        {
            using Tensor rows = MiniMaxH3Latents.PackVideo(latent, config);
            Assert.Equal(4L * 3, rows.Shape[0]);
            Assert.Equal(96L, rows.Shape[1]);
            using Tensor back = MiniMaxH3Latents.UnpackVideo(rows, 1, 8, 6, config);

            Assert.Equal(latent.AsSpan<float>().ToArray(), back.AsSpan<float>().ToArray());
        }
    }

    [Fact]
    public void PackedToken_LaysOutEachPatchChannelMajor()
    {
        MiniMaxH3Config config = new();
        Tensor latent = RampLatent(config.LatentsDim, height: 8, width: 6);
        using (latent)
        {
            using Tensor rows = MiniMaxH3Latents.PackVideo(latent, config);
            // Token (y=1, x=2) of a 4 x 3 token grid is row 1 * 3 + 2 = 5. Element (channel 3, py 1, px 0) is index 14.
            int row = 1 * 3 + 2;
            int element = ((3 * 1 + 0) * 2 + 1) * 2 + 0;
            float packed = rows.AsSpan<float>()[row * 96 + element];
            float source = latent.AsSpan<float>()[(3 * 8 + (1 * 2 + 1)) * 6 + (2 * 2 + 0)];
            Assert.Equal(source, packed);
        }
    }

    // ---- Prompt presentation ------------------------------------------------------------------------------------

    [Fact]
    public void PlainPrompt_EncodesTheRawSegmentWithoutSpecialTokens()
    {
        MiniMaxH3TextEncoding.Encoded built = MiniMaxH3TextEncoding.Build(CharIds, "a cat", conditions: null);

        Assert.Equal(new[] { 97, 32, 99, 97, 116 }, built.TokenIds);
        Assert.All(built.ModalityTags, tag => Assert.Equal(MiniMaxH3TextEncoding.TextTag, tag));
        Assert.Equal(new[] { "a cat" }, built.TextSegments);
    }

    [Fact]
    public void EmptyPrompt_EncodesOneEndOfTextToken()
    {
        MiniMaxH3TextEncoding.Encoded built = MiniMaxH3TextEncoding.Build(CharIds, "", conditions: null);

        Assert.Equal(new[] { Qwen2Tokenizer.EndOfTextId }, built.TokenIds);
    }

    [Fact]
    public void TextToImage_HasNoPictureLabelOrVisionBlock()
    {
        MiniMaxH3TextEncoding.Encoded built = MiniMaxH3TextEncoding.Build(CharIds, "a red fox", conditions: null);

        Assert.DoesNotContain(built.TokenIds, id => id == Qwen2Tokenizer.VisionStartId);
        Assert.DoesNotContain(built.TokenIds, id => id == Qwen2Tokenizer.ImagePadId);
        Assert.Single(built.TextSegments);
        Assert.Empty(built.VisionBlockTokenCounts);
    }

    [Fact]
    public void EmbeddedTokenizer_EncodesPlainTextWithoutAnySpecialToken()
    {
        Qwen2Tokenizer tokenizer = new();

        MiniMaxH3TextEncoding.Encoded built = MiniMaxH3TextEncoding.Build(tokenizer, "a red fox jumps", conditions: null);

        Assert.NotEmpty(built.TokenIds);
        Assert.All(built.TokenIds, id => Assert.True(id < Qwen2Tokenizer.EndOfTextId, $"special id {id} in a plain prompt"));
    }

    [Fact]
    public void PromptWeightSyntax_ProducesPerTokenWeightsOnlyWhenPresent()
    {
        MiniMaxH3TextEncoding.Encoded plain = MiniMaxH3TextEncoding.Build(CharIds, "a fox", conditions: null);
        MiniMaxH3TextEncoding.Encoded weighted = MiniMaxH3TextEncoding.Build(CharIds, "a (fox:1.5)", conditions: null);

        Assert.Null(plain.PromptWeights);
        Assert.NotNull(weighted.PromptWeights);
        Assert.Equal(weighted.TokenIds.Length, weighted.PromptWeights!.Length);
        Assert.Contains(weighted.PromptWeights, w => Math.Abs(w - 1.5f) < 1e-6f);
    }

    // ---- Single-frame (2D) image VAE ----------------------------------------------------------------------------

    [Fact]
    public void ImageVaeConfig_IsOneTemporalPatchTwoRopeAxesAndSevenHundredSixtyEightPatchDims()
    {
        MiniMaxH3VideoVaeConfig config = MiniMaxH3VideoVaeConfig.ImageBaseline;

        Assert.True(config.Decoder2D);
        Assert.Equal(1, config.TemporalPatch);
        Assert.Equal(2, config.RopeAxes);
        Assert.Equal(768, config.PatchDim);
    }

    [Fact]
    public void ImageVaeTiles_AtOneThousandTwentyFourPixelsGiveFiveTilesWithSixtyFourPixelOverlap()
    {
        // ComfyUI vae.py split_tiles: 256-pixel tiles, at least 64 pixels of overlap, extra overlap in 16-pixel steps.
        (int[] start, int[] length, int[] overlap) = MiniMaxH3VideoVaeConfig.ImageBaseline.SplitTiles(1024);

        Assert.Equal([0, 192, 384, 576, 768], start);
        Assert.Equal([256, 256, 256, 256, 256], length);
        Assert.Equal([64, 64, 64, 64], overlap);
    }

    [Fact]
    public void ImageVaeTiles_AtTwoThousandFortyEightPixelsGiveElevenTilesWithTheExtraOverlapOnTheFirstEightGaps()
    {
        (int[] start, int[] length, int[] overlap) = MiniMaxH3VideoVaeConfig.ImageBaseline.SplitTiles(2048);

        Assert.Equal([0, 176, 352, 528, 704, 880, 1056, 1232, 1408, 1600, 1792], start);
        Assert.Equal(Enumerable.Repeat(256, 11), length);
        Assert.Equal([80, 80, 80, 80, 80, 80, 80, 80, 64, 64], overlap);
    }

    [Fact]
    public void ImageVaeTiles_AtTwoFiftySixPixelsNoTilingHappens()
    {
        (int[] start, int[] length, int[] overlap) = MiniMaxH3VideoVaeConfig.ImageBaseline.SplitTiles(256);

        Assert.Equal([0], start);
        Assert.Equal([256], length);
        Assert.Empty(overlap);
    }

    [Fact]
    public void SingleFrameDecoder_RefusesAnyLatentWithMoreThanOneFrame()
    {
        MiniMaxH3VideoVaeDecoder decoder = new(MiniMaxH3VideoVaeConfig.ImageBaseline);
        using CpuBackend backend = new();
        using Tensor twoFrames = new(new TensorShape([1L, 24L, 2L, 4L, 4L]), DType.F32);

        Assert.Throws<ArgumentException>(() => decoder.Decode(backend, twoFrames));
    }

    private static readonly Func<string, IReadOnlyList<int>> CharIds =
        text => text.Select(c => (int)c).ToArray();

    private static double[] Axis(int dim, double sqrtArea)
    {
        double ratio = dim / sqrtArea;
        int n = dim / 2;
        double[] axis = new double[n];
        for (int i = 0; i < n; i++)
        {
            axis[i] = (i * (ratio / n) + (1.0 - ratio) / 2.0) * 32.0;
        }
        return axis;
    }

    private static (Tensor Weight, Tensor Bias) StackedHead(int outRows, int heads, int inDim)
    {
        Tensor weight = new(new TensorShape(heads * outRows, inDim), DType.F32);
        Tensor bias = new(new TensorShape(heads * outRows), DType.F32);
        Span<float> w = weight.AsSpan<float>();
        Span<float> b = bias.AsSpan<float>();
        for (int i = 0; i < w.Length; i++)
        {
            w[i] = (i % 7) * 0.25f - 0.5f;
        }
        for (int i = 0; i < b.Length; i++)
        {
            b[i] = (i % 5) * 0.125f;
        }
        return (weight, bias);
    }

    private static Tensor RampLatent(int channels, int height, int width)
    {
        Tensor latent = new(new TensorShape([1L, channels, 1L, height, width]), DType.F32);
        Span<float> span = latent.AsSpan<float>();
        for (int i = 0; i < span.Length; i++)
        {
            span[i] = i * 0.001f - 3f;
        }
        return latent;
    }
}
