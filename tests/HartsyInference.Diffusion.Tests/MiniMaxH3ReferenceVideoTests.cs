using Xunit;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Gates for MiniMax-H3's ref2va reference-video path. Every case here fails as silently wrong conditioning
/// rather than as a crash: the frame-stack preprocessor reading the same plane twice, a small clip upscaled onto a
/// bigger canvas, a frame count snapped the wrong way, or timestamps taken from the source index instead of the
/// sampled one. All are weight-free and resolution-only, so they need neither a GPU nor a checkpoint.</summary>
public sealed class MiniMaxH3ReferenceVideoTests
{
    private static Qwen3VlVisionConfig TinyVision => new()
    {
        Depth = 2,
        HiddenSize = 16,
        NumHeads = 2,
        IntermediateSize = 32,
        InChannels = 3,
        PatchSize = 16,
        SpatialMergeSize = 2,
        TemporalPatchSize = 2,
        OutHiddenSize = 24,
        NumPositionEmbeddings = 16,
        DeepstackVisualIndexes = [1],
        NormEps = 1e-6f,
    };

    private static unsafe Tensor Frame(int h, int w, int seed)
    {
        Tensor t = new Tensor(new TensorShape(3, h, w), DType.F32);
        Random rng = new Random(seed);
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++)
        {
            p[i] = (float)rng.NextDouble();
        }
        return t;
    }

    /// <summary>A video block must cost exactly what a still image of the same size costs — that is the whole reason
    /// the vision tower needs no change, and it is what makes the presentation's reserved token count correct.</summary>
    [Fact]
    public void FrameStackMatchesStillImageGeometry()
    {
        Qwen3VlImageProcessor proc = new Qwen3VlImageProcessor(TinyVision, maxPixels: 64 * 64);
        using Tensor a = Frame(64, 64, 3);
        using Tensor b = Frame(64, 64, 4);

        (Tensor still, int stillT, int stillH, int stillW) = proc.Preprocess(a);
        (Tensor stack, int stackT, int stackH, int stackW) = proc.Preprocess([a, b]);

        Assert.Equal(1, stillT);
        Assert.Equal(1, stackT);
        Assert.Equal(stillH, stackH);
        Assert.Equal(stillW, stackW);
        Assert.Equal(still.Shape[0], stack.Shape[0]);
        Assert.Equal(still.Shape[1], stack.Shape[1]);

        still.Dispose();
        stack.Dispose();
    }

    /// <summary>The point of the frame-stack overload: the temporal counter must actually index the frame. A duplicated
    /// frame has to reproduce the single-frame payload byte for byte, and two distinct frames must not.</summary>
    [Fact]
    public unsafe void DistinctFramesDifferFromADuplicatedOne()
    {
        Qwen3VlImageProcessor proc = new Qwen3VlImageProcessor(TinyVision, maxPixels: 64 * 64);
        using Tensor a = Frame(64, 64, 11);
        using Tensor b = Frame(64, 64, 12);

        (Tensor still, _, _, _) = proc.Preprocess(a);
        (Tensor duplicated, _, _, _) = proc.Preprocess([a, a]);
        (Tensor distinct, _, _, _) = proc.Preprocess([a, b]);

        float* stillP = (float*)still.DataPointer;
        float* dupP = (float*)duplicated.DataPointer;
        float* distinctP = (float*)distinct.DataPointer;

        bool duplicatedMatchesStill = true;
        bool distinctMatchesStill = true;
        for (long i = 0; i < still.ElementCount; i++)
        {
            if (dupP[i] != stillP[i])
            {
                duplicatedMatchesStill = false;
            }
            if (distinctP[i] != stillP[i])
            {
                distinctMatchesStill = false;
            }
        }

        Assert.True(duplicatedMatchesStill, "A duplicated frame must reproduce the still-image payload exactly.");
        Assert.False(distinctMatchesStill, "Two distinct frames must not collapse to the still-image payload.");

        still.Dispose();
        duplicated.Dispose();
        distinct.Dispose();
    }

    [Fact]
    public void FrameStackRejectsWrongFrameCount()
    {
        Qwen3VlImageProcessor proc = new Qwen3VlImageProcessor(TinyVision, maxPixels: 64 * 64);
        using Tensor a = Frame(64, 64, 21);
        Assert.Throws<ArgumentException>(() => proc.Preprocess([a]));
        Assert.Throws<ArgumentException>(() => proc.Preprocess([a, a, a]));
    }

    [Fact]
    public void FrameStackRejectsMismatchedFrames()
    {
        Qwen3VlImageProcessor proc = new Qwen3VlImageProcessor(TinyVision, maxPixels: 64 * 64);
        using Tensor a = Frame(64, 64, 31);
        using Tensor b = Frame(32, 64, 32);
        Assert.Throws<ArgumentException>(() => proc.Preprocess([a, b]));
    }

    /// <summary>A reference smaller than the adapted canvas must keep its own size — upscaling a small clip into a
    /// 768-short-edge canvas would spend tokens inventing detail that is not there.</summary>
    [Fact]
    public void SmallClipIsNeverUpscaledOntoTheCanvas()
    {
        Assert.Equal((768, 768), MiniMaxH3Geometry.AdaptCanvas(256, 256));
        Assert.Equal((256, 256), MiniMaxH3Geometry.RefVideoCanvas(256, 256));
    }

    /// <summary>The only branch where the shrink guard stays out of the way: a clip larger than the canvas takes the
    /// canvas, area-capped and rounded to 32.</summary>
    [Fact]
    public void LargeClipTakesTheAdaptedCanvas()
    {
        Assert.Equal((1344, 768), MiniMaxH3Geometry.AdaptCanvas(1920, 1080));
        Assert.Equal((1344, 768), MiniMaxH3Geometry.RefVideoCanvas(1920, 1080));
    }

    [Theory]
    [InlineData(5, 5)]
    [InlineData(21, 5)]
    [InlineData(22, 22)]
    [InlineData(30, 22)]
    [InlineData(124, 124)]
    [InlineData(130, 124)]
    public void ReferenceFrameCountSnapsDownOntoTheGrid(int frames, int expected)
    {
        Assert.Equal(expected, MiniMaxH3Geometry.SnapFrameCountDown(frames));
        Assert.Equal(5, expected % 17);
    }

    [Fact]
    public void FewerThanFiveFramesIsAHardError()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => MiniMaxH3Geometry.SnapFrameCountDown(4));
    }

    [Theory]
    [InlineData(5, 1)]
    [InlineData(22, 2)]
    [InlineData(39, 4)]
    [InlineData(56, 5)]
    [InlineData(124, 11)]
    public void QwenFramesAreSampledAtTwoFps(int frames, int expectedSamples)
    {
        IReadOnlyList<int> sampled = MiniMaxH3Geometry.RefVideoSampleIndices(frames);
        Assert.Equal(expectedSamples, sampled.Count);
        for (int i = 0; i < sampled.Count; i++)
        {
            Assert.Equal(i * (MiniMaxH3Geometry.Fps / 2), sampled[i]);
        }
    }

    /// <summary>Block count and labels must follow the <b>sampled</b> index, not the source frame index: an odd sample
    /// count repeat-pads its last frame into a final block rather than dropping it.</summary>
    [Theory]
    [InlineData(5, 1)]
    [InlineData(22, 1)]
    [InlineData(56, 3)]
    [InlineData(124, 6)]
    public void VideoBlocksFollowTheSampledIndex(int frames, int expectedBlocks)
    {
        IReadOnlyList<int> sampled = MiniMaxH3Geometry.RefVideoSampleIndices(frames);
        MiniMaxH3TextEncoding.Condition condition = MiniMaxH3TextEncoding.Video(sampled.Count, 4);

        Assert.Equal(expectedBlocks, condition.Blocks.Count);
        Assert.Equal(sampled.Count + (sampled.Count % 2), condition.Blocks.Count * 2);
        for (int i = 0; i < condition.Blocks.Count; i++)
        {
            // Sampled index i sits at i/2 seconds, so a pair's mean label lands on 0.25, 1.25, 2.25, ...
            double expected = i == condition.Blocks.Count - 1 && sampled.Count % 2 == 1
                ? sampled.Count - 1
                : i * 2 + 0.5;
            Assert.Equal(expected / 2.0, condition.Blocks[i].TimestampSeconds!.Value, 6);
        }
    }

    /// <summary>The presentation a soundtracked reference video produces: its <c>&lt;Audio j&gt;</c> label is emitted
    /// before its <c>&lt;Video k&gt;</c>, so the audio ordinal increments first.</summary>
    [Fact]
    public void SoundtrackLabelPrecedesItsVideoLabel()
    {
        MiniMaxH3TextEncoding.Encoded encoded = MiniMaxH3TextEncoding.Build(
            text => text.Select(c => (int)c).ToArray(),
            "prompt",
            [
                MiniMaxH3TextEncoding.Audio(),
                MiniMaxH3TextEncoding.Video(2, 4),
                MiniMaxH3TextEncoding.Audio(),
            ]);

        List<string> labels = [.. encoded.TextSegments.Where(s => s.StartsWith('<'))];
        Assert.Equal("<Audio 1>: ", labels[0]);
        Assert.Equal("<Video 1>: ", labels[1]);
        Assert.Equal("<0.2 seconds>", labels[2]);
        Assert.Equal("<Audio 2>: ", labels[3]);
    }
}
