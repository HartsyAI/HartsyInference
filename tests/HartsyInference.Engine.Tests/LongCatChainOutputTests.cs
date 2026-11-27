using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Pipelines.LongCat;
using HartsyInference.Engine.Recipes.Video;
using Xunit;

namespace HartsyInference.Engine.Tests;

/// <summary>The recipe-side chain output: each segment's decoded frames are kept (segment 0 whole, a later segment after its 13-frame context), its
/// last frames are re-encoded only when another segment follows, and the kept frames add up to the plan's pixel total. Tiny numbers: a frame is one
/// byte, so identity checks are reference checks.</summary>
public sealed class LongCatChainOutputTests
{
    private const int ContextLatents = 4, ContextPixels = 13, SegmentLatents = 7;

    [Fact]
    public void TheKeptFramesAddUpToThePlanTotalAndKeepEachSegmentsNewFrames()
    {
        LongCatAvatarChainPlan plan = LongCatAvatarChainPlan.For(SegmentLatents, 45);
        Assert.Equal(3, plan.Segments);
        int decodedPerSegment = 1 + 4 * (SegmentLatents - 1);
        LongCatChainFrames output = new(ContextPixels);
        List<byte[][]> decodes = [];
        int encodes = 0;
        for (int s = 0; s < plan.Segments; s++)
        {
            byte[][] decoded = Decoded(s, decodedPerSegment);
            decodes.Add(decoded);
            bool last = s == plan.Segments - 1;
            float[] context = output.Step(decoded, last, frames =>
            {
                encodes++;
                Assert.Equal(decodedPerSegment, frames.Length);
                return new float[ContextLatents];
            });
            Assert.Equal(last ? 0 : ContextLatents, context.Length);
        }

        Assert.Equal(plan.Segments, output.Segments);
        Assert.Equal(plan.TotalPixelFrames, output.Frames.Length);
        Assert.Equal(plan.Segments - 1, encodes);

        byte[][] kept = output.Frames;
        for (int i = 0; i < decodedPerSegment; i++)
            Assert.Same(decodes[0][i], kept[i]);
        int at = decodedPerSegment;
        for (int s = 1; s < plan.Segments; s++)
            for (int i = ContextPixels; i < decodedPerSegment; i++, at++)
                Assert.Same(decodes[s][i], kept[at]);
        Assert.Equal(at, kept.Length);
    }

    [Fact]
    public void TheLastSegmentEncodesNothingAndAloneKeepsEveryFrame()
    {
        LongCatChainFrames output = new(ContextPixels);
        byte[][] only = Decoded(0, 25);
        float[] context = output.Step(only, last: true, _ => throw new InvalidOperationException("The last segment must not be encoded."));
        Assert.Empty(context);
        Assert.Equal(1, output.Segments);
        Assert.Equal(25, output.Frames.Length);
    }

    [Fact]
    public void ANonLastSegmentEncodesItsDecodedFramesAndReturnsTheResult()
    {
        LongCatChainFrames output = new(ContextPixels);
        byte[][] decoded = Decoded(0, 25);
        float[] expected = [1f, 2f, 3f];
        byte[][]? seen = null;
        float[] context = output.Step(decoded, last: false, frames =>
        {
            seen = frames;
            return expected;
        });
        Assert.Same(decoded, seen);
        Assert.Same(expected, context);
    }

    [Fact]
    public void ALaterSegmentShorterThanItsContextIsRefused()
    {
        LongCatChainFrames output = new(ContextPixels);
        output.Step(Decoded(0, 25), last: false, _ => new float[1]);
        Assert.Throws<InvalidOperationException>(() => output.Step(Decoded(1, 12), last: true, _ => new float[1]));
    }

    [Fact]
    public void TheFramesReadBackAsACopy()
    {
        LongCatChainFrames output = new(ContextPixels);
        output.Step(Decoded(0, 25), last: true, _ => new float[1]);
        byte[][] first = output.Frames;
        first[0] = [];
        Assert.NotEmpty(output.Frames[0]);
    }

    private static byte[][] Decoded(int segment, int frames)
    {
        byte[][] result = new byte[frames][];
        for (int i = 0; i < frames; i++)
            result[i] = [(byte)(segment * 50 + i)];
        return result;
    }
}
