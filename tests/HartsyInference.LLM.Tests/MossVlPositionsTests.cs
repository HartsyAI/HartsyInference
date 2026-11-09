using HartsyInference.Core.Exceptions;
using HartsyInference.LLM.Models.MossVl;

namespace HartsyInference.LLM.Tests;

public sealed class MossVlPositionsTests
{
    private const int ImageToken = 151655;
    private static readonly int[] Prompt = [151644, 872, 198, 151652, 151655, 151653, 9707, 1879];

    [Fact]
    public void TextAfterTheImageResumesPastItsGridAndTheSeparatorSitsAfterTheGrid()
    {
        MossVlPositionPlan plan = MossVlPositions.Plan(Prompt, ImageToken, [new MossVlImage(8, 12)]);

        // Regular tokens count up to the image at 4; the image sits at 4 + max(8, 12) = 16; the text after it starts at 17.
        int[] expectedText = [0, 1, 2, 3, 16, 17, 18, 19];
        for (int axis = 0; axis < 3; axis++)
            for (int i = 0; i < Prompt.Length; i++)
                Assert.Equal(expectedText[i], plan.Text[axis * Prompt.Length + i]);

        Assert.Equal(8 * 12 + 1, plan.VisionLength);
        int l = plan.VisionLength;
        Assert.Equal((4, 4, 4), (plan.Vision[0], plan.Vision[l], plan.Vision[2 * l]));           // grid (0, 0)
        Assert.Equal((4, 4, 15), (plan.Vision[11], plan.Vision[l + 11], plan.Vision[2 * l + 11])); // grid (0, 11)
        Assert.Equal((4, 5, 4), (plan.Vision[12], plan.Vision[l + 12], plan.Vision[2 * l + 12])); // grid (1, 0)
        Assert.Equal((4, 11, 15), (plan.Vision[95], plan.Vision[l + 95], plan.Vision[2 * l + 95])); // grid (7, 11)
        Assert.Equal((16, 16, 16), (plan.Vision[96], plan.Vision[l + 96], plan.Vision[2 * l + 96])); // separator
    }

    [Fact]
    public void RowsBeforeTheImageSeeNoImage()
    {
        MossVlPositionPlan plan = MossVlPositions.Plan(Prompt, ImageToken, [new MossVlImage(8, 12)]);

        Assert.Equal(new float[] { 0f, 0f, 0f, 0f, 1f, 1f, 1f, 1f }, plan.RowMask);
    }

    [Fact]
    public void TextWithoutAnImageHasNoRowMask()
    {
        MossVlPositionPlan plan = MossVlPositions.Plan([1, 2, 3], ImageToken, []);

        Assert.Null(plan.RowMask);
        Assert.Equal(0, plan.VisionLength);
    }

    [Fact]
    public void PlaceholderCountMustMatchTheImages()
    {
        Assert.Throws<HartsyInferenceException>(() => MossVlPositions.Plan(Prompt, ImageToken, []));
    }

    [Fact]
    public void SeveralImagesAreRefused()
    {
        int[] twoImages = [ImageToken, 5, ImageToken];
        Assert.Throws<HartsyInferenceException>(() =>
            MossVlPositions.Plan(twoImages, ImageToken, [new MossVlImage(2, 2), new MossVlImage(2, 2)]));
    }

    [Fact]
    public void InterleavedMRopeReadsEachAxisForItsPairs()
    {
        // One row whose temporal, height and width positions differ, so each rotary pair's angle names its axis.
        MossVlPositionPlan plan = new()
        {
            Text = [1, 10, 100],
            Vision = [0, 0, 0],
            VisionLength = 1,
            RowMask = null,
        };
        const int headDim = 128;
        const double theta = 5_000_000.0;
        using MossVlRope rope = MossVlPositions.BuildRope(plan, headDim, theta, [24, 20, 20], interleaved: true, hiddenSize: 4096);

        ReadOnlySpan<float> cos = rope.Rope.TextCos.AsSpan<float>();
        for (int j = 0; j < headDim / 2; j++)
        {
            int axisPosition = j < 60 && j % 3 == 1 ? 10 : j < 60 && j % 3 == 2 ? 100 : 1;
            double inv = 1.0 / Math.Pow(theta, 2.0 * j / headDim);
            float expected = (float)Math.Cos(axisPosition * inv);
            Assert.True(Math.Abs(expected - cos[j]) < 1e-5f, $"pair {j}: expected {expected}, got {cos[j]}");
        }
    }
}
