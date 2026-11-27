using HartsyInference.Vision.Segmentation.Sam3;
using Xunit;

namespace HartsyInference.Vision.Tests.Segmentation.Sam3;

public sealed class Sam3PostProcessorTests
{
    [Fact]
    public void ComputeScores_MultipliesQuerySigmoidByPresenceSigmoid()
    {
        float[] logits = [0f, 2f];
        float[] scores = new float[2];

        Sam3PostProcessor.ComputeScores(logits, presenceLogit: 0f, scores);

        Assert.Equal(0.25f, scores[0], 5);
        Assert.Equal(1f / (1f + MathF.Exp(-2f)) * 0.5f, scores[1], 5);
    }

    [Fact]
    public void ComputeScores_RejectsMismatchedBufferLength()
    {
        float[] logits = [0f, 1f];
        float[] scores = new float[3];

        Assert.Throws<ArgumentException>(() => Sam3PostProcessor.ComputeScores(logits, 0f, scores));
    }

    [Fact]
    public void SelectAbove_OrdersByScoreDescending()
    {
        float[] scores = [0.6f, 0.9f, 0.7f, 0.8f];

        int[] keep = Sam3PostProcessor.SelectAbove(scores, threshold: 0.5f);

        Assert.Equal([1, 3, 2, 0], keep);
    }

    [Fact]
    public void SelectAbove_ThresholdIsStrict()
    {
        float[] scores = [0.5f, 0.5000001f, 0.4f];

        int[] keep = Sam3PostProcessor.SelectAbove(scores, threshold: 0.5f);

        Assert.Equal([1], keep);
    }

    [Fact]
    public void SelectAbove_BreaksTiesByQueryIndex()
    {
        float[] scores = [0.8f, 0.8f, 0.9f, 0.8f];

        int[] keep = Sam3PostProcessor.SelectAbove(scores, threshold: 0.5f);

        Assert.Equal([2, 0, 1, 3], keep);
    }

    [Fact]
    public void SelectAbove_ReturnsEmptyWhenNothingPasses()
    {
        float[] scores = [0.1f, 0.2f];

        Assert.Empty(Sam3PostProcessor.SelectAbove(scores, threshold: 0.5f));
    }

    [Fact]
    public void CxcywhToXyxyPixels_ScalesNormalizedBoxToImageSize()
    {
        float[] boxes = [0.5f, 0.5f, 0.2f, 0.4f];
        float[] xyxy = new float[4];

        Sam3PostProcessor.CxcywhToXyxyPixels(boxes, width: 200, height: 100, xyxy);

        Assert.Equal(80f, xyxy[0], 4);
        Assert.Equal(30f, xyxy[1], 4);
        Assert.Equal(120f, xyxy[2], 4);
        Assert.Equal(70f, xyxy[3], 4);
    }

    [Fact]
    public void ThresholdLogits_MarksOnlyStrictlyPositiveLogits()
    {
        float[] logits = [-1f, 0f, 0.0001f, 3f];
        byte[] mask = new byte[4];

        Sam3PostProcessor.ThresholdLogits(logits, threshold: 0f, mask);

        Assert.Equal([0, 0, 1, 1], mask);
    }
}
