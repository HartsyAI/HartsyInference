using HartsyInference.Diffusion.Models.Denoisers;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Head selection on the published six-step grid. The expected weights are written out from ComfyUI's
/// <c>_pdd_head</c> arithmetic: block 0 is the base, block n carries dt[n] / (sum of dt over the span).</summary>
public sealed class StackedOutputHeadsTests
{
    private static readonly double[] Grid =
    [
        1.0, 0.9675236940383911, 0.8875485062599182, 0.75, 0.5327739119529724, 0.23200887441635132, 0.0,
    ];

    private static double Dt(int k) => Grid[k] - Grid[k + 1];

    [Fact]
    public void SingleIntervalStepUsesItsOwnHeadOnly()
    {
        StackedOutputHeads heads = new(Grid);

        for (int n = 1; n <= 5; n++)
        {
            OutputHeadStep step = heads.Resolve(Grid[n], Grid[n + 1]);

            Assert.Equal(n, step.Start);
            Assert.Equal(n + 1, step.Stop);
            Assert.Equal(n, step.FirstOffset);
            Assert.Equal(1.0, step.Weights[n], 12);
            for (int k = 0; k < heads.HeadCount; k++)
            {
                if (k != n)
                {
                    Assert.Equal(0.0, step.Weights[k], 12);
                }
            }
        }
    }

    [Fact]
    public void FirstStepHasNoOffsets()
    {
        StackedOutputHeads heads = new(Grid);

        OutputHeadStep step = heads.Resolve(Grid[0], Grid[1]);

        Assert.Equal(0, step.Start);
        Assert.Equal(1, step.Stop);
        Assert.Equal(1, step.FirstOffset);
        Assert.All(step.Weights, weight => Assert.Equal(0.0, weight, 12));
    }

    [Fact]
    public void SpanOverTwoIntervalsWeightsEachByItsWidth()
    {
        StackedOutputHeads heads = new(Grid);

        OutputHeadStep step = heads.Resolve(Grid[2], Grid[4]);
        double total = Dt(2) + Dt(3);

        Assert.Equal(2, step.Start);
        Assert.Equal(4, step.Stop);
        Assert.Equal(Dt(2) / total, step.Weights[2], 12);
        Assert.Equal(Dt(3) / total, step.Weights[3], 12);
        Assert.Equal(1.0, step.Weights[2] + step.Weights[3], 12);
    }

    [Fact]
    public void SpanFromFirstIntervalKeepsTheFirstIntervalInTheTotal()
    {
        StackedOutputHeads heads = new(Grid);

        OutputHeadStep step = heads.Resolve(Grid[0], Grid[3]);
        double total = Dt(0) + Dt(1) + Dt(2);

        Assert.Equal(1, step.FirstOffset);
        Assert.Equal(0.0, step.Weights[0], 12);
        Assert.Equal(Dt(1) / total, step.Weights[1], 12);
        Assert.Equal(Dt(2) / total, step.Weights[2], 12);
        // Offsets are not renormalized: block 0's interval is in the total but carries no weight.
        Assert.Equal((Dt(1) + Dt(2)) / total, step.Weights[1] + step.Weights[2], 12);
        Assert.True(step.Weights[1] + step.Weights[2] < 1.0);
    }

    [Fact]
    public void LastStepEndsAtGridEnd()
    {
        StackedOutputHeads heads = new(Grid);

        OutputHeadStep step = heads.Resolve(Grid[5], 0.0);

        Assert.Equal(5, step.Start);
        Assert.Equal(6, step.Stop);
        Assert.Equal(1.0, step.Weights[5], 12);
    }

    [Fact]
    public void OffGridSigmasSnapToTheNearestPoint()
    {
        StackedOutputHeads heads = new(Grid);

        OutputHeadStep step = heads.Resolve(0.97, 0.87);

        Assert.Equal(1, step.Start);
        Assert.Equal(2, step.Stop);
        Assert.Equal(1.0, step.Weights[1], 12);
    }

    [Fact]
    public void SigmaAtZeroClampsTheSpanInsideTheGrid()
    {
        StackedOutputHeads heads = new(Grid);

        OutputHeadStep step = heads.Resolve(0.0, 0.0);

        Assert.Equal(5, step.Start);
        Assert.Equal(6, step.Stop);
        Assert.Equal(1.0, step.Weights[5], 12);
    }

    [Fact]
    public void NonDecreasingGridIsRefused()
    {
        Assert.Throws<ArgumentException>(() => new StackedOutputHeads([1.0, 0.5, 0.6, 0.0]));
    }
}
