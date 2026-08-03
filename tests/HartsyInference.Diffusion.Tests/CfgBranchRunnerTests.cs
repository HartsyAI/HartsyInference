using Xunit;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Utilities;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Unit coverage for <see cref="CfgBranchRunner.Run"/>: genuine concurrency (not just "runs both
/// eventually"), correct result pairing, and exception propagation from either side.</summary>
public sealed class CfgBranchRunnerTests
{
    private static unsafe Tensor Scalar(float value)
    {
        Tensor t = new(new TensorShape(1), DType.F32);
        *(float*)t.DataPointer = value;
        return t;
    }

    [Fact]
    public unsafe void Run_BothBranchesExecuteConcurrently()
    {
        // Each side blocks until the OTHER has started — this only rendezvous-succeeds if they actually
        // overlap. A sequential implementation (uncond only starting after cond returns) never has both sides
        // waiting at once, so the rendezvous times out and reachedBoth stays false.
        using Barrier barrier = new(2);
        bool condReachedRendezvous = false, uncondReachedRendezvous = false;
        (Tensor cond, Tensor uncond) = CfgBranchRunner.Run(
            () => { condReachedRendezvous = barrier.SignalAndWait(TimeSpan.FromSeconds(5)); return Scalar(1f); },
            () => { uncondReachedRendezvous = barrier.SignalAndWait(TimeSpan.FromSeconds(5)); return Scalar(2f); });

        Assert.True(condReachedRendezvous, "cond never rendezvoused with uncond — they did not run concurrently.");
        Assert.True(uncondReachedRendezvous, "uncond never rendezvoused with cond — they did not run concurrently.");
        Assert.Equal(1f, *(float*)cond.DataPointer);
        Assert.Equal(2f, *(float*)uncond.DataPointer);
    }

    [Fact]
    public unsafe void Run_ReturnsCorrectPairing()
    {
        (Tensor cond, Tensor uncond) = CfgBranchRunner.Run(() => Scalar(10f), () => Scalar(20f));
        Assert.Equal(10f, *(float*)cond.DataPointer);
        Assert.Equal(20f, *(float*)uncond.DataPointer);
    }

    [Fact]
    public void Run_CondThrows_PropagatesCondException_AndObservesUncondWithoutCrashing()
    {
        InvalidOperationException thrown = Assert.Throws<InvalidOperationException>(() =>
            CfgBranchRunner.Run(
                () => throw new InvalidOperationException("cond failed"),
                () => Scalar(1f)));
        Assert.Equal("cond failed", thrown.Message);
    }

    [Fact]
    public void Run_UncondThrows_PropagatesUncondException()
    {
        InvalidOperationException thrown = Assert.Throws<InvalidOperationException>(() =>
            CfgBranchRunner.Run(
                () => Scalar(1f),
                () => throw new InvalidOperationException("uncond failed")));
        Assert.Equal("uncond failed", thrown.Message);
    }
}
