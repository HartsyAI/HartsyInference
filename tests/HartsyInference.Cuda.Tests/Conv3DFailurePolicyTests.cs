using Xunit;

namespace HartsyInference.Cuda.Tests;

/// <summary>The session-disable decision for a failed native 3-D convolution. CPU-only: the policy touches no driver state.</summary>
public sealed class Conv3DFailurePolicyTests
{
    [Fact]
    public void DeterministicFailure_DisablesSession()
        => Assert.True(Conv3DFailurePolicy.DisablesSession(oomRetriesInFailingStage: 0));

    [Fact]
    public void OomRetryInFailingStage_DoesNotDisableSession()
    {
        Assert.False(Conv3DFailurePolicy.DisablesSession(oomRetriesInFailingStage: 1));
        Assert.False(Conv3DFailurePolicy.DisablesSession(oomRetriesInFailingStage: 7));
    }

    [Fact]
    public void OomRetryInEarlierStage_DoesNotExcuseLaterDeterministicFailure()
    {
        long retries = 0;
        Conv3DStageTracker stage = new Conv3DStageTracker(() => retries);
        stage.Enter("output alloc");
        retries++;                       // allocator retry inside output alloc
        stage.Enter("channels-last conv");
        Assert.True(stage.DisablesSession);
        Assert.Equal("channels-last conv", stage.Name);
    }

    [Fact]
    public void OomRetryInSameStage_DoesNotDisableSession()
    {
        long retries = 0;
        Conv3DStageTracker stage = new Conv3DStageTracker(() => retries);
        stage.Enter("output alloc");
        retries++;                       // allocator retry inside the failing stage
        Assert.False(stage.DisablesSession);
    }
}
