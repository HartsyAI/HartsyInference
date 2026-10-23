using Xunit;

namespace HartsyInference.Cuda.Tests;

/// <summary>The session-disable decision for a failed native 3-D convolution. CPU-only: the policy touches no driver state.</summary>
public sealed class Conv3DFailurePolicyTests
{
    [Fact]
    public void DeterministicFailure_DisablesSession()
        => Assert.True(Conv3DFailurePolicy.DisablesSession(oomRetriesDuringCall: 0));

    [Fact]
    public void FailureAfterAllocatorOomRetry_DoesNotDisableSession()
    {
        Assert.False(Conv3DFailurePolicy.DisablesSession(oomRetriesDuringCall: 1));
        Assert.False(Conv3DFailurePolicy.DisablesSession(oomRetriesDuringCall: 7));
    }
}
