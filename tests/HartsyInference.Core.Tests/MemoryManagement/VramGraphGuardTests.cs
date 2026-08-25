using HartsyInference.Core.MemoryManagement;
using Xunit;

namespace HartsyInference.Core.Tests.MemoryManagement;

/// <summary>Pins the direction of each graph rule. Getting these backwards is silent: a skipped free reports success and OOMs later, and a moved allocation under a live graph corrupts a replay.</summary>
public sealed class VramGraphGuardTests
{
    [Fact]
    public void InvalidateBeforeRelease_ResetsAndDisownsTheGraph()
    {
        using RecordingStreamingBackend backend = new RecordingStreamingBackend(cache: null)
        {
            StepGraphReady = true,
            StepGraphOwner = new object(),
        };

        VramGraphGuard.InvalidateBeforeRelease(backend);

        Assert.False(backend.StepGraphReady);
        Assert.Null(backend.StepGraphOwner);
        Assert.Contains("graph-reset", backend.Calls);
    }

    /// <summary>A required release must never be skipped just because a graph is live — that is the failure mode the
    /// optional-lever rule would introduce if it were applied here.</summary>
    [Fact]
    public void InvalidateBeforeRelease_IsUnconditional()
    {
        using RecordingStreamingBackend backend = new RecordingStreamingBackend(cache: null) { StepGraphReady = false };
        VramGraphGuard.InvalidateBeforeRelease(backend);
        Assert.Contains("graph-reset", backend.Calls);
    }

    [Fact]
    public void CanMoveMemoryFreely_IsFalseWhileAGraphIsLive()
    {
        using RecordingStreamingBackend live = new RecordingStreamingBackend(cache: null) { StepGraphReady = true };
        Assert.False(VramGraphGuard.CanMoveMemoryFreely(live));

        using RecordingStreamingBackend owned = new RecordingStreamingBackend(cache: null) { StepGraphOwner = new object() };
        Assert.False(VramGraphGuard.CanMoveMemoryFreely(owned));

        using RecordingStreamingBackend idle = new RecordingStreamingBackend(cache: null);
        Assert.True(VramGraphGuard.CanMoveMemoryFreely(idle));
    }

    [Fact]
    public void NullBackend_IsTolerated()
    {
        VramGraphGuard.InvalidateBeforeRelease(null);
        Assert.True(VramGraphGuard.CanMoveMemoryFreely(null));
    }
}
