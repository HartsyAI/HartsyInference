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
}
