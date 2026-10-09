using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using Xunit;

namespace HartsyInference.LLM.Tests;

/// <summary>The slot lease that lets a scheduled request run without the slot lock: a model cannot be freed or replaced while a lease is held, and the wait for
/// leases is bounded. The scheduled route's routing rule (what must stay on the pipeline) is covered here too.</summary>
public sealed class TextDeviceSlotLeaseTests
{
    [Fact]
    public void WaitForLeases_TimesOut_WhileALeaseIsHeld_AndReturnsOnceReleased()
    {
        TextDeviceSlot slot = new();
        slot.EnterLease();
        Assert.True(slot.HasLeases);
        Assert.False(slot.WaitForLeases(TimeSpan.FromMilliseconds(50)));

        slot.ExitLease();
        Assert.False(slot.HasLeases);
        Assert.True(slot.WaitForLeases(TimeSpan.Zero));
    }

    [Fact]
    public async Task WaitForLeases_ReturnsOnlyOnceEveryLeaseIsReleased()
    {
        TextDeviceSlot slot = new();
        slot.EnterLease();
        slot.EnterLease();
        Task<bool> waiter = Task.Run(() => slot.WaitForLeases(TimeSpan.FromSeconds(30)));

        // One lease is still held, so the waiter cannot have returned whatever the timing.
        slot.ExitLease();
        Assert.False(waiter.IsCompleted);

        slot.ExitLease();
        Assert.True(await waiter.WaitAsync(TimeSpan.FromSeconds(30)));
    }

    [Fact]
    public void ScheduledRoute_KeepsPrefixCache_AlwaysFreeMemory_AndImages_OnThePipeline()
    {
        TextRequest plain = new() { Messages = [] };
        Assert.True(TextService.ScheduledRouteAllowed(plain, hasImage: false));
        Assert.False(TextService.ScheduledRouteAllowed(plain with { PrefixCacheKey = "conversation-1" }, hasImage: false));
        Assert.False(TextService.ScheduledRouteAllowed(plain with { AlwaysFreeMemory = true }, hasImage: false));
        Assert.False(TextService.ScheduledRouteAllowed(plain, hasImage: true));
    }
}
