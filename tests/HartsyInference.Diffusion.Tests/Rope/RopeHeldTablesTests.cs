using HartsyInference.Core.Rope;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.Tests.Common;
using Xunit;

namespace HartsyInference.Diffusion.Tests.Rope;

/// <summary>A holder's Dispose must release every backend's lease even when one release throws, or the tables behind the
/// remaining leases are never freed; and a lookup after Dispose must not quietly hold a lease that nothing will release.</summary>
public sealed class RopeHeldTablesTests
{
    private static readonly RopeTableSettings Settings = new()
    {
        AngleOrder = RopeAngleOrder.MultiplyByInverse,
        Layout = RopeTableLayout.InterleavedPair,
        Preload = true,
        RetainedSignatures = 2,
    };

    [Fact]
    public void Dispose_ReleasesEveryLease_WhenOneReleaseThrows()
    {
        RopeFrequencyPlan plan = RopeFrequencyPlan.Create(new RopeSpec { AxisDims = [4, 4], Theta = 10000 });
        using RopeTableCache cache = new(plan, Settings);
        RopeHeldTables<int> held = new(cache, slot: 0, axes: 2);
        RopeRecordingBackend a = new();
        RopeRecordingBackend b = new();
        held.GetOrBuild(a, 1, static _ => [0f, 0f, 1f, 2f]);
        held.GetOrBuild(b, 1, static _ => [0f, 0f, 1f, 2f]);
        // Retires both backends' tables while the holder still leases them, so each lease's release frees its table.
        cache.Clear();
        a.FailFree = true;

        Exception? thrown = Record.Exception(held.Dispose);
        a.FailFree = false;   // lets the cache's own Dispose free a's table on its retry

        // a's free fails and is retried when b's lease goes; b's table is still freed, and the failure surfaces.
        Assert.NotNull(thrown);
        Assert.Contains("Injected free failure", thrown.ToString(), StringComparison.Ordinal);
        Assert.Contains("free-failed:2", a.Calls);
        Assert.Contains("free:2", b.Calls);
    }

    [Fact]
    public void LookupsAfterDispose_Throw()
    {
        RopeFrequencyPlan plan = RopeFrequencyPlan.Create(new RopeSpec { AxisDims = [4, 4], Theta = 10000 });
        using RopeTableCache cache = new(plan, Settings);
        RopeHeldTables<int> held = new(cache, slot: 0, axes: 2);
        RopeRecordingBackend backend = new();
        held.GetOrBuild(backend, 1, static _ => [0f, 0f, 1f, 2f]);

        held.Dispose();
        held.Dispose();

        Assert.Throws<ObjectDisposedException>(() => held.GetOrBuild(backend, 1, static _ => [0f, 0f, 1f, 2f]));
        Assert.Throws<ObjectDisposedException>(() => held.TryGet(backend, 1, out _));
        Assert.Throws<ObjectDisposedException>(() => held.GetOrBuildStaged(1, static _ => [0f, 0f, 1f, 2f]));
    }
}
