using HartsyInference.Core.MemoryManagement;
using Xunit;

namespace HartsyInference.Core.Tests.MemoryManagement;

/// <summary>Covers the scope that makes a per-request override reach the levers decided mid-generation, and — more importantly — that it never outlives the request that set it.</summary>
public sealed class VramPolicyScopeTests
{
    [Fact]
    public void OutsideAGeneration_NothingIsCurrent()
        => Assert.Null(VramPolicyScope.Current);

    [Fact]
    public void ScopeIsVisibleWhileOpenAndGoneAfter()
    {
        VramPolicy policy = VramPolicy.For(VramTier.Aggressive);
        using (VramPolicyScope.Push(policy))
        {
            Assert.Same(policy, VramPolicyScope.Current);
        }
        Assert.Null(VramPolicyScope.Current);
    }

    /// <summary>A leaked scope would quietly govern every later generation on the flow, so restoration has to be
    /// exact rather than just "cleared".</summary>
    [Fact]
    public void NestedScopesRestoreTheirParent()
    {
        VramPolicy outer = VramPolicy.For(VramTier.Balanced);
        VramPolicy inner = VramPolicy.For(VramTier.Maximum);
        using (VramPolicyScope.Push(outer))
        {
            Assert.Same(outer, VramPolicyScope.Current);
            using (VramPolicyScope.Push(inner))
            {
                Assert.Same(inner, VramPolicyScope.Current);
            }
            Assert.Same(outer, VramPolicyScope.Current);
        }
        Assert.Null(VramPolicyScope.Current);
    }

    /// <summary>A request that overrode nothing pushes null, which must LEAVE the enclosing policy alone rather
    /// than blanking it — the refiner and segment passes nest inside the image generation's scope.</summary>
    [Fact]
    public void PushingNullKeepsTheEnclosingScope()
    {
        VramPolicy outer = VramPolicy.For(VramTier.Aggressive);
        using (VramPolicyScope.Push(outer))
        {
            using (VramPolicyScope.Push(null))
            {
                Assert.Same(outer, VramPolicyScope.Current);
            }
            Assert.Same(outer, VramPolicyScope.Current);
        }
    }

    [Fact]
    public void DisposingTwiceIsHarmless()
    {
        VramPolicy policy = VramPolicy.For(VramTier.Balanced);
        IDisposable scope = VramPolicyScope.Push(policy);
        scope.Dispose();
        scope.Dispose();
        Assert.Null(VramPolicyScope.Current);
    }

    /// <summary>The scope must beat the per-backend registry — that is the whole point — but only while it is open.</summary>
    [Fact]
    public void ScopeWinsOverTheBackendRegistryThenYieldsBack()
    {
        string? previous = Environment.GetEnvironmentVariable(LowVramPolicy.EnvironmentVariable);
        try
        {
            Environment.SetEnvironmentVariable(LowVramPolicy.EnvironmentVariable, null);
            LowVramPolicy.ResetCacheForTests();
            using RecordingStreamingBackend backend = new RecordingStreamingBackend(cache: null);
            VramPolicyRegistry.Set(backend, VramPolicy.For(VramTier.Performance));
            Assert.Equal(VramTier.Performance, VramPolicyRegistry.Resolve(backend).Tier);

            using (VramPolicyScope.Push(VramPolicy.For(VramTier.Maximum)))
            {
                Assert.Equal(VramTier.Maximum, VramPolicyRegistry.Resolve(backend).Tier);
                // And the legacy bridge every un-migrated call site still uses must agree.
                Assert.Equal(LowVramMode.ForceOn, LowVramPolicy.Resolve(backend));
            }
            Assert.Equal(VramTier.Performance, VramPolicyRegistry.Resolve(backend).Tier);
            VramPolicyRegistry.Clear(backend);
        }
        finally
        {
            Environment.SetEnvironmentVariable(LowVramPolicy.EnvironmentVariable, previous);
            LowVramPolicy.ResetCacheForTests();
        }
    }

    /// <summary>Concurrent generations on different devices must not see each other's override — the same
    /// last-writer-wins defect the per-backend registry exists to avoid.</summary>
    [Fact]
    public async Task ConcurrentFlowsDoNotSeeEachOther()
    {
        VramPolicy a = VramPolicy.For(VramTier.Aggressive);
        VramPolicy b = VramPolicy.For(VramTier.Performance);
        using ManualResetEventSlim bothInside = new ManualResetEventSlim(false);
        int arrived = 0;

        async Task<VramTier?> Run(VramPolicy policy)
        {
            using (VramPolicyScope.Push(policy))
            {
                if (Interlocked.Increment(ref arrived) == 2) bothInside.Set();
                bothInside.Wait(TimeSpan.FromSeconds(5));
                await Task.Yield();
                return VramPolicyScope.Current?.Tier;
            }
        }

        Task<VramTier?> ta = Task.Run(() => Run(a));
        Task<VramTier?> tb = Task.Run(() => Run(b));
        VramTier?[] results = await Task.WhenAll(ta, tb);

        Assert.Equal(VramTier.Aggressive, results[0]);
        Assert.Equal(VramTier.Performance, results[1]);
        Assert.Null(VramPolicyScope.Current);
    }
}
