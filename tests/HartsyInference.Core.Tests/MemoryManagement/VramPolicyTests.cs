using HartsyInference.Core.Configuration;
using HartsyInference.Core.Backends;
using HartsyInference.Core.MemoryManagement;
using Xunit;

namespace HartsyInference.Core.Tests.MemoryManagement;

/// <summary>Covers the tier→lever expansion, the per-request override merge, and the legacy bridge.</summary>
/// <remarks>The whole resolver is pure, so every tier × lever combination is assertable on CPU with no GPU — which is
/// what makes "every combination is selectable" verifiable without a run per model.</remarks>
[Collection(EnvironmentSensitiveCollection.Name)]
public sealed class VramPolicyTests
{
    /// <summary>Runs <paramref name="body"/> with <c>HARTSY_LOWVRAM</c> pinned, then restores it — resolution falls
    /// back to the environment, so a stray value on the dev box would otherwise decide these assertions.</summary>
    private static void WithEnvironment(string? value, Action body)
    {
        string? previous = Environment.GetEnvironmentVariable(LowVramPolicy.EnvironmentVariable);
        try
        {
            Environment.SetEnvironmentVariable(LowVramPolicy.EnvironmentVariable, value);
            LowVramPolicy.ResetCacheForTests();
            body();
        }
        finally
        {
            Environment.SetEnvironmentVariable(LowVramPolicy.EnvironmentVariable, previous);
            LowVramPolicy.ResetCacheForTests();
        }
    }

    /// <summary>The Phase 0 regression gate: <see cref="VramTier.Auto"/> must leave every lever on
    /// <see cref="LeverState.Auto"/>, so each consumer keeps making exactly the measured decision it made before the
    /// policy object existed. A preset creeping into Auto is a silent behavior change across every model.</summary>
    [Fact]
    public void AutoTier_LeavesEveryLeverUndecided()
    {
        VramPolicy policy = VramPolicyResolver.Expand(VramTier.Auto);
        Assert.Equal(VramTier.Auto, policy.Tier);
        Assert.Equal(LeverState.Auto, policy.KeepResident);
        Assert.Equal(LeverState.Auto, policy.PhaseUnload);
        Assert.Equal(LeverState.Auto, policy.WeightStreaming);
        Assert.Equal(LeverState.Auto, policy.ActivationOffload);
        Assert.Equal(LeverState.Auto, policy.FreeAfterGeneration);
        Assert.Equal(LeverState.Auto, policy.QuantizedCompute);
        Assert.Equal(LeverState.Auto, policy.MultiGpuSpill);
        Assert.Equal(CachePrecision.Auto, policy.Caches);
        Assert.Equal(1.0f, policy.ChunkScale);
        Assert.Null(policy.PrefetchAhead);
        Assert.Null(policy.HeadroomBytes);
    }

    [Fact]
    public void PerformanceTier_NeverStreamsAndNeverEvicts()
    {
        VramPolicy policy = VramPolicyResolver.Expand(VramTier.Performance);
        Assert.Equal(LeverState.Off, policy.WeightStreaming);
        Assert.Equal(LeverState.Off, policy.PhaseUnload);
        Assert.Equal(LeverState.On, policy.KeepResident);
        Assert.Equal(CachePrecision.Full, policy.Caches);
        Assert.Equal(LeverState.Off, policy.MultiGpuSpill);
    }

    [Fact]
    public void MaximumTier_TurnsOnEveryLeverIncludingTheConstructionScopedOnes()
    {
        VramPolicy policy = VramPolicyResolver.Expand(VramTier.Maximum);
        Assert.Equal(LeverState.On, policy.WeightStreaming);
        Assert.Equal(LeverState.On, policy.PhaseUnload);
        Assert.Equal(LeverState.On, policy.ActivationOffload);
        Assert.Equal(LeverState.On, policy.FreeAfterGeneration);
        Assert.Equal(LeverState.On, policy.QuantizedCompute);
        Assert.Equal(LeverState.On, policy.MultiGpuSpill);
        Assert.Equal(LeverState.Off, policy.KeepResident);
        Assert.Equal(CachePrecision.Half, policy.Caches);
        Assert.True(policy.ChunkScale < 1.0f);
    }

    /// <summary>Aggressive streams unconditionally but must NOT quantize or spill — those are construction-scoped and
    /// change numerics or device placement, which a "stream harder" tier has no business doing implicitly.</summary>
    [Fact]
    public void AggressiveTier_StreamsButLeavesConstructionScopedLeversAlone()
    {
        VramPolicy policy = VramPolicyResolver.Expand(VramTier.Aggressive);
        Assert.Equal(LeverState.On, policy.WeightStreaming);
        Assert.Equal(LeverState.Auto, policy.QuantizedCompute);
        Assert.Equal(LeverState.Auto, policy.MultiGpuSpill);
    }

    [Theory]
    [InlineData(0L, VramTier.Balanced)]
    [InlineData(6L << 30, VramTier.Aggressive)]
    [InlineData(8L << 30, VramTier.Aggressive)]
    [InlineData(12L << 30, VramTier.Balanced)]
    [InlineData(16L << 30, VramTier.Balanced)]
    [InlineData(24L << 30, VramTier.Performance)]
    [InlineData(80L << 30, VramTier.Performance)]
    public void GpuClass_SeedsTheTierFromTotalVram(long totalBytes, VramTier expected)
        => Assert.Equal(expected, GpuVramClass.Seed(totalBytes));

    [Fact]
    public void Overrides_PinOneLeverWithoutDisturbingTheRest()
    {
        VramPolicy basePolicy = VramPolicyResolver.Expand(VramTier.Performance);
        VramPolicy merged = VramPolicyResolver.Apply(basePolicy, new VramOverrides { WeightStreaming = LeverState.On });
        Assert.Equal(LeverState.On, merged.WeightStreaming);
        Assert.Equal(LeverState.On, merged.KeepResident);
        Assert.Equal(CachePrecision.Full, merged.Caches);
        Assert.Equal(VramTier.Performance, merged.Tier);
    }

    /// <summary>An override naming a tier re-expands from it, and the remaining members refine THAT preset rather than
    /// the backend's — otherwise "Aggressive but keep caches exact" would silently inherit Performance's levers.</summary>
    [Fact]
    public void Overrides_TierReExpandsBeforeTheRemainingMembersApply()
    {
        VramPolicy basePolicy = VramPolicyResolver.Expand(VramTier.Performance);
        VramPolicy merged = VramPolicyResolver.Apply(basePolicy,
            new VramOverrides { Tier = VramTier.Aggressive, Caches = CachePrecision.Full });
        Assert.Equal(VramTier.Aggressive, merged.Tier);
        Assert.Equal(LeverState.On, merged.WeightStreaming);
        Assert.Equal(CachePrecision.Full, merged.Caches);
    }

    [Fact]
    public void Overrides_EmptyOrNullReturnsTheBaseUntouched()
    {
        VramPolicy basePolicy = VramPolicyResolver.Expand(VramTier.Balanced);
        Assert.Same(basePolicy, VramPolicyResolver.Apply(basePolicy, null));
        Assert.Same(basePolicy, VramPolicyResolver.Apply(basePolicy, new VramOverrides()));
        Assert.True(new VramOverrides().IsEmpty);
        Assert.False(new VramOverrides { ChunkScale = 0.5f }.IsEmpty);
    }

    [Theory]
    [InlineData(LowVramMode.Auto, LeverState.Auto)]
    [InlineData(LowVramMode.ForceOn, LeverState.On)]
    [InlineData(LowVramMode.ForceOff, LeverState.Off)]
    public void LegacyMode_MapsOntoTheStreamingLeverAndBack(LowVramMode mode, LeverState expected)
    {
        VramPolicy policy = VramPolicyResolver.FromLegacyMode(mode);
        Assert.Equal(expected, policy.WeightStreaming);
        Assert.Equal(mode, VramPolicyResolver.ToLegacyMode(policy));
    }

    /// <summary>The bridge that keeps every un-migrated call site working: a policy pinned through the new registry
    /// must be visible to the old <see cref="LowVramPolicy.Resolve(IBackend?)"/>, and vice versa.</summary>
    [Fact]
    public void Registry_AndLegacyOverride_ShareOneSourceOfTruth()
    {
        WithEnvironment(null, () =>
        {
            using RecordingStreamingBackend backend = new RecordingStreamingBackend(cache: null);
            Assert.False(VramPolicyRegistry.HasPolicy(backend));

            LowVramPolicy.SetOverride(backend, LowVramMode.ForceOn);
            Assert.True(VramPolicyRegistry.HasPolicy(backend));
            Assert.Equal(LeverState.On, VramPolicyRegistry.Resolve(backend).WeightStreaming);
            Assert.Equal(LowVramMode.ForceOn, LowVramPolicy.Resolve(backend));

            VramPolicyRegistry.Set(backend, VramPolicyResolver.Expand(VramTier.Performance));
            Assert.Equal(LowVramMode.ForceOff, LowVramPolicy.Resolve(backend));

            LowVramPolicy.ClearOverride(backend);
            Assert.False(VramPolicyRegistry.HasPolicy(backend));
            Assert.Equal(LowVramMode.Auto, LowVramPolicy.Resolve(backend));
        });
    }

    /// <summary>With no per-backend policy the environment still decides, so an existing deployment that sets only
    /// <c>HARTSY_LOWVRAM</c> keeps behaving exactly as it did.</summary>
    [Fact]
    public void Registry_FallsBackToTheConfiguredPostureWhenNoPolicyIsPinned()
    {
        try
        {
            KnobStore.Set(EngineKnobs.LowVram, "on");
            LowVramPolicy.ResetCacheForTests();
            Assert.Equal(LeverState.On, VramPolicyRegistry.Resolve(backend: null, overrides: null).WeightStreaming);
        }
        finally
        {
            KnobStore.Clear(EngineKnobs.LowVram);
            LowVramPolicy.ResetCacheForTests();
        }
    }

    /// <summary>A per-request override must beat the backend's pinned policy for the runtime levers.</summary>
    [Fact]
    public void Registry_RequestOverrideRefinesTheBackendPolicy()
    {
        WithEnvironment(null, () =>
        {
            using RecordingStreamingBackend backend = new RecordingStreamingBackend(cache: null);
            VramPolicyRegistry.Set(backend, VramPolicyResolver.Expand(VramTier.Performance));
            VramPolicy resolved = VramPolicyRegistry.Resolve(backend,
                new VramOverrides { WeightStreaming = LeverState.On, ChunkScale = 0.25f });
            Assert.Equal(LeverState.On, resolved.WeightStreaming);
            Assert.Equal(0.25f, resolved.ChunkScale);
            Assert.Equal(LeverState.On, resolved.KeepResident);
        });
    }
}
