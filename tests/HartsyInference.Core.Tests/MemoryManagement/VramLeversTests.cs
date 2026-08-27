using HartsyInference.Core.MemoryManagement;
using HartsyInference.Core.Configuration;
using Xunit;

namespace HartsyInference.Core.Tests.MemoryManagement;

/// <summary>Pins the equivalence that makes absorbing <c>HARTSY_KEEP_MODELS</c> a no-op for existing deployments: an unpinned lever must resolve to exactly what the raw environment read produced.</summary>
[Collection(EnvironmentSensitiveCollection.Name)]
public sealed class VramLeversTests
{
    /// <summary>Runs <paramref name="body"/> with the residency fallback pinned, then restores it.</summary>
    private static void WithKeepModels(bool? value, Action body)
    {
        try
        {
            if (value is bool v)
            {
                KnobStore.Set(EngineKnobs.KeepModels, v);
            }
            body();
        }
        finally
        {
            KnobStore.Clear(EngineKnobs.KeepModels);
        }
    }

    /// <summary>An unpinned lever takes the fallback knob's value, which is what makes a tier able to state a default at all.</summary>
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void AutoLever_TakesTheFallbackKnob(bool configured)
    {
        WithKeepModels(configured, () =>
            Assert.Equal(configured, VramLevers.KeepResident(VramPolicyResolver.Expand(VramTier.Auto))));
    }

    /// <summary>Unset must keep weights resident — the shipped default the 15 pipelines encoded as <c>defaultOn: true</c>.</summary>
    [Fact]
    public void AutoLever_DefaultsToKeepingWeightsResident()
        => WithKeepModels(null, () => Assert.True(VramLevers.KeepResident(VramPolicyResolver.Expand(VramTier.Auto))));

    /// <summary>A pinned lever must beat the fallback in BOTH directions, or a tier could never override the machine's configuration.</summary>
    [Theory]
    [InlineData(LeverState.On, false, true)]
    [InlineData(LeverState.On, null, true)]
    [InlineData(LeverState.Off, true, false)]
    [InlineData(LeverState.Off, null, false)]
    public void PinnedLever_WinsOverTheFallback(LeverState state, bool? env, bool expected)
    {
        WithKeepModels(env, () =>
        {
            VramPolicy policy = VramPolicyResolver.Expand(VramTier.Auto) with { KeepResident = state };
            Assert.Equal(expected, VramLevers.KeepResident(policy));
        });
    }

    /// <summary>The tiers that take an explicit position on residency must actually carry it through the resolver.</summary>
    [Fact]
    public void Tiers_CarryTheirResidencyStanceThroughTheResolver()
    {
        WithKeepModels(null, () =>
        {
            Assert.True(VramLevers.KeepResident(VramPolicyResolver.Expand(VramTier.Performance)));
            Assert.False(VramLevers.KeepResident(VramPolicyResolver.Expand(VramTier.Aggressive)));
            Assert.False(VramLevers.KeepResident(VramPolicyResolver.Expand(VramTier.Maximum)));
        });
    }
}
