using HartsyInference.Core.MemoryManagement;
using HartsyInference.Core.Runtime;
using Xunit;

namespace HartsyInference.Core.Tests.MemoryManagement;

/// <summary>Pins the equivalence that makes absorbing <c>HARTSY_KEEP_MODELS</c> a no-op for existing deployments: an unpinned lever must resolve to exactly what the raw environment read produced.</summary>
public sealed class VramLeversTests
{
    private static void WithKeepModels(string? value, Action body)
    {
        string? previous = Environment.GetEnvironmentVariable(VramLevers.KeepModelsVariable);
        try
        {
            Environment.SetEnvironmentVariable(VramLevers.KeepModelsVariable, value);
            body();
        }
        finally
        {
            Environment.SetEnvironmentVariable(VramLevers.KeepModelsVariable, previous);
        }
    }

    /// <summary>The regression gate for Phase 1a. Every spelling the 15 pipelines used to read directly must survive
    /// the move to the policy unchanged — a drift here silently flips cross-generation residency fleet-wide.</summary>
    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("1")]
    [InlineData("true")]
    [InlineData("TRUE")]
    [InlineData("0")]
    [InlineData("false")]
    [InlineData("False")]
    [InlineData("nonsense")]
    public void AutoLever_MatchesTheRawEnvironmentReadItReplaces(string? value)
    {
        WithKeepModels(value, () =>
        {
            bool legacy = EnvSwitch.IsEnabled(VramLevers.KeepModelsVariable, defaultOn: true);
            bool viaPolicy = VramLevers.KeepResident(VramPolicyResolver.Expand(VramTier.Auto));
            Assert.Equal(legacy, viaPolicy);
        });
    }

    /// <summary>Unset must keep weights resident — the shipped default the 15 pipelines encoded as <c>defaultOn: true</c>.</summary>
    [Fact]
    public void AutoLever_DefaultsToKeepingWeightsResident()
        => WithKeepModels(null, () => Assert.True(VramLevers.KeepResident(VramPolicyResolver.Expand(VramTier.Auto))));

    /// <summary>A pinned lever must beat the environment in BOTH directions, or a tier could never override a machine's exported default.</summary>
    [Theory]
    [InlineData(LeverState.On, "0", true)]
    [InlineData(LeverState.On, null, true)]
    [InlineData(LeverState.Off, "1", false)]
    [InlineData(LeverState.Off, null, false)]
    public void PinnedLever_WinsOverTheEnvironment(LeverState state, string? env, bool expected)
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
