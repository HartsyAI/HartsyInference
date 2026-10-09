using HartsyInference.Engine.Services;
using Xunit;

namespace HartsyInference.LLM.Tests;

/// <summary>The deployment state machine: the transitions a deployment may take and the ones it may not. The registry refuses the rest, so a bug shows as an error rather
/// than as a wrong state.</summary>
public sealed class DeploymentStateMachineTests
{
    [Theory]
    [InlineData(DeploymentState.Loading, DeploymentState.Ready)]
    [InlineData(DeploymentState.Loading, DeploymentState.Failed)]
    [InlineData(DeploymentState.Ready, DeploymentState.Degraded)]
    [InlineData(DeploymentState.Ready, DeploymentState.Draining)]
    [InlineData(DeploymentState.Ready, DeploymentState.Unloaded)]
    [InlineData(DeploymentState.Degraded, DeploymentState.Ready)]
    [InlineData(DeploymentState.Degraded, DeploymentState.Draining)]
    [InlineData(DeploymentState.Draining, DeploymentState.Unloaded)]
    [InlineData(DeploymentState.Failed, DeploymentState.Loading)]
    [InlineData(DeploymentState.Unloaded, DeploymentState.Loading)]
    public void An_Allowed_Transition_Is_Accepted(DeploymentState from, DeploymentState to)
    {
        Assert.True(DeploymentStateMachine.CanTransition(from, to));
    }

    [Theory]
    [InlineData(DeploymentState.Ready, DeploymentState.Loading)]
    [InlineData(DeploymentState.Unloaded, DeploymentState.Ready)]
    [InlineData(DeploymentState.Draining, DeploymentState.Ready)]
    [InlineData(DeploymentState.Failed, DeploymentState.Ready)]
    [InlineData(DeploymentState.Unloaded, DeploymentState.Draining)]
    public void A_Forbidden_Transition_Is_Refused(DeploymentState from, DeploymentState to)
    {
        Assert.False(DeploymentStateMachine.CanTransition(from, to));
    }
}
