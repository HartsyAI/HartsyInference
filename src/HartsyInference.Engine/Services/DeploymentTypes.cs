using HartsyInference.Engine.Dispatch;

namespace HartsyInference.Engine.Services;

/// <summary>Where a deployment is in its life. Only the transitions <see cref="DeploymentStateMachine"/> allows are reached.</summary>
public enum DeploymentState
{
    /// <summary>The model is being loaded onto its device.</summary>
    Loading,

    /// <summary>Loaded and taking requests.</summary>
    Ready,

    /// <summary>Loaded, but its serving loop has stopped; it takes no requests until it is reloaded.</summary>
    Degraded,

    /// <summary>Taking no new requests: its queue is cancelled and its active requests are finishing.</summary>
    Draining,

    /// <summary>Its load failed; it holds nothing.</summary>
    Failed,

    /// <summary>Its model has been freed.</summary>
    Unloaded,
}

/// <summary>The allowed deployment transitions. A transition outside this table is a bug in the caller, so the registry refuses it rather than recording it.</summary>
public static class DeploymentStateMachine
{
    /// <summary>Whether a deployment may move from <paramref name="from"/> to <paramref name="to"/>.</summary>
    public static bool CanTransition(DeploymentState from, DeploymentState to) => (from, to) switch
    {
        (DeploymentState.Loading, DeploymentState.Ready) => true,
        (DeploymentState.Loading, DeploymentState.Failed) => true,
        (DeploymentState.Loading, DeploymentState.Unloaded) => true,
        (DeploymentState.Ready, DeploymentState.Degraded) => true,
        (DeploymentState.Ready, DeploymentState.Draining) => true,
        (DeploymentState.Ready, DeploymentState.Unloaded) => true,
        (DeploymentState.Degraded, DeploymentState.Ready) => true,
        (DeploymentState.Degraded, DeploymentState.Draining) => true,
        (DeploymentState.Degraded, DeploymentState.Failed) => true,
        (DeploymentState.Draining, DeploymentState.Unloaded) => true,
        (DeploymentState.Draining, DeploymentState.Failed) => true,
        (DeploymentState.Failed, DeploymentState.Loading) => true,
        (DeploymentState.Failed, DeploymentState.Unloaded) => true,
        (DeploymentState.Unloaded, DeploymentState.Loading) => true,
        _ => false,
    };
}

/// <summary>A request to deploy a model: a named deployment on one device. One deployment holds a device at a time, so deploying onto a device a deployment
/// already holds replaces it.</summary>
public sealed record DeploymentRequest
{
    /// <summary>The caller's name for the deployment.</summary>
    public required string DeploymentId { get; init; }

    /// <summary>The model to load.</summary>
    public required ModelSpec Model { get; init; }

    /// <summary>The device to load onto (for example <c>cuda:0</c>); null uses the engine's primary device.</summary>
    public string? Device { get; init; }
}

/// <summary>A deployment as the registry reports it.</summary>
public sealed record DeploymentStatus(string DeploymentId, string Model, string? Device, DeploymentState State, string? Problem);

/// <summary>What a deployment's device is doing now. The KV figures are page counts, and they are null on a model with no KV pool (the V4.1 host keeps its own state).</summary>
public sealed record DeploymentCapacity(DeploymentState State, int Active, int Queued, int MaxConcurrent, int? KvPagesFree, int? KvPagesTotal);
