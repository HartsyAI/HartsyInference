using System.Collections.ObjectModel;

namespace HartsyInference.Core.Moe;

/// <summary>
/// One sparse (MoE) feed-forward layer: a router selecting among routed experts, an optional shared expert group that
/// always runs, and the expert program both groups execute.
/// </summary>
/// <param name="Router">Routing recipe; its <see cref="RouterDescriptor.NumExperts"/> must equal <see
/// cref="Routed"/>.<see cref="ExpertGroupDescriptor.Count"/>.</param>
/// <param name="Routed">Routed experts.</param>
/// <param name="Shared">Always-on shared experts, or null.</param>
/// <param name="SharedIsGated">Shared output is scaled by a learned sigmoid gate (Qwen2-MoE).</param>
/// <param name="Program">Computation each expert runs.</param>
public sealed record MoeLayerDescriptor(
    RouterDescriptor Router, ExpertGroupDescriptor Routed, ExpertGroupDescriptor? Shared, bool SharedIsGated, ExpertProgram Program)
{
    /// <summary>Validates the layer and returns it.</summary>
    /// <exception cref="ArgumentException">Router width and expert count disagree.</exception>
    public MoeLayerDescriptor Validated()
    {
        ArgumentNullException.ThrowIfNull(Router);
        ArgumentNullException.ThrowIfNull(Routed);
        ArgumentNullException.ThrowIfNull(Program);
        Router.Validated();
        Routed.Validated();
        Shared?.Validated();
        Program.Validated();
        if (SharedIsGated && Shared is null) throw new ArgumentException("A shared-output gate needs a shared expert group.", nameof(SharedIsGated));
        if (Router.NumExperts != Routed.Count)
            throw new ArgumentException($"Router selects among {Router.NumExperts} experts but the routed group holds {Routed.Count}.",
                    nameof(Router));
        return this;
    }

    /// <summary>Number of routed experts.</summary>
    public int ExpertCount => Routed.Count;

    /// <summary>Payload bytes of all routed and shared experts in this layer.</summary>
    public long PayloadBytes => Routed.PayloadBytes + (Shared?.PayloadBytes ?? 0);
}
