namespace HartsyInference.Core.Backends.Attention;

/// <summary>One kernel a backend can run for attention: its declared capability, the runtime conditions that admit a call
/// (knobs, kernel presence, failure state, budget) and the launch.</summary>
public abstract class AttentionKernelRow
{
    /// <summary>What the row can serve.</summary>
    public abstract AttentionKernelCapability Capability { get; }

    /// <summary>Runtime admission; the planner asks only after <see cref="AttentionKernelCapability.Excludes"/> passed.</summary>
    public abstract bool Admits(in AttentionCall call);

    /// <summary>Runs the call. False is a recoverable decline (no state changed) and the next admitted row runs.</summary>
    public abstract bool Run(in AttentionCall call);

    /// <summary>The variant a run took, when one row covers several (a materialized row's dtype path); null otherwise.</summary>
    public virtual string? Variant(in AttentionCall call) => null;
}
