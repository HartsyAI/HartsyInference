using HartsyInference.Core.Backends;

namespace HartsyInference.Core.Moe;

/// <summary>
/// A policy that ignores residency and places each expert by a caller-supplied rule. Used to force splits in tests, for
/// example all-CPU, all-GPU, or a deterministic half by key, so heterogeneous execution can be compared against references.
/// </summary>
public sealed class ForcedPlacementPolicy : IMissExecutionPolicy
{
    private readonly Func<ExpertKey, ExpertPlacement> _rule;

    /// <summary>Creates a policy from a placement rule.</summary>
    public ForcedPlacementPolicy(Func<ExpertKey, ExpertPlacement> rule)
    {
        ArgumentNullException.ThrowIfNull(rule);
        _rule = rule;
    }

    /// <inheritdoc/>
    public ExpertPlacement Place(ExpertKey key, bool resident, int rows) => _rule(key);
}
