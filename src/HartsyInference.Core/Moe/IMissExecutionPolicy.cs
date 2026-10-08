using HartsyInference.Core.Backends;

namespace HartsyInference.Core.Moe;

/// <summary>
/// Decides where a routed expert runs, given whether the cache holds it. The scheduler asks this once per routed expert; the
/// policy is the only place that encodes a placement choice, so cost-based policies can replace the default later.
/// </summary>
public interface IMissExecutionPolicy
{
    /// <summary>
    /// Placement for <paramref name="key"/>, which the cache reports as <paramref name="resident"/>, serving
    /// <paramref name="rows"/> pairs.
    /// </summary>
    ExpertPlacement Place(ExpertKey key, bool resident, int rows);
}
