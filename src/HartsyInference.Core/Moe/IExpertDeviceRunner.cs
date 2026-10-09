using HartsyInference.Core.Backends;

namespace HartsyInference.Core.Moe;

/// <summary>
/// Runs one expert on the device from its resident copy. The heterogeneous executor calls it for every assignment placed on
/// the GPU, and only for experts the planner found resident.
/// </summary>
public interface IExpertDeviceRunner
{
    /// <summary>Runs <paramref name="key"/> over <paramref name="rows"/> token rows; <paramref name="y"/> is overwritten and
    /// complete when this returns.</summary>
    /// <param name="key">The expert, resident on the device.</param>
    /// <param name="x"><c>rows × H</c> inputs, row-major.</param>
    /// <param name="rows">Token rows.</param>
    /// <param name="y"><c>rows × H</c> outputs.</param>
    void Run(ExpertKey key, ReadOnlySpan<float> x, int rows, Span<float> y);
}
