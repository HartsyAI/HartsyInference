using HartsyInference.Core.Backends;

namespace HartsyInference.Core.Moe;

/// <summary>Runs the CPU share of a planned layer: one expert over its token rows, on the host.</summary>
public interface IExpertHostRunner
{
    /// <summary>Runs <paramref name="key"/> with <paramref name="program"/> over <paramref name="rows"/> token rows;
    /// <paramref name="y"/> is overwritten and complete when this returns.</summary>
    /// <param name="program">The activation and clamp bounds the layer runs with, the same ones the device side gets.</param>
    /// <param name="key">The expert, held by the host.</param>
    /// <param name="x"><c>rows × H</c> inputs, row-major.</param>
    /// <param name="rows">Token rows.</param>
    /// <param name="y"><c>rows × H</c> outputs.</param>
    void Run(ExpertProgram program, ExpertKey key, ReadOnlySpan<float> x, int rows, Span<float> y);
}
