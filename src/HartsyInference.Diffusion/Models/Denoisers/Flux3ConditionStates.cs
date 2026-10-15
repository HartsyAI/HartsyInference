using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>The text and reference streams after their per-mode blocks. They depend on the conditioning alone (and,
/// for a flow-map build, on the step's target time), so a guided build computes them once per generation.</summary>
public sealed class Flux3ConditionStates(Tensor text, Tensor? references) : IDisposable
{
    private int _disposed;

    /// <summary>Text stream <c>[1, T, D]</c>.</summary>
    public Tensor Text { get; } = text;

    /// <summary>Reference stream <c>[1, R, D]</c>, or null without references.</summary>
    public Tensor? References { get; } = references;

    /// <inheritdoc/>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        Text.Dispose();
        References?.Dispose();
    }
}
