using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelAssets.MiniMaxH3;

/// <summary>F64-fitted affine map from a pruned H3 curve table to dense AdaLN time features.</summary>
public sealed class MiniMaxH3PddAffineBasis : IDisposable
{
    private int _disposed;

    /// <summary>Creates an owned basis after its residual has passed the requested gate.</summary>
    public MiniMaxH3PddAffineBasis(Tensor intercept, Tensor projection, double relativeResidual)
    {
        Intercept = intercept ?? throw new ArgumentNullException(nameof(intercept));
        Projection = projection ?? throw new ArgumentNullException(nameof(projection));
        RelativeResidual = relativeResidual;
    }

    /// <summary>Dense-feature DC term <c>c</c>, shape <c>[dense]</c>.</summary>
    public Tensor Intercept { get; }

    /// <summary>Curve-coordinate projection <c>V</c>, shape <c>[dense, curve]</c>.</summary>
    public Tensor Projection { get; }

    /// <summary>Relative Frobenius error of <c>c + table*V^T</c> against the dense time curve.</summary>
    public double RelativeResidual { get; }

    /// <summary>Releases the two fitted F32 tensors.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        Intercept.Dispose();
        Projection.Dispose();
    }
}
