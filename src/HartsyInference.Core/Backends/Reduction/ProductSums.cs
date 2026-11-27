using HartsyInference.Core.Backends.Layout;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends.Reduction;

/// <summary>Reads of <see cref="IBackend.SumProducts"/> that a host decision needs.</summary>
public static class ProductSums
{
    /// <summary><c>Σ a·b</c> and <c>Σ b·b</c> over <paramref name="view"/> of both operands, as one op and one read of its two results. That read is a
    /// device-to-host sync, so it belongs where the host needs the numbers (a CFG-zero projection once per step).</summary>
    public static (double Dot, double SquaredB) DotAndSquaredB(IBackend backend, Tensor a, Tensor b, StridedView view)
    {
        ArgumentNullException.ThrowIfNull(backend);
        using Tensor sums = new(new TensorShape(2), DType.F64);
        backend.SumProducts(sums, a, b, SumProductsSpec.Over(view, ProductTerms.Dot | ProductTerms.SquaredB));
        ReadOnlySpan<double> values = sums.AsReadOnlySpan<double>();
        return (values[0], values[1]);
    }
}
