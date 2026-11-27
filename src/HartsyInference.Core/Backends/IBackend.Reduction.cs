using HartsyInference.Core.Backends.Host;
using HartsyInference.Core.Backends.Reduction;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends;

// Reductions over strided views.
public partial interface IBackend
{
    /// <summary>Dot products and squared norms over strided views: per position on the kept axes, the sums of <paramref name="spec"/>'s terms over the other
    /// axes, accumulated in F64 and written to <paramref name="output"/> as F64 or rounded once to F32. The result is a tensor; reading it is the
    /// caller's sync (<see cref="ProductSums"/>). F32 operands, which may be one tensor. Contract: <see cref="Contracts.SumProductsContract"/>;
    /// reference: <see cref="SumProductsReference"/>.</summary>
    void SumProducts(Tensor output, Tensor a, Tensor b, in SumProductsSpec spec) => SumProductsReference.Apply(output, a, b, spec);
}
