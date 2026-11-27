namespace HartsyInference.Core.Backends.Reduction;

/// <summary>Which products a <see cref="SumProductsSpec"/> sums. The sums of one group land in adjacent output elements in the order of these flags.</summary>
[Flags]
public enum ProductTerms
{
    /// <summary>No sum; refused.</summary>
    None = 0,

    /// <summary><c>Σ a·b</c>, the dot product.</summary>
    Dot = 1,

    /// <summary><c>Σ a·a</c>, the squared norm of the first operand.</summary>
    SquaredA = 2,

    /// <summary><c>Σ b·b</c>, the squared norm of the second operand.</summary>
    SquaredB = 4,
}
