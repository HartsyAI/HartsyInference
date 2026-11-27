using System.Numerics;
using HartsyInference.Core.Backends.Layout;

namespace HartsyInference.Core.Backends.Reduction;

/// <summary>One <see cref="IBackend.SumProducts"/>: a view of each operand with the same extents, how many leading axes are kept, and which products
/// are summed over the rest. A range is a <see cref="StridedView.Slice"/>; rows inside a longer dimension are a <see cref="StridedView.Split"/>.</summary>
public readonly struct SumProductsSpec
{
    /// <summary>Sums over every axis of the views past the first <paramref name="keptAxes"/>.</summary>
    public SumProductsSpec(StridedView a, StridedView b, int keptAxes, ProductTerms terms)
    {
        if (!a.HasSameDims(b))
            throw new ArgumentException($"SumProducts views {a} and {b} must have the same extents.", nameof(b));
        if (keptAxes < 0 || keptAxes >= a.Rank)
            throw new ArgumentOutOfRangeException(nameof(keptAxes), keptAxes, $"SumProducts keeps 0 to {a.Rank - 1} of {a.Rank} axes, so at least one is summed over.");
        if (terms == ProductTerms.None || (terms & ~(ProductTerms.Dot | ProductTerms.SquaredA | ProductTerms.SquaredB)) != 0)
            throw new ArgumentOutOfRangeException(nameof(terms), terms, "SumProducts needs at least one known term.");
        A = a;
        B = b;
        KeptAxes = keptAxes;
        Terms = terms;
    }

    /// <summary>The elements of the first operand.</summary>
    public StridedView A { get; }

    /// <summary>The elements of the second operand, with the extents of <see cref="A"/>.</summary>
    public StridedView B { get; }

    /// <summary>Leading axes that stay: each position on them is one group with its own sums.</summary>
    public int KeptAxes { get; }

    /// <summary>The sums each group gets.</summary>
    public ProductTerms Terms { get; }

    /// <summary>Sums per group.</summary>
    public int TermCount => BitOperations.PopCount((uint)Terms);

    /// <summary>One sum of each term over the whole view, the same for both operands.</summary>
    public static SumProductsSpec Over(StridedView view, ProductTerms terms) => new(view, view, 0, terms);

    /// <summary>One sum of each term per row of <paramref name="width"/> elements of a row-major operand of <paramref name="elements"/> elements,
    /// whatever its shape (a head of a token row).</summary>
    public static SumProductsSpec PerRow(long elements, long width, ProductTerms terms)
    {
        if (width <= 0 || elements % width != 0)
            throw new ArgumentException($"{elements} elements do not split into rows of {width}.", nameof(width));
        StridedView rows = StridedView.Of([elements / width, width]);
        return new SumProductsSpec(rows, rows, 1, terms);
    }

    public override string ToString() => $"a {A}, b {B}, keep {KeptAxes}, {Terms}";
}
