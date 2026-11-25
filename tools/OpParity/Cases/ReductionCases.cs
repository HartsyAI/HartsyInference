using HartsyInference.Core.Backends.Host;
using HartsyInference.Core.Backends.Layout;
using HartsyInference.Core.Backends.Reduction;
using HartsyInference.Core.Tensors;

namespace HartsyInference.OpParity.Cases;

/// <summary>Sums of products in the shapes their callers use: LongCat's CFG-zero projection over the guided frames, and one norm per attention head.
/// Judged against the host reference: there is no entry point they replaced.</summary>
public static class ReductionCases
{
    public const string Family = "reduction";

    private static readonly string[] All = ["cpu", "cuda", "vulkan"];

    /// <summary>The sums form in F64 in any order, then round once to F32.</summary>
    private static readonly Acceptance Sums = Acceptance.AgainstHost(1e-5, 1e-6, "the F64 sums may be added in another order and the result is rounded to F32");

    public static IReadOnlyList<OpCase> Cases()
    {
        TensorShape latent = new(4, 5, 24);
        SumProductsSpec guided = SumProductsSpec.Over(StridedView.Of(latent).Slice(1, 2, 3), ProductTerms.Dot | ProductTerms.SquaredB);
        TensorShape rows = new(6, 32);
        SumProductsSpec heads = SumProductsSpec.PerRow(6 * 32, 8, ProductTerms.SquaredA | ProductTerms.SquaredB);
        // A column reduction: the kept axis is the inner one in storage.
        SumProductsSpec columns = new(StridedView.Of(rows).Permute([1, 0]), StridedView.Of(rows).Permute([1, 0]), 1, ProductTerms.Dot);
        return
        [
            Case("sum_products_range_f32", latent, 2, guided),
            Case("sum_products_rows_f32", rows, 24 * 2, heads),
            Case("sum_products_columns_f32", rows, 32, columns),
            Case("sum_products_wide_f32", new TensorShape(3, 40000), 3, SumProductsSpec.PerRow(3 * 40000, 40000, ProductTerms.Dot)),
        ];
    }

    private static OpCase Case(string name, TensorShape shape, int sums, SumProductsSpec spec) => new(Family, name, seed =>
        {
            OpTensors t = new();
            t.Inputs["a"] = Seeded.F32(shape, seed, 1, 2f);
            t.Inputs["b"] = Seeded.F32(shape, seed, 2, 2f);
            t.Outputs["out"] = Seeded.Zeros(new TensorShape(sums));
            return t;
        },
        (be, t) => be.SumProducts(t.Outputs["out"], t.Inputs["a"], t.Inputs["b"], spec), _ => true, All, Sums,
        t => SumProductsReference.Apply(t.Outputs["out"], t.Inputs["a"], t.Inputs["b"], spec));
}
