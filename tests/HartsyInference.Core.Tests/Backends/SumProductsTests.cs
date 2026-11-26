using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Contracts;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Backends.Layout;
using HartsyInference.Core.Backends.Reduction;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using Xunit;

namespace HartsyInference.Core.Tests.Backends;

/// <summary><see cref="IBackend.SumProducts"/> against loops written the way its callers index, and its refusals.</summary>
public sealed class SumProductsTests
{
    private const ProductTerms All = ProductTerms.Dot | ProductTerms.SquaredA | ProductTerms.SquaredB;
    private static readonly IBackend Backend = new CpuBackend();

    [Theory]
    [InlineData("sum_products.ptx")]
    [InlineData("sum_products.sm75.ptx")]
    public void TheShippedKernels_TakeTheParameterTheLauncherPasses(string ptx)
    {
        string text = File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "Ptx", ptx));

        int[] sizes = [.. System.Text.RegularExpressions.Regex.Matches(text, @"\.param \.align 8 \.b8 sum_products_\w+_param_\d+\[(\d+)\]")
            .Select(static m => int.Parse(m.Groups[1].Value, System.Globalization.CultureInfo.InvariantCulture))];

        Assert.Equal([CudaKernels.SumProductsArgsBytes, CudaKernels.SumProductsArgsBytes, CudaKernels.SumProductsArgsBytes], sizes);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(2)]
    [InlineData(4)]
    public void AFrameRangeOfALatent_SumsOnlyThatRange(int fromFrame)
    {
        const int channels = 3, frames = 5, hw = 7;
        Random rng = new(1);
        using Tensor pos = Filled(rng, channels, frames, hw), neg = Filled(rng, channels, frames, hw);
        ReadOnlySpan<float> p = pos.AsReadOnlySpan<float>(), n = neg.AsReadOnlySpan<float>();
        double dot = 0, squared = 0;
        for (int c = 0; c < channels; c++)
            for (int f = fromFrame; f < frames; f++)
                for (int e = 0; e < hw; e++)
                {
                    int index = (c * frames + f) * hw + e;
                    dot += (double)p[index] * n[index];
                    squared += (double)n[index] * n[index];
                }

        (double actualDot, double actualSquared) =
            ProductSums.DotAndSquaredB(Backend, pos, neg, StridedView.Of(pos.Shape).Slice(1, fromFrame, frames - fromFrame));

        Assert.Equal(dot, actualDot);
        Assert.Equal(squared, actualSquared);
    }

    [Fact]
    public void RowsInsideALongerDimension_GetTheirOwnSums_InTermOrder()
    {
        const int tokens = 4, heads = 3, headDim = 5;
        Random rng = new(2);
        using Tensor x = Filled(rng, tokens, heads * headDim), reference = Filled(rng, tokens, heads * headDim);
        using Tensor sums = new(new TensorShape(tokens * heads, 3), DType.F64);

        Backend.SumProducts(sums, x, reference, SumProductsSpec.PerRow(x.ElementCount, headDim, All));

        ReadOnlySpan<float> xs = x.AsReadOnlySpan<float>(), rs = reference.AsReadOnlySpan<float>();
        ReadOnlySpan<double> actual = sums.AsReadOnlySpan<double>();
        for (int row = 0; row < tokens * heads; row++)
        {
            double dot = 0, squaredX = 0, squaredR = 0;
            for (int d = row * headDim; d < (row + 1) * headDim; d++)
            {
                dot += (double)xs[d] * rs[d];
                squaredX += (double)xs[d] * xs[d];
                squaredR += (double)rs[d] * rs[d];
            }
            Assert.Equal(dot, actual[row * 3]);
            Assert.Equal(squaredX, actual[row * 3 + 1]);
            Assert.Equal(squaredR, actual[row * 3 + 2]);
        }
    }

    [Theory]
    [InlineData(ProductTerms.Dot)]
    [InlineData(ProductTerms.SquaredA)]
    [InlineData(ProductTerms.SquaredB)]
    [InlineData(ProductTerms.Dot | ProductTerms.SquaredB)]
    [InlineData(ProductTerms.SquaredA | ProductTerms.SquaredB)]
    public void AnyTerms_ArePackedInFlagOrder(ProductTerms terms)
    {
        Random rng = new(3);
        using Tensor a = Filled(rng, 6, 4), b = Filled(rng, 6, 4);
        using Tensor every = new(new TensorShape(6, 3), DType.F64), some = new(new TensorShape(6, SumProductsSpec.PerRow(24, 4, terms).TermCount), DType.F64);

        Backend.SumProducts(every, a, b, SumProductsSpec.PerRow(24, 4, All));
        Backend.SumProducts(some, a, b, SumProductsSpec.PerRow(24, 4, terms));

        List<double> expected = [];
        for (int row = 0; row < 6; row++)
            for (int bit = 0; bit < 3; bit++)
                if (((int)terms & (1 << bit)) != 0)
                    expected.Add(every.AsReadOnlySpan<double>()[row * 3 + bit]);
        Assert.Equal(expected, some.AsReadOnlySpan<double>().ToArray());
    }

    [Fact]
    public void ViewsMayDiffer_AndTheKeptAxisNeedNotBeTheOuterOneInStorage()
    {
        // Column sums of a [3, 4] operand against a [4, 3] one that holds its transpose's layout, two kept axes and a broadcast operand.
        Random rng = new(4);
        using Tensor a = Filled(rng, 3, 4), b = Filled(rng, 4, 3), vector = Filled(rng, 4);
        using Tensor columns = new(new TensorShape(4), DType.F64), weighted = new(new TensorShape(3), DType.F64), each = new(new TensorShape(3, 2), DType.F64);

        Backend.SumProducts(columns, a, b, new SumProductsSpec(StridedView.Of(a.Shape).Permute([1, 0]), StridedView.Of(b.Shape), 1, ProductTerms.Dot));
        Backend.SumProducts(weighted, a, vector, new SumProductsSpec(StridedView.Of(a.Shape), StridedView.VectorAlong([3, 4], 1), 1, ProductTerms.Dot));
        StridedView pairs = StridedView.Of(a.Shape).Split(1, 2, 2);
        Backend.SumProducts(each, a, a, new SumProductsSpec(pairs, pairs, 2, ProductTerms.Dot));

        ReadOnlySpan<float> x = a.AsReadOnlySpan<float>(), y = b.AsReadOnlySpan<float>(), v = vector.AsReadOnlySpan<float>();
        for (int column = 0; column < 4; column++)
        {
            double sum = 0;
            for (int row = 0; row < 3; row++) sum += (double)x[row * 4 + column] * y[column * 3 + row];
            Assert.Equal(sum, columns.AsReadOnlySpan<double>()[column]);
        }
        for (int row = 0; row < 3; row++)
        {
            double sum = 0;
            for (int column = 0; column < 4; column++) sum += (double)x[row * 4 + column] * v[column];
            Assert.Equal(sum, weighted.AsReadOnlySpan<double>()[row]);
            for (int pair = 0; pair < 2; pair++)
            {
                float first = x[row * 4 + pair * 2], second = x[row * 4 + pair * 2 + 1];
                Assert.Equal((double)first * first + (double)second * second, each.AsReadOnlySpan<double>()[row * 2 + pair]);
            }
        }
    }

    [Fact]
    public void TheSumsFormInF64_AndAnF32OutputIsRoundedOnce()
    {
        // One large product and a million small ones: an F32 accumulator drops every small one.
        const int count = 1 << 20;
        using Tensor a = new(new TensorShape(count), DType.F32);
        a.AsSpan<float>().Fill(0.0625f);
        a.AsSpan<float>()[0] = 4096f;
        double exact = 4096.0 * 4096.0 + (count - 1) * 0.0625 * 0.0625;
        using Tensor wide = new(new TensorShape(1), DType.F64), narrow = new(new TensorShape(1), DType.F32);
        SumProductsSpec spec = SumProductsSpec.Over(StridedView.Of(a.Shape), ProductTerms.Dot);

        Backend.SumProducts(wide, a, a, spec);
        Backend.SumProducts(narrow, a, a, spec);

        Assert.Equal(exact, wide.AsReadOnlySpan<double>()[0]);
        Assert.Equal((float)exact, narrow.AsReadOnlySpan<float>()[0]);
        Assert.NotEqual(4096f * 4096f, narrow.AsReadOnlySpan<float>()[0]);
    }

    // The index arithmetic the CUDA kernel runs, on the CPU: row-major element i of the collapsed axes, in group i / ElementsPerGroup.
    [Theory]
    [InlineData("range")]
    [InlineData("rows")]
    [InlineData("columns")]
    [InlineData("kept pair")]
    [InlineData("broadcast")]
    [InlineData("stepped")]
    public void TheCollapsedAxes_AddressTheSameElementsInTheSameGroups(string name)
    {
        StridedView whole = StridedView.Of([4, 6, 10]);
        StridedView columns = whole.Permute([2, 0, 1]);
        SumProductsSpec spec = name switch
        {
            "range" => SumProductsSpec.Over(whole.Slice(1, 2, 3), All),
            "rows" => SumProductsSpec.PerRow(240, 5, All),
            "columns" => new SumProductsSpec(columns, columns, 1, All),
            "kept pair" => new SumProductsSpec(whole.Split(2, 2, 5), whole.Split(2, 2, 5), 3, All),
            "broadcast" => new SumProductsSpec(whole, StridedView.VectorAlong([4, 6, 10], 1), 2, All),
            _ => new SumProductsSpec(whole.Slice(2, 1, 4, step: 2), whole.Slice(2, 0, 4), 1, All),
        };
        Random rng = new(6);
        using Tensor a = Filled(rng, 240), b = Filled(rng, 240);
        long groups = 1;
        for (int axis = 0; axis < spec.KeptAxes; axis++) groups *= spec.A[axis];
        using Tensor expected = new(new TensorShape(groups, 3), DType.F64);
        Backend.SumProducts(expected, a, b, spec);

        SumProductsGeometry g = SumProductsContract.Validate(expected, a, b, spec);
        double[] actual = new double[groups * 3];
        ReadOnlySpan<float> x = a.AsReadOnlySpan<float>(), y = b.AsReadOnlySpan<float>();
        Assert.Equal(g.Groups * g.ElementsPerGroup, g.Axes.ElementCount);
        for (long i = 0; i < g.Groups * g.ElementsPerGroup; i++)
        {
            long ia = g.Axes.SourceOffset, ib = g.Axes.DestinationOffset, rest = i;
            for (int axis = g.Axes.Rank - 1; axis >= 0; axis--)
            {
                long at = axis == 0 ? rest : rest % g.Axes.Dim(axis);
                rest /= g.Axes.Dim(axis);
                ia += at * g.Axes.SourceStride(axis);
                ib += at * g.Axes.DestinationStride(axis);
            }
            long group = i / g.ElementsPerGroup;
            actual[group * 3] += (double)x[(int)ia] * y[(int)ib];
            actual[group * 3 + 1] += (double)x[(int)ia] * x[(int)ia];
            actual[group * 3 + 2] += (double)y[(int)ib] * y[(int)ib];
        }

        Assert.Equal(expected.AsReadOnlySpan<double>().ToArray(), actual);
    }

    [Fact]
    public void ABadSpecIsRefused()
    {
        StridedView whole = StridedView.Of([3, 4]);
        Assert.Throws<ArgumentException>(() => new SumProductsSpec(whole, StridedView.Of([4, 3]), 1, ProductTerms.Dot));
        Assert.Throws<ArgumentOutOfRangeException>(() => new SumProductsSpec(whole, whole, 2, ProductTerms.Dot));
        Assert.Throws<ArgumentOutOfRangeException>(() => new SumProductsSpec(whole, whole, -1, ProductTerms.Dot));
        Assert.Throws<ArgumentOutOfRangeException>(() => new SumProductsSpec(whole, whole, 1, ProductTerms.None));
        Assert.Throws<ArgumentOutOfRangeException>(() => new SumProductsSpec(whole, whole, 1, (ProductTerms)8));
        Assert.Throws<ArgumentException>(() => SumProductsSpec.PerRow(12, 5, ProductTerms.Dot));
        Assert.Throws<ArgumentException>(() => SumProductsSpec.PerRow(12, 0, ProductTerms.Dot));
    }

    [Fact]
    public void BadOperandsAreRefused_AndTheInputsStayUntouched()
    {
        Random rng = new(5);
        using Tensor a = Filled(rng, 3, 4), b = Filled(rng, 3, 4), half = a.CastTo(DType.F16), small = Filled(rng, 2, 4);
        using Tensor sums = new(new TensorShape(3), DType.F64), wrongCount = new(new TensorShape(2), DType.F64), wrongType = new(new TensorShape(3), DType.I32);
        using Tensor inside = a.Reshape(new TensorShape(12));
        using Tensor overlapping = inside.SliceRows(0, 3);
        float[] before = a.AsReadOnlySpan<float>().ToArray();
        SumProductsSpec spec = SumProductsSpec.PerRow(12, 4, ProductTerms.Dot);

        Assert.Throws<NotSupportedException>(() => Backend.SumProducts(sums, half, b, spec));
        Assert.Throws<NotSupportedException>(() => Backend.SumProducts(sums, a, half, spec));
        Assert.Throws<NotSupportedException>(() => Backend.SumProducts(wrongType, a, b, spec));
        Assert.Throws<ArgumentException>(() => Backend.SumProducts(wrongCount, a, b, spec));
        Assert.Throws<ArgumentException>(() => Backend.SumProducts(sums, small, b, spec));
        Assert.Throws<ArgumentException>(() => Backend.SumProducts(sums, a, small, spec));
        Assert.Throws<ArgumentException>(() => Backend.SumProducts(overlapping, a, b, spec));
        Assert.Throws<ArgumentException>(() => Backend.SumProducts(sums, a, b, default));
        Assert.Throws<ArgumentNullException>(() => Backend.SumProducts(sums, null!, b, spec));
        Assert.Equal(before, a.AsReadOnlySpan<float>().ToArray());
    }

    private static Tensor Filled(Random rng, params long[] shape)
    {
        Tensor t = new(new TensorShape(shape), DType.F32);
        foreach (ref float v in t.AsSpan<float>()) v = (float)(rng.NextDouble() * 4 - 2);
        return t;
    }
}
