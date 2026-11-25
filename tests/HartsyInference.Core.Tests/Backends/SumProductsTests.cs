using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Backends.Layout;
using HartsyInference.Core.Backends.Reduction;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Core.Tests.Backends;

/// <summary><see cref="IBackend.SumProducts"/> against loops written the way its callers index, and its refusals.</summary>
public sealed class SumProductsTests
{
    private const ProductTerms All = ProductTerms.Dot | ProductTerms.SquaredA | ProductTerms.SquaredB;
    private static readonly IBackend Backend = new CpuBackend();

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
