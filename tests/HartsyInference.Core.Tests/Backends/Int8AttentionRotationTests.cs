using System.Numerics;
using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Core.Tests.Backends;

/// <summary>The rotation and the representative key of INT8 attention: each transform is the Hadamard its name says,
/// in the pinned selection, the representative key is found by the pinned rule, and both do what they are for. An
/// outlier channel or a large component common to every key no longer costs the INT8 result its accuracy against the
/// F32 host reference.</summary>
public sealed class Int8AttentionRotationTests
{
    private static float[] Random(Random random, int count, float magnitude = 1f)
    {
        float[] values = new float[count];
        for (int i = 0; i < count; i++) values[i] = (float)(random.NextDouble() * 2 - 1) * magnitude;
        return values;
    }

    /// <summary>The natural-order Walsh-Hadamard of <paramref name="x"/> in double precision.</summary>
    private static double[] Sylvester(ReadOnlySpan<float> x)
    {
        int n = x.Length;
        double[] y = new double[n];
        for (int i = 0; i < n; i++)
        {
            double sum = 0;
            for (int j = 0; j < n; j++) sum += (BitOperations.PopCount((uint)(i & j)) & 1) == 0 ? x[j] : -x[j];
            y[i] = sum / Math.Sqrt(n);
        }
        return y;
    }

    private static void AssertClose(ReadOnlySpan<double> expected, ReadOnlySpan<float> actual, string what)
    {
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 2e-6 * (1 + Math.Abs(expected[i])), $"{what}: channel {i} is {actual[i]}, expected {expected[i]}.");
    }

    [Theory]
    [InlineData(64, 1, Int8Rotation.H4)]
    [InlineData(64, 256, Int8Rotation.H4)]
    [InlineData(64, 257, Int8Rotation.H64)]
    [InlineData(128, 256, Int8Rotation.H4)]
    [InlineData(128, 257, Int8Rotation.SignedH128)]
    [InlineData(256, 256, Int8Rotation.H4)]
    [InlineData(256, 8193, Int8Rotation.PaddedH128)]
    public void ThePinnedRulePicksTheTransformFromThePaddedHeadDimAndTheKeyCount(int paddedHeadDim, int keys, Int8Rotation expected)
    {
        Assert.Equal(expected, Int8AttentionRotation.Select(paddedHeadDim, keys));
        Assert.Equal(expected, Int8AttentionRotation.Resolve(Int8Rotation.Pinned, paddedHeadDim, keys));
    }

    [Fact]
    public void ATransformThatDoesNotFitThePaddedHeadDimIsRefused()
    {
        Assert.Throws<ArgumentException>(() => Int8AttentionRotation.Resolve(Int8Rotation.H64, 128, 300));
        Assert.Throws<ArgumentException>(() => Int8AttentionRotation.Resolve(Int8Rotation.SignedH128, 64, 300));
        Assert.Throws<ArgumentException>(() => Int8AttentionRotation.Resolve(Int8Rotation.PaddedH128, 128, 300));
        Assert.Equal(Int8Rotation.H4, Int8AttentionRotation.Resolve(Int8Rotation.H4, 256, 9000));
        Assert.Throws<ArgumentException>(() => Int8AttentionRotation.Apply(new float[64], Int8Rotation.Pinned));
    }

    [Fact]
    public void TheFourPointTransformIsAHadamardOnEveryFourChannels()
    {
        float[] row = Random(new Random(3), 128);
        float[] rotated = [.. row];
        Int8AttentionRotation.Apply(rotated, Int8Rotation.H4);
        for (int c = 0; c < row.Length; c += 4)
            AssertClose(Sylvester(row.AsSpan(c, 4)), rotated.AsSpan(c, 4), $"channels {c} to {c + 3}");
    }

    [Fact]
    public void TheSixtyFourPointTransformIsOneHadamardOverTheRow()
    {
        float[] row = Random(new Random(4), 64);
        float[] rotated = [.. row];
        Int8AttentionRotation.Apply(rotated, Int8Rotation.H64);
        AssertClose(Sylvester(row), rotated, "H64");
    }

    [Fact]
    public void TheSignedTransformFlipsItsFixedChannelsThenRunsOneHadamard()
    {
        float[] row = Random(new Random(5), 128);
        float[] signed = [.. row];
        for (int c = 0; c < 128; c++)
            if (!Int8AttentionRotation.KeepsSign(c)) signed[c] = -signed[c];
        float[] rotated = [.. row];
        Int8AttentionRotation.Apply(rotated, Int8Rotation.SignedH128);
        AssertClose(Sylvester(signed), rotated, "signed H128");
        // The sign pattern is a constant of the pinned kernel: 0x1035997b for channels 0 to 31, and so on.
        Assert.True(Int8AttentionRotation.KeepsSign(0));
        Assert.True(Int8AttentionRotation.KeepsSign(1));
        Assert.False(Int8AttentionRotation.KeepsSign(2));
        Assert.False(Int8AttentionRotation.KeepsSign(31));
        Assert.False(Int8AttentionRotation.KeepsSign(32));
        Assert.Equal(60, Enumerable.Range(0, 128).Count(Int8AttentionRotation.KeepsSign));
    }

    [Fact]
    public void ThePaddedTransformRunsOneHadamardOnEachHalf()
    {
        float[] row = Random(new Random(6), 256);
        float[] rotated = [.. row];
        Int8AttentionRotation.Apply(rotated, Int8Rotation.PaddedH128);
        AssertClose(Sylvester(row.AsSpan(0, 128)), rotated.AsSpan(0, 128), "low half");
        AssertClose(Sylvester(row.AsSpan(128, 128)), rotated.AsSpan(128, 128), "high half");
    }

    [Theory]
    [InlineData(Int8Rotation.H4, 64)]
    [InlineData(Int8Rotation.H64, 64)]
    [InlineData(Int8Rotation.SignedH128, 128)]
    [InlineData(Int8Rotation.PaddedH128, 256)]
    public void RotatingBothSidesLeavesTheirDotProductUnchanged(Int8Rotation rotation, int dim)
    {
        Random random = new(7);
        float[] q = Random(random, dim, 3f), k = Random(random, dim, 3f);
        double before = 0;
        for (int c = 0; c < dim; c++) before += (double)q[c] * k[c];
        Int8AttentionRotation.Apply(q, rotation);
        Int8AttentionRotation.Apply(k, rotation);
        double after = 0;
        for (int c = 0; c < dim; c++) after += (double)q[c] * k[c];
        Assert.Equal(before, after, 4);
    }

    [Fact]
    public void ARowOfOneLargeChannelIsSpreadOverEveryChannel()
    {
        float[] row = new float[128];
        row[5] = 100f;
        Int8AttentionRotation.Apply(row, Int8Rotation.SignedH128);
        // One 128-point Hadamard turns a spike of 100 into 128 values of 100 / sqrt(128).
        Assert.All(row, value => Assert.Equal(100f / MathF.Sqrt(128f), MathF.Abs(value), 4));
    }

    [Fact]
    public void KeysSharingALargeComponentHaveARepresentativeKeyAmongTheSamples()
    {
        const int keys = 300, dim = 64;
        Random random = new(8);
        float[] common = Random(random, dim, 40f);
        float[] k = Random(random, keys * dim);
        for (int j = 0; j < keys; j++)
            for (int c = 0; c < dim; c++) k[j * dim + c] += common[c];
        int anchor = Int8RepresentativeKey.Detect(k, keys, dim);
        Assert.Contains(anchor, Enumerable.Range(0, Int8RepresentativeKey.Samples).Select(s => Int8RepresentativeKey.SampleRow(s, keys)));
    }

    [Fact]
    public void KeysWithNothingInCommonHaveNoRepresentativeKey()
    {
        const int keys = 300, dim = 64;
        float[] k = Random(new Random(9), keys * dim);
        // Subtracting any one of nine unrelated rows doubles the energy of the rest.
        Assert.Equal(-1, Int8RepresentativeKey.Detect(k, keys, dim));
    }

    [Fact]
    public void TheRepresentativeKeyIsTheSampleNearestTheSamplesMeanAndTheLowestOnATie()
    {
        const int keys = 9, dim = 64;
        float[] k = new float[keys * dim];
        // Nine sampled rows along channel 0: 10, 11, 12, 13, 14, 15, 16, 17, 18. The mean is 14, row 4.
        for (int j = 0; j < keys; j++) k[j * dim] = 10 + j;
        Assert.Equal(4, Int8RepresentativeKey.Detect(k, keys, dim));
        // Every row equal: all candidates tie at zero distance and the first is taken; subtracting it leaves no energy.
        for (int j = 0; j < keys; j++) k[j * dim] = 5f;
        Assert.Equal(0, Int8RepresentativeKey.Detect(k, keys, dim));
        // All zero: nothing to shrink, so no key is subtracted.
        Array.Clear(k);
        Assert.Equal(-1, Int8RepresentativeKey.Detect(k, keys, dim));
    }

    /// <summary>A grouped causal call whose rows are plain noise, with what each case needs added on top.</summary>
    private sealed class Case : IDisposable
    {
        public readonly Tensor Query, Key, Value, Output, Expected;
        public readonly AttentionCall Call, Reference;

        public Case(int heads, int kvHeads, int queryRows, int keys, int dim, Action<float[], float[]> shape, int seed = 11)
        {
            Random random = new(seed);
            Query = new Tensor(new TensorShape(1, heads, queryRows, dim), DType.F32);
            Key = new Tensor(new TensorShape(1, kvHeads, keys, dim), DType.F32);
            Value = new Tensor(new TensorShape(1, kvHeads, keys, dim), DType.F32);
            Output = new Tensor(Query.Shape, DType.F32);
            Expected = new Tensor(Query.Shape, DType.F32);
            float[] q = Random(random, (int)Query.ElementCount), k = Random(random, (int)Key.ElementCount);
            shape(q, k);
            q.CopyTo(Query.AsSpan<float>());
            k.CopyTo(Key.AsSpan<float>());
            Random(random, (int)Value.ElementCount).CopyTo(Value.AsSpan<float>());
            float scale = 1f / MathF.Sqrt(dim);
            Call = AttentionCall.FromFlashAttention(Output, Query, Key, Value, keys, causal: false, qOffset: 0, scale, softcap: 0f,
                sink: null, slidingWindow: 0, alibiSlopes: null);
            Reference = AttentionCall.FromFlashAttention(Expected, Query, Key, Value, keys, causal: false, qOffset: 0, scale, softcap: 0f,
                sink: null, slidingWindow: 0, alibiSlopes: null);
        }

        /// <summary>rms(INT8 result - F32 result) / rms(F32 result) under <paramref name="options"/>.</summary>
        public double Nrmse(Int8AttentionV2Options options, Span<float> lse = default)
        {
            Output.AsSpan<float>().Clear();
            Int8AttentionV2Reference.Run(in Call, options, default, lse);
            AttentionHostReference.Run(in Reference);
            ReadOnlySpan<float> actual = Output.AsReadOnlySpan<float>(), expected = Expected.AsReadOnlySpan<float>();
            double error = 0, norm = 0;
            for (int i = 0; i < actual.Length; i++)
            {
                error += Math.Pow(actual[i] - expected[i], 2);
                norm += Math.Pow(expected[i], 2);
            }
            return Math.Sqrt(error / norm);
        }

        public void Dispose()
        {
            Query.Dispose();
            Key.Dispose();
            Value.Dispose();
            Output.Dispose();
            Expected.Dispose();
        }
    }

    [Theory]
    [InlineData(64, 300)]
    [InlineData(128, 300)]
    [InlineData(128, 200)]
    [InlineData(192, 300)]
    public void RotationKeepsAnOutlierChannelFromCostingTheResultItsAccuracy(int dim, int keys)
    {
        // One channel of Q and of K is twelve times the rest, the case the pinned library's own test uses.
        using Case outlier = new(4, 2, 5, keys, dim, (q, k) =>
        {
            for (int i = 3; i < q.Length; i += dim) q[i] *= 12f;
            for (int i = 3; i < k.Length; i += dim) k[i] *= 12f;
        });
        Int8AttentionV2Options plain = new() { SmoothKeys = false };
        double without = outlier.Nrmse(plain), with = outlier.Nrmse(plain with { Rotation = Int8Rotation.Pinned });
        Assert.True(with < 0.03, $"rotated: NRMSE {with:F4}");
        // The unrotated reference already scales each key and each query row on its own, so the gain is a fraction, not an
        // order of magnitude: 0.0085 to 0.0136 rotated against 0.0168 to 0.0190 unrotated over these four cases.
        Assert.True(with < without, $"rotation was meant to help: {with:F4} against {without:F4} unrotated.");
    }

    [Fact]
    public void TheRepresentativeKeyKeepsALargeCommonKeyComponentFromCostingTheResultItsAccuracy()
    {
        const int dim = 64, keys = 300;
        float[] common = Random(new Random(12), dim, 40f);
        using Case shifted = new(4, 2, 5, keys, dim, (_, k) =>
        {
            for (int i = 0; i < k.Length; i++) k[i] += common[i % dim];
        });
        Int8AttentionV2Options uncentered = new() { SmoothKeys = false };
        Int8AttentionV2Options anchored = new() { KeyCenter = Int8KeyCenter.RepresentativeKey };
        double without = shifted.Nrmse(uncentered), with = shifted.Nrmse(anchored);
        Assert.True(with < 0.03, $"representative key: NRMSE {with:F4}");
        Assert.True(with < without * 0.5, $"the representative key was meant to help: {with:F4} against {without:F4} uncentered.");
        // Every KV head found one, among the sampled rows.
        Int8AttentionV2Operands operands = Int8AttentionV2Reference.Quantize(in shifted.Call, anchored);
        Assert.All(operands.RepresentativeKeys, anchor => Assert.True(anchor >= 0));
    }

    [Fact]
    public void TheScoreASubtractedKeyRemovedIsAddedBackSoTheLseIsTheUnshiftedOne()
    {
        const int dim = 64, keys = 300, heads = 4, rows = 5;
        float[] common = Random(new Random(13), dim, 6f);
        using Case shifted = new(heads, 2, rows, keys, dim, (_, k) =>
        {
            for (int i = 0; i < k.Length; i++) k[i] += common[i % dim];
        });
        float[] lse = new float[heads * rows], expected = new float[heads * rows];
        shifted.Nrmse(new Int8AttentionV2Options { KeyCenter = Int8KeyCenter.RepresentativeKey, Rotation = Int8Rotation.Pinned }, lse);
        AttentionHostReference.Run(in shifted.Reference, expected);
        // Without the term added back, each row's LSE would be off by its own q . k_a, far more than this.
        for (int i = 0; i < lse.Length; i++)
            Assert.True(MathF.Abs(expected[i] - lse[i]) < 0.05f, $"row {i}: LSE {lse[i]} against {expected[i]}.");
    }

    [Fact]
    public void RotationAndTheRepresentativeKeyAreOffUnlessAsked()
    {
        using Case plain = new(4, 2, 3, 300, 64, static (_, _) => { });
        Int8AttentionV2Operands operands = Int8AttentionV2Reference.Quantize(in plain.Call, new Int8AttentionV2Options());
        Assert.Equal(Int8Rotation.None, operands.Rotation);
        Assert.All(operands.RepresentativeKeys, anchor => Assert.Equal(-1, anchor));
    }

    [Fact]
    public void ARepresentativeKeyCentersOneSegmentOnly()
    {
        using Case plain = new(4, 2, 3, 300, 64, static (_, _) => { });
        Int8AttentionV2Options options = new() { KeyCenter = Int8KeyCenter.RepresentativeKey, KeySegmentStarts = [100] };
        Assert.Throws<ArgumentException>(() => Int8AttentionV2Reference.Quantize(in plain.Call, options));
    }
}
