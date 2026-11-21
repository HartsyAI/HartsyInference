using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Tensors;
using HartsyInference.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.LLM.Tests.Parity;

/// <summary>The Vulkan query side of INT8 attention against <see cref="Int8AttentionV2Reference.Quantize"/>, and the exact
/// fused multiply-add it rests on (Shaders/fma_rn.glsl) against <see cref="MathF.FusedMultiplyAdd"/>.
/// <para>The bounds were fixed before the first run. The fused multiply-add: bit-equal for every triple whose
/// multiplicands are zero or within [2^-40, 2^40] and whose addend is zero or within [2^-80, 2^80]. The device's own
/// <c>fma()</c> is counted and reported, not asserted. The query side: every dot and scale bit-equal and every byte equal
/// to the reference's, on finite operands of ordinary magnitude.</para></summary>
[Trait("Category", "GpuIntegration")]
public sealed class Int8AttentionQueriesVulkanTests(ITestOutputHelper output)
{
    private static Tensor Random(TensorShape shape, int seed, float shared)
    {
        Tensor tensor = new(shape, DType.F32);
        Random random = new(seed);
        int width = (int)shape[shape.Rank - 1];
        long rows = shape.ElementCount / width / shape[shape.Rank - 2];
        float[] common = new float[rows * width];
        for (int i = 0; i < common.Length; i++) common[i] = (random.NextSingle() * 2f - 1f) * shared;
        Span<float> values = tensor.AsSpan<float>();
        int perHead = (int)shape[shape.Rank - 2] * width;
        for (int i = 0; i < values.Length; i++)
            values[i] = random.NextSingle() * 2f - 1f + common[i / perHead * width + i % width];
        return tensor;
    }

    private static int Misses(ReadOnlySpan<float> expected, ReadOnlySpan<float> actual, out int first)
    {
        Assert.Equal(expected.Length, actual.Length);
        int misses = 0;
        first = -1;
        for (int i = 0; i < expected.Length; i++)
        {
            bool same = float.IsNaN(expected[i]) ? float.IsNaN(actual[i])
                : BitConverter.SingleToInt32Bits(expected[i]) == BitConverter.SingleToInt32Bits(actual[i]);
            if (same) continue;
            if (first < 0) first = i;
            misses++;
        }
        return misses;
    }

    private static void AssertBits(ReadOnlySpan<float> expected, ReadOnlySpan<float> actual, string what)
    {
        int misses = Misses(expected, actual, out int first);
        Assert.True(misses == 0, $"{what}: {misses} of {expected.Length} differ; element {first} is "
            + $"{(first < 0 ? 0f : actual[first]):R}, expected {(first < 0 ? 0f : expected[first]):R}");
    }

    private static void AssertBytes(ReadOnlySpan<sbyte> expected, ReadOnlySpan<sbyte> actual, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        int misses = 0, first = -1;
        for (int i = 0; i < expected.Length; i++)
            if (expected[i] != actual[i])
            {
                if (first < 0) first = i;
                misses++;
            }
        Assert.True(misses == 0, $"{what}: {misses} of {expected.Length} differ; element {first} is "
            + $"{(first < 0 ? 0 : actual[first])}, expected {(first < 0 ? 0 : expected[first])}");
    }

    private static float Pick(Random random, int minExponent, int maxExponent)
    {
        float mantissa = 1f + random.Next(1 << 23) / (float)(1 << 23);
        float value = MathF.ScaleB(mantissa, random.Next(minExponent, maxExponent + 1));
        return random.Next(2) == 0 ? value : -value;
    }

    private static bool InRange(float a, float b, float c)
    {
        static bool Ok(float x, int bound) => x == 0f || (float.IsFinite(x) && MathF.Abs(x) >= MathF.ScaleB(1f, -bound)
            && MathF.Abs(x) <= MathF.ScaleB(1f, bound));
        return Ok(a, 40) && Ok(b, 40) && Ok(c, 80);
    }

    /// <summary>Triples in the classes a double rounding gets wrong: sums that land on or beside the midpoint of two floats
    /// (with the addend near the product, and with it far above), a product cancelled by its own rounding, zeros, and
    /// ordinary random ones.</summary>
    private static (float[] A, float[] B, float[] C, string[] Kind) Triples()
    {
        List<float> a = [], b = [], c = [];
        List<string> kind = [];
        void Add(float x, float y, float z, string k)
        {
            if (!InRange(x, y, z)) return;
            a.Add(x);
            b.Add(y);
            c.Add(z);
            kind.Add(k);
        }
        Random random = new(20261120);
        for (int i = 0; i < 20000; i++) Add(Pick(random, -20, 20), Pick(random, -20, 20), Pick(random, -40, 40), "random");
        int[] gaps = [-3, -2, -1, 0, 1, 2, 3, 20, 21, 22, 23, 24, 25, 26];
        for (int i = 0; i < 40000; i++)
        {
            float x = Pick(random, -10, 10), y = Pick(random, -10, 10);
            double product = (double)x * y;
            int exponent = Math.ILogB(product) + gaps[i % gaps.Length];
            float sum = MathF.ScaleB(1f + random.Next(1 << 23) / (float)(1 << 23), exponent);
            double half = ((double)MathF.BitIncrement(sum) - sum) / 2;
            double mid = random.Next(2) == 0 ? sum + half : sum - half;
            if (random.Next(2) == 0) mid = -mid;
            Add(x, y, (float)(mid - product), gaps[i % gaps.Length] < 10 ? "tie, addend near the product" : "tie, addend far above");
        }
        for (int i = 0; i < 10000; i++)
        {
            float x = Pick(random, -10, 10), y = Pick(random, -10, 10), p = x * y;
            float z = (i % 3) switch { 0 => -p, 1 => -MathF.BitIncrement(p), _ => -MathF.BitDecrement(p) };
            Add(x, y, z, "cancellation");
        }
        // The class a double rounding gets wrong, by construction: the rounded product and the addend sum to an exact
        // midpoint while the product itself was not exact, so only the part the product lost says which way to go.
        // (1 + 2^-j)^2 = 1 + 2^(1-j) + 2^-2j rounds down to 1 + 2^(1-j); (1 + 2^-j)(1 - 2^-j) = 1 - 2^-2j rounds up to 1.
        for (int i = 0; i < 24000; i++)
        {
            int j = 13 + i % 11, shift = random.Next(-10, 11);
            bool square = i % 2 == 0;
            float x = MathF.ScaleB(1f + MathF.ScaleB(1f, -j), shift), y = square ? 1f + MathF.ScaleB(1f, -j) : 1f - MathF.ScaleB(1f, -j);
            float mantissa = 1f + random.Next(1 << 23) / (float)(1 << 23);
            float z = MathF.ScaleB(mantissa, (square ? 25 - j : 24) + shift);
            if (random.Next(2) == 0) x = -x;
            if (random.Next(2) == 0) z = -z;
            Add(x, y, z, square ? "exact tie, product rounded down" : "exact tie, product rounded up");
        }
        for (int i = 0; i < 2000; i++)
        {
            float x = Pick(random, -20, 20), y = Pick(random, -20, 20), z = Pick(random, -40, 40);
            Add(i % 2 == 0 ? 0f : x, y, z, "zero product");
            Add(x, y, 0f, "zero addend");
        }
        return ([.. a], [.. b], [.. c], [.. kind]);
    }

    [VulkanFact]
    public void FusedMultiplyAddMatchesTheHost()
    {
        (float[] a, float[] b, float[] c, string[] kind) = Triples();
        int n = a.Length;
        float[] expected = new float[n];
        for (int i = 0; i < n; i++) expected[i] = MathF.FusedMultiplyAdd(a[i], b[i], c[i]);

        using VulkanBackend backend = new();
        using Tensor ta = new(new TensorShape(n), DType.F32), tb = new(ta.Shape, DType.F32), tc = new(ta.Shape, DType.F32);
        a.CopyTo(ta.AsSpan<float>());
        b.CopyTo(tb.AsSpan<float>());
        c.CopyTo(tc.AsSpan<float>());
        using Tensor emulated = new(ta.Shape, DType.F32), native = new(ta.Shape, DType.F32);
        backend.FusedMultiplyAddProbe(emulated, native, ta, tb, tc);
        backend.Sync();
        ReadOnlySpan<float> e = emulated.AsReadOnlySpan<float>(), d = native.AsReadOnlySpan<float>();

        Dictionary<string, (int Count, int Emulated, int Native)> byKind = [];
        int firstMiss = -1;
        for (int i = 0; i < n; i++)
        {
            (int count, int em, int na) = byKind.GetValueOrDefault(kind[i]);
            bool emOk = BitConverter.SingleToInt32Bits(e[i]) == BitConverter.SingleToInt32Bits(expected[i]);
            bool naOk = BitConverter.SingleToInt32Bits(d[i]) == BitConverter.SingleToInt32Bits(expected[i]);
            if (!emOk && firstMiss < 0) firstMiss = i;
            byKind[kind[i]] = (count + 1, em + (emOk ? 0 : 1), na + (naOk ? 0 : 1));
        }
        int misses = 0;
        foreach ((string name, (int count, int em, int na)) in byKind)
        {
            output.WriteLine($"{name}: {count} triples; fmaRn differs in {em}, the device fma() in {na}");
            misses += em;
        }
        Assert.True(misses == 0, firstMiss < 0 ? "" : $"fmaRn differs in {misses} of {n}; first is ({a[firstMiss]:R}, "
            + $"{b[firstMiss]:R}, {c[firstMiss]:R}) [{kind[firstMiss]}]: {e[firstMiss]:R}, expected {expected[firstMiss]:R}");
    }

    private static void RunQueries(int batch, int kvHeads, int group, int headDim, Int8Rotation rotation, int[] rows,
        bool smooth, string what)
    {
        const int queryRows = 37, keys = 200;
        int heads = kvHeads * group;
        using Tensor query = Random(new TensorShape(batch, heads, queryRows, headDim), 61 + headDim + group, 0.5f);
        using Tensor key = Random(new TensorShape(batch, kvHeads, keys, headDim), 62 + headDim + group, 3f);
        using Tensor value = Random(key.Shape, 63 + headDim, 1f);
        using Tensor result = new(query.Shape, DType.F32);
        AttentionCall call = AttentionCall.FromFlashAttention(result, query, key, value, keys, false, 0,
            1f / MathF.Sqrt(headDim), 0f, null, 0, null);
        Int8AttentionV2Operands expected = Int8AttentionV2Reference.Quantize(in call, new Int8AttentionV2Options
        {
            SmoothKeys = smooth, KeyCenter = Int8KeyCenter.SegmentMean, Rotation = rotation,
            KeyScales = Int8KeyScaleGrouping.PerKey, Probabilities = Int8ProbabilityFormat.U8,
        }, rows);
        int taken = rows.Length == 0 ? queryRows : rows.Length;

        using VulkanBackend backend = new();
        using Tensor keyRows = new(new TensorShape(batch * kvHeads, keys, headDim), DType.F32);
        key.AsReadOnlySpan<float>().CopyTo(keyRows.AsSpan<float>());
        using Tensor keyBytes = new(new TensorShape(batch * kvHeads * keys, headDim), DType.I8);
        using Tensor keyScales = new(new TensorShape(batch * kvHeads * keys), DType.F32);
        using Tensor means = new(new TensorShape(batch * kvHeads, headDim), DType.F32);
        if (smooth) backend.QuantizeInt8AttentionKeys(keyBytes, keyScales, means, keyRows, rotation);

        // The rows the call attends, gathered head by head as the reference indexes them.
        using Tensor queryTaken = new(new TensorShape(batch * heads, taken, headDim), DType.F32);
        ReadOnlySpan<float> q = query.AsReadOnlySpan<float>();
        Span<float> gathered = queryTaken.AsSpan<float>();
        for (int bh = 0; bh < batch * heads; bh++)
            for (int ri = 0; ri < taken; ri++)
                q.Slice((bh * queryRows + (rows.Length == 0 ? ri : rows[ri])) * headDim, headDim)
                    .CopyTo(gathered.Slice((bh * taken + ri) * headDim, headDim));
        using Tensor bytes = new(new TensorShape(batch * heads * taken, headDim), DType.I8);
        using Tensor scales = new(new TensorShape(batch * heads * taken), DType.F32);
        using Tensor dots = new(new TensorShape(batch * heads, taken), DType.F32);
        backend.QuantizeInt8AttentionQueries(bytes, scales, smooth ? dots : null, queryTaken, smooth ? means : null, rotation);
        backend.Sync();
        if (smooth)
        {
            Assert.Contains(expected.QueryMeanDots, x => x != 0f);
            AssertBits(expected.QueryMeanDots, dots.AsReadOnlySpan<float>(), $"query mean dots, {what}");
        }
        AssertBits(expected.QueryScales, scales.AsReadOnlySpan<float>(), $"query scales, {what}");
        AssertBytes(expected.Queries, bytes.AsReadOnlySpan<sbyte>(), $"query bytes, {what}");
    }

    [VulkanTheory]
    [InlineData(1, 1, 64, Int8Rotation.None)]
    [InlineData(2, 2, 64, Int8Rotation.H64)]
    [InlineData(1, 4, 128, Int8Rotation.SignedH128)]
    [InlineData(2, 1, 128, Int8Rotation.H4)]
    [InlineData(1, 2, 256, Int8Rotation.PaddedH128)]
    [InlineData(2, 3, 256, Int8Rotation.H4)]
    public void QueriesAndTheirMeanDotsEqualTheReference(int batch, int group, int headDim, Int8Rotation rotation) =>
        RunQueries(batch, 2, group, headDim, rotation, [], true, $"B {batch}, group {group}, D {headDim}, {rotation}");

    /// <summary>A call that attends some of its rows: the dots, scales and bytes of those rows only.</summary>
    [VulkanFact]
    public void ASubsetOfRowsEqualsTheReference() =>
        RunQueries(2, 2, 2, 128, Int8Rotation.SignedH128, [3, 17, 36], true, "rows 3, 17 and 36");

    /// <summary>Keys that are not centered: no means and no dots, the query bytes and scales alone.</summary>
    [VulkanFact]
    public void UncenteredQueriesEqualTheReference() =>
        RunQueries(1, 2, 2, 64, Int8Rotation.H64, [], false, "no key centering");
}
