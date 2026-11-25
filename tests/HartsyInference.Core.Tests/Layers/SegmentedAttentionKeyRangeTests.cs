using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Core.Tests.Layers;

/// <summary><see cref="SegmentedAttention.RunHeadMajor"/> against the dense attention it replaces: one call with an additive bias of 0 on the
/// visible pairs and negative infinity elsewhere.</summary>
public sealed class SegmentedAttentionKeyRangeTests
{
    private const int Heads = 3, HeadDim = 8, Batch = 2;
    private const float Tolerance = 1e-5f;
    private static readonly IBackend Backend = new CpuBackend();

    public static TheoryData<string> Plans => new() { "cached", "groups", "prefix", "shared", "returning", "dense", "single" };

    private static (int Queries, int Keys, KeyRangeSegment[] Segments) Plan(string name) => name switch
    {
        // A history of 10 rows, then 6 current ones: the text reads the history and itself, the noisy rows read everything, padding reads itself.
        "cached" => (8, 18, [new(0, 2, [new(0, 4), new(10, 2)]), KeyRangeSegment.Of(2, 4, 0, 16), KeyRangeSegment.Of(6, 2, 16, 2)]),
        // Duo's uncached shape: a group reads several separate spans of one sequence.
        "groups" => (12, 12, [KeyRangeSegment.Of(0, 3, 0, 3), new(3, 4, [new(0, 3), new(5, 2)]), new(7, 5, [new(0, 1), new(4, 2), new(9, 3)])]),
        // LongCat's cached attention: only the noisy queries run, over the cached and the noisy keys.
        "prefix" => (5, 14, [KeyRangeSegment.Of(0, 5, 0, 14)]),
        // Frame-causal (MovieRoll's VAE) with two segments that name the same keys.
        "shared" => (9, 9, [KeyRangeSegment.Of(0, 3, 0, 3), KeyRangeSegment.Of(3, 2, 0, 6), KeyRangeSegment.Of(5, 1, 0, 6), KeyRangeSegment.Of(6, 3, 0, 9)]),
        // The same ranges again after a segment that reads others: not neighbours, so two calls.
        "returning" => (6, 10, [KeyRangeSegment.Of(0, 2, 1, 4), KeyRangeSegment.Of(2, 1, 6, 4), KeyRangeSegment.Of(3, 3, 1, 4)]),
        "dense" => (7, 7, [KeyRangeSegment.Of(0, 4, 0, 7), KeyRangeSegment.Of(4, 3, 0, 7)]),
        _ => (1, 11, [new(0, 1, [new(2, 1), new(7, 3)])]),
    };

    [Theory]
    [MemberData(nameof(Plans))]
    public void MatchesTheDenseBias(string name)
    {
        (int queries, int keys, KeyRangeSegment[] segments) = Plan(name);
        Random rng = new(5);
        using Tensor q = Filled(rng, queries), k = Filled(rng, keys), v = Filled(rng, keys), bias = Bias(queries, keys, segments);
        AttentionLayer layer = Layer();

        using Tensor expected = layer.AttendHeadMajorOperands(Backend, q, k, v, new AttentionCallContext { Bias = bias }, OperandOwnership.None);
        using Tensor actual = SegmentedAttention.RunHeadMajor(Backend, layer, q, k, v, segments);

        Assert.Equal(expected.Shape, actual.Shape);
        ReadOnlySpan<float> e = expected.AsReadOnlySpan<float>(), a = actual.AsReadOnlySpan<float>();
        for (int i = 0; i < e.Length; i++)
            Assert.True(MathF.Abs(e[i] - a[i]) <= Tolerance, $"{name}[{i}]: segments {a[i]} vs bias {e[i]}.");
    }

    [Theory]
    [MemberData(nameof(Plans))]
    public void AKeyOutsideASegmentCannotReachIt_AndOneInsideDoes(string name)
    {
        (int queries, int keys, KeyRangeSegment[] segments) = Plan(name);
        Random rng = new(6);
        using Tensor q = Filled(rng, queries), k = Filled(rng, keys), v = Filled(rng, keys);
        float[] before = Run(q, k, v, segments);
        for (int key = 0; key < keys; key++)
        {
            using Tensor poisonedKey = Poisoned(k, key, keys);
            using Tensor poisonedValue = Poisoned(v, key, keys);
            float[] afterKey = Run(q, poisonedKey, v, segments);
            float[] afterValue = Run(q, k, poisonedValue, segments);
            foreach (KeyRangeSegment s in segments)
            {
                bool reads = s.Keys.Any(range => key >= range.Start && key < range.End);
                Assert.Equal(reads, Differs(before, afterValue, s, queries));
                // A segment with one key gives it all the weight, so its score cannot move the output.
                Assert.Equal(reads && s.KeyCount > 1, Differs(before, afterKey, s, queries));
            }
        }
    }

    [Fact]
    public void TheOperandsStayTheCallers()
    {
        (int queries, int keys, KeyRangeSegment[] segments) = Plan("cached");
        Random rng = new(7);
        using Tensor q = Filled(rng, queries), k = Filled(rng, keys), v = Filled(rng, keys);
        float[] qBefore = q.AsReadOnlySpan<float>().ToArray(), kBefore = k.AsReadOnlySpan<float>().ToArray(), vBefore = v.AsReadOnlySpan<float>().ToArray();

        using Tensor result = SegmentedAttention.RunHeadMajor(Backend, Layer(), q, k, v, segments);

        Assert.Equal(qBefore, q.AsReadOnlySpan<float>().ToArray());
        Assert.Equal(kBefore, k.AsReadOnlySpan<float>().ToArray());
        Assert.Equal(vBefore, v.AsReadOnlySpan<float>().ToArray());
    }

    [Fact]
    public void Validate_RefusesAMalformedPlan()
    {
        static void Refused(int queries, int keys, params KeyRangeSegment[] segments) =>
            Assert.Throws<ArgumentException>(() => KeyRangeSegment.Validate(segments, queries, keys));

        Refused(4, 4);
        Refused(4, 4, KeyRangeSegment.Of(0, 3, 0, 4));
        Refused(4, 4, KeyRangeSegment.Of(1, 3, 0, 4));
        Refused(4, 4, KeyRangeSegment.Of(0, 4, 0, 5));
        Refused(4, 4, KeyRangeSegment.Of(0, 4, -1, 2));
        Refused(4, 4, new KeyRangeSegment(0, 4, []));
        Refused(4, 4, new KeyRangeSegment(0, 4, [new(0, 3), new(2, 2)]));
        Refused(4, 4, new KeyRangeSegment(0, 4, [new(2, 2), new(0, 1)]));
        Refused(4, 4, new KeyRangeSegment(0, 4, [new(0, 0)]));
        Refused(4, 4, KeyRangeSegment.Of(0, 2, 0, 4), KeyRangeSegment.Of(2, 3, 0, 4));
        KeyRangeSegment.Validate([KeyRangeSegment.Of(0, 2, 0, 9), new(2, 2, [new(0, 3), new(5, 4)])], 4, 9);
    }

    private static AttentionLayer Layer() =>
        new(new AttentionLayerConfig { Heads = Heads, KvHeads = Heads, HeadDim = HeadDim, ComputeDType = DType.F32 });

    private static float[] Run(Tensor q, Tensor k, Tensor v, KeyRangeSegment[] segments)
    {
        using Tensor result = SegmentedAttention.RunHeadMajor(Backend, Layer(), q, k, v, segments);
        return result.AsReadOnlySpan<float>().ToArray();
    }

    private static bool Differs(float[] before, float[] after, KeyRangeSegment segment, int queries)
    {
        for (int b = 0; b < Batch; b++)
            for (int h = 0; h < Heads; h++)
                for (int row = segment.QueryStart; row < segment.QueryStart + segment.QueryCount; row++)
                    for (int d = 0; d < HeadDim; d++)
                    {
                        int at = ((b * Heads + h) * queries + row) * HeadDim + d;
                        if (before[at] != after[at]) return true;
                    }
        return false;
    }

    private static Tensor Filled(Random rng, int tokens)
    {
        Tensor t = new(new TensorShape(Batch, Heads, tokens, HeadDim), DType.F32);
        foreach (ref float value in t.AsSpan<float>()) value = (float)(rng.NextDouble() * 2 - 1);
        return t;
    }

    private static Tensor Poisoned(Tensor source, int key, int keys)
    {
        Tensor copy = new(source.Shape, DType.F32);
        source.AsReadOnlySpan<float>().CopyTo(copy.AsSpan<float>());
        Span<float> values = copy.AsSpan<float>();
        for (int block = 0; block < Batch * Heads; block++)
            values.Slice((block * keys + key) * HeadDim, HeadDim).Fill(50f);
        return copy;
    }

    private static Tensor Bias(int queries, int keys, KeyRangeSegment[] segments)
    {
        Tensor bias = new(new TensorShape(queries, keys), DType.F32);
        Span<float> values = bias.AsSpan<float>();
        values.Fill(float.NegativeInfinity);
        foreach (KeyRangeSegment s in segments)
            for (int row = s.QueryStart; row < s.QueryStart + s.QueryCount; row++)
                foreach (AttentionKeyRange range in s.Keys)
                    values.Slice(row * keys + range.Start, range.Count).Fill(0f);
        return bias;
    }
}
