using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Utilities;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The family-agnostic reference-conditioning blocks: segment layouts and their row index, per-segment
/// modulation against the uniform fused ops, the reference key/value cache's concatenation and bookkeeping, and the
/// reference RoPE rows MOSS-Image appends.</summary>
public sealed unsafe class ReferenceConditioningTests
{
    [Fact]
    public void Layout_RowIndex_MarksEachSegmentsGroup()
    {
        using TokenSegmentLayout layout = new([3, 2, 1], [0, 1, 0]);
        Assert.Equal(6, layout.TotalLength);
        Assert.Equal(2, layout.GroupCount);
        Assert.Equal(new TokenSegment(3, 2, 1), layout.Segments[1]);
        int[] index = new ReadOnlySpan<int>((void*)layout.RowIndex.DataPointer, 6).ToArray();
        Assert.Equal([0, 0, 0, 1, 1, 0], index);
    }

    [Fact]
    public void Layout_RejectsGroupsWithGapsAndEmptySegments()
    {
        Assert.Throws<ArgumentException>(() => new TokenSegmentLayout([2, 2], [0, 2]));
        Assert.Throws<ArgumentOutOfRangeException>(() => new TokenSegmentLayout([2, 0], [0, 1]));
        Assert.Throws<ArgumentException>(() => new TokenSegmentLayout([2], [0, 1]));
    }

    [Fact]
    public void SegmentModulation_MatchesTheUniformOps_WhenEveryGroupSharesRows()
    {
        using IBackend backend = new CpuBackend();
        const int rows = 5, dim = 8;
        using Tensor x = Random(new TensorShape(1, rows, dim), 1);
        using Tensor weight = Random(new TensorShape(dim), 2);
        using Tensor shift = Random(new TensorShape(1, dim), 3);
        using Tensor scale = Random(new TensorShape(1, dim), 4);
        using Tensor gate = Random(new TensorShape(1, dim), 5);
        using Tensor value = Random(new TensorShape(1, rows, dim), 6);
        using TokenSegmentLayout layout = new([3, 2], [0, 1]);
        using SegmentModulation modulation = SegmentModulation.Build(backend, [(shift, scale, gate), (shift, scale, gate)], layout);

        using Tensor fused = new(x.Shape, DType.F32);
        backend.RmsNormModulate(fused, x, weight, scale, shift, 1e-6f);
        using Tensor normed = new(x.Shape, DType.F32);
        backend.RmsNorm(normed, x, weight, 1e-6f);
        using Tensor segmented = modulation.Affine(backend, normed);
        AssertClose(fused, segmented, 1e-5f);

        using Tensor uniformResidual = new(x.Shape, DType.F32);
        backend.GatedResidualLastDim(uniformResidual, x, value, gate);
        using Tensor segmentResidual = modulation.GatedResidual(backend, x, value);
        AssertClose(uniformResidual, segmentResidual, 0f);
    }

    [Fact]
    public void SegmentModulation_GivesEachSegmentItsOwnRows()
    {
        using IBackend backend = new CpuBackend();
        const int dim = 4;
        using Tensor x = Random(new TensorShape(1, 3, dim), 7);
        using Tensor live = Random(new TensorShape(1, dim), 8);
        using Tensor reference = Random(new TensorShape(1, dim), 9);
        using TokenSegmentLayout layout = new([2, 1], [0, 1]);
        using SegmentModulation modulation = SegmentModulation.Build(backend, [(live, live, live), (reference, reference, reference)], layout);
        using Tensor result = modulation.Affine(backend, x);
        float* px = (float*)x.DataPointer, pr = (float*)result.DataPointer, pl = (float*)live.DataPointer, pf = (float*)reference.DataPointer;
        for (int t = 0; t < 3; t++)
        {
            float* row = t < 2 ? pl : pf;
            for (int d = 0; d < dim; d++)
                Assert.Equal(px[t * dim + d] * (1f + row[d]) + row[d], pr[t * dim + d], 6);
        }
        Assert.Throws<ArgumentException>(() => modulation.Affine(backend, Random(new TensorShape(1, 4, dim), 10)));
    }

    [Fact]
    public void KvCache_ConcatTokens_PutsLiveTokensFirst()
    {
        using IBackend backend = new CpuBackend();
        using Tensor live = Random(new TensorShape(1, 2, 3, 4), 11);
        using Tensor cached = Random(new TensorShape(1, 5, 3, 4), 12);
        using Tensor joined = ReferenceKvCache.ConcatTokens(backend, live, cached);
        Assert.Equal(new TensorShape(1, 7, 3, 4), joined.Shape);
        ReadOnlySpan<float> j = new((void*)joined.DataPointer, (int)joined.ElementCount);
        Assert.True(j[..24].SequenceEqual(new ReadOnlySpan<float>((void*)live.DataPointer, 24)));
        Assert.True(j[24..].SequenceEqual(new ReadOnlySpan<float>((void*)cached.DataPointer, 60)));
        using Tensor wrongHeads = Random(new TensorShape(1, 5, 2, 4), 13);
        Assert.Throws<ArgumentException>(() => ReferenceKvCache.ConcatTokens(backend, live, wrongHeads));
    }

    [Fact]
    public void KvCache_Builder_TracksLayersAndRefusesMismatchedTokens()
    {
        using IBackend backend = new CpuBackend();
        Assert.Equal(2L * 40 * 4096 * 3072 * 2, ReferenceKvCache.EstimateBytes(40, 4096, 3072, DType.F16));
        using ReferenceKvCache.Builder builder = new(3);
        builder.Set(0, Random(new TensorShape(1, 4, 2, 8), 14), Random(new TensorShape(1, 4, 2, 8), 15));
        Assert.Throws<ArgumentException>(() => builder.Set(2, Random(new TensorShape(1, 5, 2, 8), 16), Random(new TensorShape(1, 5, 2, 8), 17)));
        builder.Set(2, Random(new TensorShape(1, 4, 2, 8), 18), Random(new TensorShape(1, 4, 2, 8), 19));
        ReferenceKvCache cache = builder.Build(backend, 42, "test");
        Assert.Equal(4, cache.Tokens);
        Assert.Equal(3, cache.Layers);
        Assert.Equal(42, cache.Fingerprint);
        Assert.NotNull(cache.Keys(0));
        Assert.Null(cache.Keys(1));
        Assert.NotNull(cache.Values(2));
        cache.Dispose();
        cache.Dispose();
        Assert.Null(cache.Keys(0));
    }

    [Fact]
    public void CachePolicy_AcceptsBackendsWithoutDeviceMemory()
    {
        using IBackend backend = new CpuBackend();
        Assert.True(ReferenceCachePolicy.Check(backend, ReferenceConditioningOptions.Default, long.MaxValue / 4, 3, "test"));
        Assert.True(ReferenceCachePolicy.Check(backend, new ReferenceConditioningOptions { Mode = ReferenceAttentionMode.Joint }, 1, 1, "test"));
    }

    [Fact]
    public void MossRope_ReferenceRows_FollowTheTextAndMatchThePrepassTables()
    {
        (int, int)[] grids = [(2, 2), (1, 3)];
        (Tensor jointCos, Tensor jointSin) = MossImageRope.Joint(3, 4, 5, grids, 32, 10000f);
        (Tensor refCos, Tensor refSin) = MossImageRope.References(grids, 5, 32, 10000f);
        (Tensor plainCos, Tensor plainSin) = MossImageRope.Joint(3, 4, 5, 32, 10000f);
        try
        {
            Assert.Equal(new TensorShape(1, 12 + 5 + 7, 32), jointCos.Shape);
            ReadOnlySpan<float> joint = new((void*)jointCos.DataPointer, (int)jointCos.ElementCount);
            ReadOnlySpan<float> plain = new((void*)plainCos.DataPointer, (int)plainCos.ElementCount);
            ReadOnlySpan<float> refs = new((void*)refCos.DataPointer, (int)refCos.ElementCount);
            Assert.True(joint[..plain.Length].SequenceEqual(plain));
            Assert.True(joint[plain.Length..].SequenceEqual(refs));
            // The first band of axis r is unrotated by phase r·1: reference 1 rows read cos(1), reference 2 rows cos(2).
            Assert.Equal(MathF.Cos(1f), refs[0], 6);
            Assert.Equal(MathF.Cos(2f), refs[4 * 32], 6);
        }
        finally
        {
            jointCos.Dispose();
            jointSin.Dispose();
            refCos.Dispose();
            refSin.Dispose();
            plainCos.Dispose();
            plainSin.Dispose();
        }
    }

    private static Tensor Random(TensorShape shape, int seed)
    {
        Tensor t = new(shape, DType.F32);
        Random rng = new(seed);
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++) p[i] = (float)(rng.NextDouble() * 2 - 1);
        return t;
    }

    private static void AssertClose(Tensor expected, Tensor actual, float tolerance)
    {
        Assert.Equal(expected.ElementCount, actual.ElementCount);
        float* e = (float*)expected.DataPointer, a = (float*)actual.DataPointer;
        for (long i = 0; i < expected.ElementCount; i++)
            Assert.True(MathF.Abs(e[i] - a[i]) <= tolerance, $"element {i}: {e[i]} vs {a[i]}");
    }
}
