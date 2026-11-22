using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.ThreeD.Models.CoLa3D;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.ThreeD.Tests.CoLa3D;

/// <summary><see cref="CoLa3DBlock.Attend"/> attends through a shared <see cref="Core.Layers.Attention.AttentionLayer"/> and merges
/// the heads with <c>MergeHeads</c>. It used to call <c>ScaledDotProductAttention(..., null, scale)</c> and permute back itself; that body is
/// kept here as the oracle. On the CPU backend the new path must be bit-identical to it, take the same row the same number of times
/// without the F16 permission, and dispose q, k and v like the old body did. (The CUDA backend is compared in the GpuIntegration theory below; Vulkan is compared on a real generation, not here.)</summary>
[Collection("AttentionEvidence")]
public sealed class CoLa3DBlockAttendTests(ITestOutputHelper output)
{
    private static void AssertDisposed(Tensor t) => Assert.Throws<ObjectDisposedException>(() => _ = t.AsSpan<float>().Length);

    private static Tensor Filled(TensorShape shape, int seed)
    {
        Tensor tensor = new(shape, DType.F32);
        Random random = new(seed);
        Span<float> values = tensor.AsSpan<float>();
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = random.NextSingle() * 2f - 1f;
        }

        return tensor;
    }

    /// <summary>The deleted body: the legacy dense entry without a mask and without the F16 permission, the permute back.</summary>
    private static Tensor Former(IBackend backend, Tensor q, Tensor k, Tensor v, int tokens, int heads, int headDim, float scale)
    {
        Tensor att = new(new TensorShape(1, heads, tokens, headDim), DType.F32);
        backend.ScaledDotProductAttention(att, q, k, v, null, scale);
        q.Dispose();
        k.Dispose();
        v.Dispose();
        Tensor flat = new(new TensorShape(1, tokens, (long)heads * headDim), DType.F32);
        backend.Permute0213(flat, att, heads, tokens, headDim);
        att.Dispose();
        return flat;
    }

    private static string Rows(AttentionEvidenceEvent[] events) => string.Join("; ", events
        .Where(e => e.Kind == AttentionEvidenceKind.Execution)
        .GroupBy(e => $"{e.Implementation} {e.Route} {e.QueryType}/{e.KvType} h{e.Heads}/{e.KvHeads} q{e.QueryTokens} k{e.KeyCapacity} d{e.QkHeadDim} "
            + $"f16permission={e.Numerics.LegacyAssertedBounded}")
        .OrderBy(g => g.Key, StringComparer.Ordinal).Select(g => $"{g.Sum(e => e.Count)} x {g.Key}"));

    [Theory]
    [InlineData(1, 1)]
    [InlineData(5, 5)]
    [InlineData(37, 37)]
    [InlineData(23, 6)]
    [InlineData(9, 41)]
    public void OnTheCpuTheLayerPathIsBitIdenticalToTheFormerBodyAndTakesTheSameRow(int queryTokens, int keyTokens)
    {
        CoLa3DConfig config = CoLa3DTestSupport.Tiny(CoLa3DConfig.QuerySourceKind.Fixed);
        Dictionary<string, Tensor> tensors = CoLa3DTestSupport.RandomWeights(config, 5);
        using CoLa3DBlock block = new(tensors, 0, config);
        int heads = config.NumHeads, headDim = config.HeadDim;
        float scale = 1f / MathF.Sqrt(headDim);
        using CpuBackend backend = new();
        TensorShape qShape = new(1, heads, queryTokens, headDim), kvShape = new(1, heads, keyTokens, headDim);
        Tensor q1 = Filled(qShape, 1), k1 = Filled(kvShape, 2), v1 = Filled(kvShape, 3);
        Tensor q2 = Filled(qShape, 1), k2 = Filled(kvShape, 2), v2 = Filled(kvShape, 3);

        string formerRows, layerRows;
        Tensor former, layer;
        using (AttentionEvidenceSession evidence = AttentionEvidence.Begin(64))
        {
            former = Former(backend, q1, k1, v1, queryTokens, heads, headDim, scale);
            formerRows = Rows(evidence.Snapshot());
        }

        using (AttentionEvidenceSession evidence = AttentionEvidence.Begin(64))
        {
            layer = block.Attend(backend, q2, k2, v2, queryTokens);
            layerRows = Rows(evidence.Snapshot());
        }

        using (former)
        using (layer)
        {
            Assert.Equal(former.Shape, layer.Shape);
            ReadOnlySpan<uint> expected = System.Runtime.InteropServices.MemoryMarshal.Cast<float, uint>(former.AsReadOnlySpan<float>());
            ReadOnlySpan<uint> actual = System.Runtime.InteropServices.MemoryMarshal.Cast<float, uint>(layer.AsReadOnlySpan<float>());
            for (int i = 0; i < expected.Length; i++)
            {
                Assert.True(expected[i] == actual[i], $"element {i}: {expected[i]:x8} against {actual[i]:x8}");
            }
        }

        Assert.Equal(formerRows, layerRows);
        Assert.Contains("f16permission=False", layerRows);
        AssertDisposed(q2);
        AssertDisposed(k2);
        AssertDisposed(v2);
        CoLa3DTestSupport.Dispose(tensors.Values);
    }

    /// <summary>The CUDA backend at the shapes of the real decomposer (12 heads of 64: queries over queries, queries over points, points
    /// over queries, and a square one). The former body runs twice to show that the call is
    /// deterministic on this device; the layer path must then be bit-identical to it and take the same rows.</summary>
    [Theory]
    [Trait("Category", "GpuIntegration")]
    [InlineData(20, 20, 12, 64)]
    [InlineData(20, 690, 12, 64)]
    [InlineData(690, 20, 12, 64)]
    [InlineData(690, 690, 12, 64)]
    public void OnCudaTheLayerPathIsBitIdenticalToTheFormerBodyAndTakesTheSameRow(int queryTokens, int keyTokens, int heads, int headDim)
    {
        if (!CudaContext.IsAvailable())
        {
            output.WriteLine("SKIPPED: CUDA unavailable");
            return;
        }

        CoLa3DConfig config = CoLa3DTestSupport.Tiny(CoLa3DConfig.QuerySourceKind.Fixed) with { HiddenDim = heads * headDim, NumHeads = heads, NumLayers = 1 };
        Dictionary<string, Tensor> tensors = CoLa3DTestSupport.RandomWeights(config, 5);
        using CoLa3DBlock block = new(tensors, 0, config);
        float scale = 1f / MathF.Sqrt(headDim);
        using CudaBackend backend = new(0, PtxDirectory());
        TensorShape qShape = new(1, heads, queryTokens, headDim), kvShape = new(1, heads, keyTokens, headDim);

        Tensor[] results = new Tensor[3];
        string[] rows = new string[3];
        for (int run = 0; run < 3; run++)
        {
            Tensor q = Filled(qShape, 1), k = Filled(kvShape, 2), v = Filled(kvShape, 3);
            using AttentionEvidenceSession evidence = AttentionEvidence.Begin(256);
            results[run] = run < 2 ? Former(backend, q, k, v, queryTokens, heads, headDim, scale) : block.Attend(backend, q, k, v, queryTokens);
            rows[run] = Rows(evidence.Snapshot());
        }

        ReadOnlySpan<float> first = results[0].AsReadOnlySpan<float>(), second = results[1].AsReadOnlySpan<float>(), layer = results[2].AsReadOnlySpan<float>();
        bool formerRepeats = first.SequenceEqual(second);
        bool layerMatches = first.SequenceEqual(layer);
        float formerSpread = MaxAbsDifference(first, second), layerSpread = Math.Max(MaxAbsDifference(first, layer), MaxAbsDifference(second, layer));
        output.WriteLine($"q{queryTokens} k{keyTokens} h{heads} d{headDim}: rows [{rows[2]}]; former repeats bit for bit: {formerRepeats} "
            + $"(largest difference between two former runs {formerSpread:g4}); layer equals former: {layerMatches} (largest difference to either former run {layerSpread:g4})");
        foreach (Tensor t in results)
        {
            t.Dispose();
        }

        CoLa3DTestSupport.Dispose(tensors.Values);
        Assert.Equal(rows[0], rows[2]);
        Assert.Equal(rows[0], rows[1]);
        if (formerRepeats)
        {
            Assert.True(layerMatches, "the layer path differs from the former body, which repeats bit for bit on this device");
        }
        else
        {
            // The kernel of this row is not deterministic on this device (the former body differs from itself), so the layer path may only
            // differ by what two runs of the former body differ by, plus rounding slack.
            Assert.True(layerSpread <= formerSpread * 4 + 1e-6f, $"the layer path differs by {layerSpread:g4}, two former runs differ by {formerSpread:g4}");
        }
    }

    private static float MaxAbsDifference(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        float max = 0f;
        for (int i = 0; i < a.Length; i++)
        {
            max = Math.Max(max, Math.Abs(a[i] - b[i]));
        }

        return max;
    }

    private static string PtxDirectory()
    {
        for (DirectoryInfo? dir = new(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            string candidate = Path.Combine(dir.FullName, "src", "HartsyInference.Cuda", "Ptx");
            if (Directory.Exists(candidate))
            {
                return candidate;
            }
        }

        throw new DirectoryNotFoundException("src/HartsyInference.Cuda/Ptx was not found above the test directory.");
    }
}
