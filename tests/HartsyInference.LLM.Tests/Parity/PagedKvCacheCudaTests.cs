using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.LLM.Tests.Parity;

/// <summary>On the device the paged cache's contiguous prefix is kept between calls, so it must survive what a KV buffer
/// survives: activations freed or offloaded between steps. It is compared with a contiguous cache fed the same rows and
/// with the rows themselves.</summary>
[Trait("Category", "GpuIntegration")]
[Collection("AttentionSemantics")]
public sealed class PagedKvCacheCudaTests
{
    private const int Layers = 2, KvHeads = 2, HeadDim = 64, PageSize = 16, MaxPages = 32;

    [CudaTheory]
    [InlineData("none", 0)]
    [InlineData("offload", 0)]
    [InlineData("free", 0)]
    [InlineData("none", 101)]
    [InlineData("offload", 101)]
    [InlineData("free", 101)]
    public void ThePrefixMatchesAContiguousCacheAcrossActivationRelease(string release, int expectedTokens)
    {
        using CudaBackend backend = new(0, CudaPtxDirectory.Resolve());
        PrefixMatchesAContiguousCache(backend, release, expectedTokens);
    }

    /// <summary>The prefix check on any device backend; <see cref="PagedKvCacheVulkanTests"/> runs it on Vulkan.</summary>
    internal static void PrefixMatchesAContiguousCache(IBackend backend, string release, int expectedTokens)
    {
        using PagedKvPool pool = new(Layers, KvHeads, HeadDim, PageSize, MaxPages);
        using PagedKvCache paged = new(pool, expectedTokens);
        using FixedKvCache contiguous = new(Layers, 1, KvHeads, HeadDim, PageSize * MaxPages, DType.F32);
        Random random = new(99);
        List<float[]>[] keys = [[], []], values = [[], []];

        void Step(int tokens)
        {
            for (int layer = 0; layer < Layers; layer++)
            {
                using Tensor k = Rows(random, tokens), v = Rows(random, tokens);
                Remember(keys[layer], k, tokens);
                Remember(values[layer], v, tokens);
                paged.AppendStep(backend, layer, k, v);
                contiguous.AppendStep(backend, layer, k, v);
                int valid = paged.CurrentLength + tokens;
                AssertRows(paged.KeyPrefix(layer), contiguous.KeyPrefix(layer), keys[layer], valid, $"layer {layer} keys");
                AssertRows(paged.ValuePrefix(layer), contiguous.ValuePrefix(layer), values[layer], valid, $"layer {layer} values");
            }
            paged.AdvanceLength(tokens);
            contiguous.AdvanceLength(tokens);
            if (release == "offload") backend.OffloadActivations(long.MaxValue);
            else if (release == "free") backend.FreeActivations(trimPool: false);
        }

        Step(40);
        for (int step = 0; step < 40; step++) Step(1);
        paged.Truncate(50);
        contiguous.Truncate(50);
        foreach (List<float[]> rows in keys) rows.RemoveRange(50, rows.Count - 50);
        foreach (List<float[]> rows in values) rows.RemoveRange(50, rows.Count - 50);
        for (int step = 0; step < 20; step++) Step(1);
    }

    /// <summary>Two sequences grow in one pool so their pages interleave, one rolls back and regrows on pages the other
    /// released, and every committed row is then read straight from the pool through each page table.</summary>
    /// <remarks>Offloading keeps the rows (they move to the host and reload). Freeing activations is not a case here: it
    /// drops a device-only buffer's contents, for this pool and for a <see cref="FixedKvCache"/> alike, and the case above
    /// survives it only because it reads each buffer on the host every step. No text path frees activations mid-generation.</remarks>
    [CudaTheory]
    [InlineData("none")]
    [InlineData("offload")]
    public void ThePagesOfTwoSequencesHoldTheirOwnRowsAcrossActivationRelease(string release)
    {
        using CudaBackend backend = new(0, CudaPtxDirectory.Resolve());
        PagesOfTwoSequencesHoldTheirOwnRows(backend, release);
    }

    /// <summary>The page read-back check on any device backend; <see cref="PagedKvCacheVulkanTests"/> runs it on Vulkan.</summary>
    internal static void PagesOfTwoSequencesHoldTheirOwnRows(IBackend backend, string release)
    {
        using PagedKvPool pool = new(Layers, KvHeads, HeadDim, PageSize, MaxPages);
        using PagedKvCache a = new(pool), b = new(pool, 101);
        Random random = new(7);
        List<float[]>[] aKeys = [[], []], aValues = [[], []], bKeys = [[], []], bValues = [[], []];

        void Step(PagedKvCache cache, List<float[]>[] keys, List<float[]>[] values, int tokens)
        {
            for (int layer = 0; layer < Layers; layer++)
            {
                using Tensor k = Rows(random, tokens), v = Rows(random, tokens);
                Remember(keys[layer], k, tokens);
                Remember(values[layer], v, tokens);
                cache.AppendStep(backend, layer, k, v);
            }
            cache.AdvanceLength(tokens);
            if (release == "offload") backend.OffloadActivations(long.MaxValue);
        }

        void Cut(PagedKvCache cache, List<float[]>[] keys, List<float[]>[] values, int length)
        {
            cache.Truncate(length);
            foreach (List<float[]> rows in keys) rows.RemoveRange(length, rows.Count - length);
            foreach (List<float[]> rows in values) rows.RemoveRange(length, rows.Count - length);
        }

        Step(a, aKeys, aValues, 40);
        Step(b, bKeys, bValues, 23);
        for (int step = 0; step < 30; step++)
        {
            Step(a, aKeys, aValues, 1);
            Step(b, bKeys, bValues, 1);
        }
        Cut(a, aKeys, aValues, 20);
        Step(b, bKeys, bValues, 37);
        Step(a, aKeys, aValues, 45);
        for (int step = 0; step < 5; step++)
        {
            Step(a, aKeys, aValues, 1);
            Step(b, bKeys, bValues, 1);
        }

        Assert.True(Scattered(a.PageTable) && Scattered(b.PageTable), "The sequences' pages were meant to interleave.");
        Assert.Empty(a.PageTable.Intersect(b.PageTable));
        for (int layer = 0; layer < Layers; layer++)
        {
            AssertPages(pool, a, pool.KeyPages(layer), aKeys[layer], $"sequence A, layer {layer} keys");
            AssertPages(pool, a, pool.ValuePages(layer), aValues[layer], $"sequence A, layer {layer} values");
            AssertPages(pool, b, pool.KeyPages(layer), bKeys[layer], $"sequence B, layer {layer} keys");
            AssertPages(pool, b, pool.ValuePages(layer), bValues[layer], $"sequence B, layer {layer} values");
        }
    }

    private static bool Scattered(IReadOnlyList<int> table)
    {
        for (int i = 0; i < table.Count; i++)
            if (table[i] != i) return true;
        return false;
    }

    private static void AssertPages(PagedKvPool pool, PagedKvCache cache, Tensor pages, List<float[]> expected, string what)
    {
        ReadOnlySpan<float> actual = pages.AsReadOnlySpan<float>();
        int poolRows = (int)pages.Shape[2];
        IReadOnlyList<int> table = cache.PageTable;
        Assert.Equal(cache.CurrentLength, expected.Count);
        for (int head = 0; head < KvHeads; head++)
            for (int row = 0; row < expected.Count; row++)
            {
                int poolRow = pool.FirstRow(table[row / PageSize]) + row % PageSize;
                Assert.True(actual.Slice((head * poolRows + poolRow) * HeadDim, HeadDim).SequenceEqual(expected[row].AsSpan(head * HeadDim, HeadDim)),
                    $"{what}: head {head}, row {row} (pool row {poolRow}) differs.");
            }
    }

    private static Tensor Rows(Random random, int tokens)
    {
        Tensor tensor = new(new TensorShape(1, KvHeads, tokens, HeadDim), DType.F32);
        Span<float> data = tensor.AsSpan<float>();
        for (int i = 0; i < data.Length; i++) data[i] = (float)random.NextDouble();
        return tensor;
    }

    /// <summary>Keeps a host copy of each appended row as <c>[head][headDim]</c> flattened head-major.</summary>
    private static void Remember(List<float[]> rows, Tensor appended, int tokens)
    {
        ReadOnlySpan<float> data = appended.AsReadOnlySpan<float>();
        for (int token = 0; token < tokens; token++)
        {
            float[] row = new float[KvHeads * HeadDim];
            for (int head = 0; head < KvHeads; head++)
                data.Slice((head * tokens + token) * HeadDim, HeadDim).CopyTo(row.AsSpan(head * HeadDim, HeadDim));
            rows.Add(row);
        }
    }

    private static void AssertRows(Tensor paged, Tensor contiguous, List<float[]> expected, int validRows, string what)
    {
        ReadOnlySpan<float> actual = paged.AsReadOnlySpan<float>(), reference = contiguous.AsReadOnlySpan<float>();
        int pagedRows = (int)paged.Shape[2], contiguousRows = (int)contiguous.Shape[2];
        for (int head = 0; head < KvHeads; head++)
            for (int row = 0; row < validRows; row++)
            {
                ReadOnlySpan<float> want = expected[row].AsSpan(head * HeadDim, HeadDim);
                Assert.True(reference.Slice((head * contiguousRows + row) * HeadDim, HeadDim).SequenceEqual(want),
                    $"{what}: the contiguous cache lost head {head}, row {row}.");
                Assert.True(actual.Slice((head * pagedRows + row) * HeadDim, HeadDim).SequenceEqual(want),
                    $"{what}: the paged prefix differs at head {head}, row {row} of {validRows}.");
            }
    }
}
