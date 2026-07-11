using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.LLM.Transformer;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>Correctness + robustness gates for the Phase 3 paged KV cache
/// (<see cref="PagedKvPool"/>/<see cref="PagedKvCache"/>): (1) byte-for-byte parity against
/// <see cref="FixedKvCache"/> across a prefill (multi-page-spanning append) + several decode steps
/// (page-boundary-crossing single-token appends), and (2) a synthetic multi-admission/eviction stress
/// harness exercising fragmentation, page reuse, and the exhaustion (reject) policy — per the production
/// plan's explicit ask not to defer dynamic-load validation to the batching phase.</summary>
[Collection("CudaSerial")]
public sealed class PagedKvCacheTests
{
    private readonly ITestOutputHelper _output;
    public PagedKvCacheTests(ITestOutputHelper output) => _output = output;

    private const int NumLayers = 2, NumKvHeads = 2, HeadDim = 8, PageSize = 4;

    [Fact]
    public unsafe void PagedKvCache_MatchesFixedKvCache_AcrossPrefillAndDecode()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        string ptxDir = PtxDir();
        using CudaBackend backend = new(0, ptxDir);

        using FixedKvCache fixedCache = new(NumLayers, 1, NumKvHeads, HeadDim, maxSequenceLength: 32);
        using PagedKvPool pool = new(NumLayers, NumKvHeads, HeadDim, PageSize, maxPages: 16);
        using PagedKvCache pagedCache = new(pool);

        Random rng = new(42);
        int len = 0;

        // Prefill: 10 tokens at pageSize=4 spans pages [0,1,2] (4+4+2) — exercises the multi-page-spanning
        // append path (SliceTimeRange chunking) that a single decode step (tNew=1) never touches.
        len += AppendBothAndAdvance(backend, fixedCache, pagedCache, tNew: 10, rng);
        AssertParity(fixedCache, pagedCache, len);

        // Decode steps: single-token appends, several of which cross a page boundary (len 12 -> new page 3,
        // len 16 -> new page 4).
        for (int step = 0; step < 8; step++)
        {
            len += AppendBothAndAdvance(backend, fixedCache, pagedCache, tNew: 1, rng);
            AssertParity(fixedCache, pagedCache, len);
        }
        _output.WriteLine($"PASS: parity held through {len} tokens across {pagedCache.PagesHeld} pages.");
    }

    [Fact]
    public void PagedKvPool_SurvivesFragmentedMultiSequenceAdmissionAndEviction()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        string ptxDir = PtxDir();
        using CudaBackend backend = new(0, ptxDir);

        // Deliberately small budget relative to the workload so the exhaustion (reject) path gets exercised.
        const int maxPages = 8;
        using PagedKvPool pool = new(NumLayers, NumKvHeads, HeadDim, PageSize, maxPages);

        Random rng = new(7);
        List<PagedKvCache> active = [];
        int admissions = 0, evictions = 0, exhaustionHits = 0;

        for (int round = 0; round < 300; round++)
        {
            int action = active.Count == 0 ? 0 : rng.Next(3);
            switch (action)
            {
                case 0: // admit a new sequence
                {
                    PagedKvCache cache = new(pool);
                    try
                    {
                        int tNew = rng.Next(1, 6);
                        AppendRandom(backend, cache, tNew, rng);
                        cache.AdvanceLength(tNew);
                        active.Add(cache);
                        admissions++;
                    }
                    catch (KvPoolExhaustedException)
                    {
                        cache.Dispose(); // return whatever partial pages it grabbed before hitting the wall
                        exhaustionHits++;
                    }
                    break;
                }
                case 1: // grow an existing sequence by one token
                {
                    int idx = rng.Next(active.Count);
                    try
                    {
                        AppendRandom(backend, active[idx], 1, rng);
                        active[idx].AdvanceLength(1);
                    }
                    catch (KvPoolExhaustedException) { exhaustionHits++; }
                    break;
                }
                default: // evict a random active sequence
                {
                    int idx = rng.Next(active.Count);
                    active[idx].Dispose();
                    active.RemoveAt(idx);
                    evictions++;
                    break;
                }
            }
        }

        foreach (PagedKvCache cache in active) cache.Dispose();

        _output.WriteLine($"admissions={admissions} evictions={evictions + active.Count} exhaustionHits={exhaustionHits} freeAtEnd={pool.FreePageCount}/{maxPages}");
        Assert.True(exhaustionHits > 0, "budget was sized to guarantee at least one exhaustion hit — if this never fires the test isn't exercising the reject path");
        // The load-bearing invariant: every page allocated across 300 rounds of random admit/grow/evict came
        // back. A leak here would mean the pool silently shrinks over a long-running server's lifetime.
        Assert.Equal(maxPages, pool.FreePageCount);
    }

    private static unsafe int AppendBothAndAdvance(CudaBackend backend, FixedKvCache fixedCache, PagedKvCache pagedCache, int tNew, Random rng)
    {
        for (int layer = 0; layer < NumLayers; layer++)
        {
            using Tensor k = RandomKv(tNew, rng);
            using Tensor v = RandomKv(tNew, rng);
            fixedCache.AppendStep(backend, layer, k, v);
            pagedCache.AppendStep(backend, layer, k, v);
        }
        fixedCache.AdvanceLength(tNew);
        pagedCache.AdvanceLength(tNew);
        return tNew;
    }

    private static unsafe void AppendRandom(CudaBackend backend, PagedKvCache cache, int tNew, Random rng)
    {
        for (int layer = 0; layer < NumLayers; layer++)
        {
            using Tensor k = RandomKv(tNew, rng);
            using Tensor v = RandomKv(tNew, rng);
            cache.AppendStep(backend, layer, k, v);
        }
    }

    private static unsafe Tensor RandomKv(int tNew, Random rng)
    {
        Tensor t = new(new TensorShape(1, NumKvHeads, tNew, HeadDim), DType.F32);
        float* p = (float*)t.DataPointer;
        long count = (long)NumKvHeads * tNew * HeadDim;
        for (long i = 0; i < count; i++) p[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return t;
    }

    private static unsafe void AssertParity(FixedKvCache fixedCache, PagedKvCache pagedCache, int len)
    {
        for (int layer = 0; layer < NumLayers; layer++)
        {
            ComparePrefix(fixedCache.KeyPrefix(layer), pagedCache.KeyPrefix(layer), len);
            ComparePrefix(fixedCache.ValuePrefix(layer), pagedCache.ValuePrefix(layer), len);
        }
    }

    private static unsafe void ComparePrefix(Tensor fixedBuf, Tensor pagedBuf, int len)
    {
        int fixedStride = (int)fixedBuf.Shape[2];
        int pagedStride = (int)pagedBuf.Shape[2];
        Assert.Equal(len, pagedStride); // paged gather returns EXACTLY the written length, no over-allocation
        float* fp = (float*)fixedBuf.DataPointer;
        float* pp = (float*)pagedBuf.DataPointer;
        for (int h = 0; h < NumKvHeads; h++)
        {
            for (int t = 0; t < len; t++)
            {
                for (int d = 0; d < HeadDim; d++)
                {
                    float fv = fp[((long)h * fixedStride + t) * HeadDim + d];
                    float pv = pp[((long)h * pagedStride + t) * HeadDim + d];
                    Assert.True(fv == pv, $"mismatch at h={h} t={t} d={d}: fixed={fv} paged={pv}");
                }
            }
        }
    }

    private static string PtxDir()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        return Directory.Exists(dir) ? dir : Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
    }
}
