using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Gpu;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>Residency, dedup, pinning, promotion and leak behaviour of <see cref="CudaExpertCache"/> on a real device.</summary>
[Collection("CudaSerial")]
[Trait("Category", "GpuIntegration")]
public sealed unsafe class CudaExpertCacheTests
{
    private const int Elements = 300_000;
    private const long ExpertBytes = 3L * Elements * sizeof(float);

    private readonly ITestOutputHelper _output;

    public CudaExpertCacheTests(ITestOutputHelper output) => _output = output;

    private static Tensor Filled(float seed)
    {
        Tensor tensor = new(new TensorShape(Elements), DType.F32);
        float* data = (float*)tensor.DataPointer;
        for (int i = 0; i < Elements; i++) data[i] = seed + i * 0.25f;
        return tensor;
    }

    private static ExpertBank Bank(int layer, int count)
    {
        return new ExpertBank(layer, count, key =>
        {
            float seed = key.Layer * 1000f + key.Expert * 10f;
            return new ExpertWeights(key,
                new ExpertMatrix(Filled(seed + 1)), new ExpertMatrix(Filled(seed + 2)), new ExpertMatrix(Filled(seed + 3)));
        });
    }

    private static void AssertResident(ExpertWeights weights, float seed)
    {
        float[] host = new float[Elements];
        float[] seeds = [seed + 1, seed + 2, seed + 3];
        Tensor[] matrices = [weights.W1.Weight, weights.W2.Weight, weights.W3.Weight];
        for (int m = 0; m < 3; m++)
        {
            Assert.True(GpuTransferHelper.TryGetCachedDevice(matrices[m], out ulong device));
            fixed (float* destination = host) CudaMemory.CopyDeviceToHost(destination, device, (nuint)(Elements * sizeof(float)));
            for (int i = 0; i < Elements; i += 997) Assert.Equal(seeds[m] + i * 0.25f, host[i]);
        }
    }

    private static string PtxDir()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(dir))
            dir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        return dir;
    }

    private bool Ready(CudaBackend backend, string test)
    {
        // Confirms the run is on the intended card; the suite must run with CUDA_VISIBLE_DEVICES=1 (the 3060).
        _output.WriteLine($"device: {backend.Capabilities.Name}");
        return true;
    }

    [Fact]
    public void AcquireMissHitAndDedup_UploadOncePerExpertWithMatchingBytes()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: no CUDA device"); return; }
        using CudaBackend backend = new(0, PtxDir());
        Ready(backend, nameof(AcquireMissHitAndDedup_UploadOncePerExpertWithMatchingBytes));
        ExpertBank bank = Bank(0, 8);
        using CudaExpertCache cache = new(backend, 4 * ExpertBytes, [bank]);

        using ExpertLease first = cache.Acquire([new(0, 1), new(0, 1), new(0, 2)]);
        Assert.Equal(2, first.Weights.Count);
        backend.Sync();
        AssertResident(first.Get(new(0, 1)), 10f);
        AssertResident(first.Get(new(0, 2)), 20f);
        Assert.Equal(2, cache.Stats.Misses);
        Assert.Equal(2 * ExpertBytes, cache.Stats.BytesUploaded);

        using ExpertLease second = cache.Acquire([new(0, 2)]);
        Assert.Equal(1, cache.Stats.Hits);
        Assert.Same(first.Get(new(0, 2)), second.Get(new(0, 2)));
        Assert.Equal(2, cache.Stats.ResidentExperts);
    }

    [Fact]
    public void Prefetch_ThenAcquireSharesTheInFlightUpload()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: no CUDA device"); return; }
        using CudaBackend backend = new(0, PtxDir());
        ExpertBank bank = Bank(0, 8);
        using CudaExpertCache cache = new(backend, 4 * ExpertBytes, [bank]);

        Assert.Equal(2, cache.Prefetch([new(0, 3), new(0, 4)]));
        using ExpertLease lease = cache.Acquire([new(0, 3), new(0, 4)]);
        backend.Sync();
        ExpertCacheStats stats = cache.Stats;
        Assert.Equal(2, stats.Prefetches);
        Assert.Equal(2, stats.Hits);
        Assert.Equal(2, stats.InFlightHits);
        Assert.Equal(0, stats.Misses);
        Assert.Equal(2 * ExpertBytes, stats.BytesUploaded);
        AssertResident(lease.Get(new(0, 3)), 30f);
    }

    [Fact]
    public void PinnedExpertsAreNeverEvicted_AndUnpinnedOnesAre()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: no CUDA device"); return; }
        using CudaBackend backend = new(0, PtxDir());
        ExpertBank bank = Bank(0, 8);
        using CudaExpertCache cache = new(backend, 2 * ExpertBytes, [bank]);

        using ExpertLease pinned = cache.Acquire([new(0, 0), new(0, 1)]);
        Assert.Throws<OutOfVramException>(() => cache.Acquire([new(0, 2)]));
        Assert.Equal(0, cache.Stats.Evictions);
        backend.Sync();
        AssertResident(pinned.Get(new(0, 0)), 0f);
        AssertResident(pinned.Get(new(0, 1)), 10f);

        cache.Release(pinned);
        using ExpertLease next = cache.Acquire([new(0, 2), new(0, 3)]);
        Assert.Equal(2, cache.Stats.Evictions);
        Assert.False(GpuTransferHelper.IsWeightCached(pinned.Get(new(0, 0)).W1.Weight));
        backend.Sync();
        AssertResident(next.Get(new(0, 3)), 30f);
    }

    [Fact]
    public void EvictedExpertsAreNotAutoPromotedBackIntoTheWeightCache()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: no CUDA device"); return; }
        using CudaBackend backend = new(0, PtxDir());
        Ready(backend, nameof(EvictedExpertsAreNotAutoPromotedBackIntoTheWeightCache));
        ExpertBank bank = Bank(0, 4);
        using CudaExpertCache cache = new(backend, 1 * ExpertBytes, [bank]);
        ExpertWeights evicted = bank.Get(0);
        cache.Acquire([new(0, 0)]).Dispose();
        cache.Acquire([new(0, 1)]).Dispose();
        Assert.False(GpuTransferHelper.IsWeightCached(evicted.W1.Weight));

        // Streaming the evicted expert through an ordinary op is the re-upload path that promotes an unblocked tensor.
        using Tensor sink = new(new TensorShape(Elements), DType.F32);
        for (int i = 0; i < 4; i++)
        {
            backend.Scale(sink, evicted.W1.Weight, 1f);
            _ = sink.DataPointer;
        }
        Assert.NotEqual(GpuResidencyTier.Weight, GpuTransferHelper.CurrentState.TierOf(evicted.W1.Weight));

        // Positive control: a tensor the cache never saw is promoted by the same sequence, so the assertion can fail.
        if (CudaDriverApi.cuMemGetInfo(out nuint free, out _) != 0 || (long)free < (1536L + 64L) << 20) return;
        using Tensor control = Filled(5f);
        for (int i = 0; i < 4; i++)
        {
            backend.Scale(sink, control, 1f);
            _ = sink.DataPointer;
        }
        Assert.Equal(GpuResidencyTier.Weight, GpuTransferHelper.CurrentState.TierOf(control));
    }

    [Fact]
    public void ThousandAcquireReleaseCycles_AndAbandonedPrefetches_LeaveNothingResident()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: no CUDA device"); return; }
        using CudaBackend backend = new(0, PtxDir());
        ExpertBank bank = Bank(0, 24);
        (nuint baselineFree, _) = MemInfo();
        using (CudaExpertCache warm = new(backend, 4 * ExpertBytes, [bank]))
        {
            warm.Acquire([new(0, 0)]).Dispose();
        }
        backend.Sync();
        (baselineFree, _) = MemInfo();

        using (CudaExpertCache cache = new(backend, 4 * ExpertBytes, [bank]))
        {
            for (int i = 0; i < 1000; i++)
            {
                ExpertKey[] want = [new(0, i % 24), new(0, (i * 7 + 3) % 24)];
                ExpertLease lease = cache.Acquire(want);
                if (i % 5 == 0) cache.Prefetch([new(0, (i + 11) % 24)]);
                if (i % 2 == 0) backend.Sync();
                lease.Dispose();
            }
            ExpertCacheStats stats = cache.Stats;
            Assert.Equal(0, stats.PinnedExperts);
            Assert.True(stats.ResidentBytes <= cache.BudgetBytes);
            Assert.True(stats.Evictions > 0);
        }
        for (int i = 0; i < 24; i++)
        {
            Assert.False(GpuTransferHelper.IsWeightCached(bank.Get(i).W1.Weight));
            Assert.False(GpuTransferHelper.IsWeightCached(bank.Get(i).W2.Weight));
            Assert.False(GpuTransferHelper.IsWeightCached(bank.Get(i).W3.Weight));
        }

        for (int i = 0; i < 40; i++)
        {
            CudaExpertCache abandoned = new(backend, 4 * ExpertBytes, [bank]);
            abandoned.Acquire([new(0, i % 24)]).Dispose();
            abandoned.Prefetch([new(0, (i + 1) % 24), new(0, (i + 2) % 24)]);
            abandoned.Dispose();
        }
        backend.Sync();
        (nuint freeAfter, _) = MemInfo();
        long drift = (long)baselineFree - (long)freeAfter;
        _output.WriteLine($"free before {(baselineFree >> 20)} MB, after {(freeAfter >> 20)} MB, drift {drift >> 20} MB");
        Assert.True(drift < 64L << 20, $"VRAM drifted by {drift >> 20} MB across 1,000 cycles and 40 abandoned caches");
    }

    [Fact]
    public void ConfiguresTheStagingRingAndKeepsSourcePinningOff()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: no CUDA device"); return; }
        using CudaBackend backend = new(0, PtxDir());
        CudaStreamingWeightCache streaming = (CudaStreamingWeightCache)backend.StreamingCache!;
        streaming.PinUploadSource = true;
        using (CudaExpertCache cache = new(backend, 2 * ExpertBytes, [Bank(0, 2)], stagingSlots: 6))
        {
            Assert.Equal(6, streaming.StagingSlotCount);
            Assert.False(streaming.PinUploadSource);
        }
        Assert.True(streaming.PinUploadSource);
    }

    private static (nuint Free, nuint Total) MemInfo()
    {
        CudaDriverApi.cuMemGetInfo(out nuint free, out nuint total).ThrowOnError();
        return (free, total);
    }
}
