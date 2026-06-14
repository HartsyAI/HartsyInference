using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>
/// Phase 1 verification for <see cref="CudaStreamingWeightCache"/>. Tests both the
/// state-management contract (cache hits, empty tokens, cross-cache token rejection)
/// and the end-to-end correctness contract: data uploaded via the streaming path
/// must be byte-identical to the source after <see cref="IStreamingWeightCache.AwaitWeights"/>
/// is satisfied — confirming the cuEventRecord / cuStreamWaitEvent sync actually
/// makes the upload visible to subsequent ops.
/// </summary>
public sealed class CudaStreamingWeightCacheTests
{
    private readonly ITestOutputHelper _output;

    public CudaStreamingWeightCacheTests(ITestOutputHelper output)
    {
        _output = output;
    }

    private static bool CudaAvailable() => CudaContext.IsAvailable();

    /// <summary>Builds a Tensor with deterministic floats for verification.</summary>
    private static unsafe Tensor MakeTensorF32(int count, float seed)
    {
        Tensor t = new Tensor(new TensorShape(count), DType.F32);
        float* p = (float*)t.DataPointer;
        for (int i = 0; i < count; i++)
        {
            p[i] = seed + i * 0.5f;
        }
        return t;
    }

    private static unsafe float[] ReadbackFromDevice(ulong devicePtr, int count)
    {
        float[] result = new float[count];
        fixed (float* p = result)
        {
            CudaMemory.CopyDeviceToHost((void*)p, devicePtr, (nuint)(count * sizeof(float)));
        }
        return result;
    }

    /// <summary>Pulls the cached device pointer for a tensor by triggering a
    /// CopyToDevice — if it's cached this returns the existing dptr without any
    /// new H2D transfer, which is exactly how production code reads the cache.</summary>
    private static ulong GetCachedDevicePointer(Tensor weight)
    {
        return GpuTransferHelper.CopyToDevice(weight);
    }

    // ── BeginUploadAsync + AwaitWeights — the core happy path ───────────

    [Fact]
    public void Upload_Then_Await_Then_Readback_Matches_Source()
    {
        if (!CudaAvailable()) { _output.WriteLine("SKIPPED: no CUDA device"); return; }
        using CudaBackend backend = new CudaBackend();
        IStreamingWeightCache cache = backend.StreamingCache!;

        const int n = 1024;
        using Tensor weight = MakeTensorF32(n, seed: 100f);

        StreamingUploadToken token = cache.BeginUploadAsync(new[] { weight });
        Assert.False(token.IsEmpty, "First upload of a fresh tensor should produce a real token.");
        cache.AwaitWeights(token);

        ulong dptr = GetCachedDevicePointer(weight);
        Assert.NotEqual(0ul, dptr);

        // Force the compute stream to drain so the readback is well-defined.
        backend.Sync();
        float[] roundTrip = ReadbackFromDevice(dptr, n);
        for (int i = 0; i < n; i++)
        {
            Assert.Equal(100f + i * 0.5f, roundTrip[i]);
        }
    }

    [Fact]
    public void Upload_Of_Already_Cached_Returns_Empty_Token()
    {
        if (!CudaAvailable()) { _output.WriteLine("SKIPPED: no CUDA device"); return; }
        using CudaBackend backend = new CudaBackend();
        IStreamingWeightCache cache = backend.StreamingCache!;

        using Tensor weight = MakeTensorF32(64, seed: 1f);

        StreamingUploadToken first = cache.BeginUploadAsync(new[] { weight });
        cache.AwaitWeights(first);

        StreamingUploadToken second = cache.BeginUploadAsync(new[] { weight });
        Assert.True(second.IsEmpty, "Re-uploading an already-cached tensor must be a free hit.");
        cache.AwaitWeights(second); // no-op should not throw
    }

    [Fact]
    public void Upload_Of_Empty_Collection_Returns_Empty_Token()
    {
        if (!CudaAvailable()) { _output.WriteLine("SKIPPED: no CUDA device"); return; }
        using CudaBackend backend = new CudaBackend();
        IStreamingWeightCache cache = backend.StreamingCache!;

        StreamingUploadToken token = cache.BeginUploadAsync(Array.Empty<Tensor>());
        Assert.True(token.IsEmpty);
    }

    [Fact]
    public void Multi_Tensor_Upload_All_Visible_After_Await()
    {
        if (!CudaAvailable()) { _output.WriteLine("SKIPPED: no CUDA device"); return; }
        using CudaBackend backend = new CudaBackend();
        IStreamingWeightCache cache = backend.StreamingCache!;

        Tensor[] weights = new Tensor[8];
        for (int i = 0; i < weights.Length; i++)
        {
            weights[i] = MakeTensorF32(256, seed: 1000f + i * 1000f);
        }

        try
        {
            StreamingUploadToken token = cache.BeginUploadAsync(weights);
            Assert.False(token.IsEmpty);
            cache.AwaitWeights(token);

            backend.Sync();
            for (int i = 0; i < weights.Length; i++)
            {
                ulong dptr = GetCachedDevicePointer(weights[i]);
                float[] data = ReadbackFromDevice(dptr, 256);
                for (int j = 0; j < 256; j++)
                {
                    Assert.Equal(1000f + i * 1000f + j * 0.5f, data[j]);
                }
            }
        }
        finally
        {
            foreach (Tensor t in weights) t.Dispose();
        }
    }

    // ── EvictAsync ───────────────────────────────────────────────────────

    [Fact]
    public void Evict_Removes_From_Cache_And_Reupload_Reflects_New_Data()
    {
        if (!CudaAvailable()) { _output.WriteLine("SKIPPED: no CUDA device"); return; }
        using CudaBackend backend = new CudaBackend();
        IStreamingWeightCache cache = backend.StreamingCache!;

        unsafe
        {
            using Tensor weight = MakeTensorF32(128, seed: 50f);

            // Upload v1
            StreamingUploadToken t1 = cache.BeginUploadAsync(new[] { weight });
            cache.AwaitWeights(t1);

            // Mutate CPU-side data, then evict — eviction makes the next upload re-fetch from CPU.
            float* p = (float*)weight.DataPointer;
            for (int i = 0; i < 128; i++) p[i] = 999f - i;

            cache.EvictAsync(new[] { weight });
            backend.Sync(); // drain the async free so the next BeginUpload gets a real upload token

            // Re-upload — the dptr will be a fresh allocation containing the new data.
            StreamingUploadToken t2 = cache.BeginUploadAsync(new[] { weight });
            Assert.False(t2.IsEmpty, "After eviction, re-upload must allocate fresh memory and produce a real token.");
            cache.AwaitWeights(t2);

            backend.Sync();
            ulong dptr = GetCachedDevicePointer(weight);
            float[] data = ReadbackFromDevice(dptr, 128);
            for (int i = 0; i < 128; i++)
            {
                Assert.Equal(999f - i, data[i]);
            }
        }
    }

    [Fact]
    public void Evict_Of_Uncached_Tensor_Is_Silent_Noop()
    {
        if (!CudaAvailable()) { _output.WriteLine("SKIPPED: no CUDA device"); return; }
        using CudaBackend backend = new CudaBackend();
        IStreamingWeightCache cache = backend.StreamingCache!;

        using Tensor weight = MakeTensorF32(16, seed: 7f);
        // Never uploaded — should not throw.
        cache.EvictAsync(new[] { weight });
    }

    // ── Token validation ────────────────────────────────────────────────

    [Fact]
    public void Await_With_Token_From_Different_Cache_Throws()
    {
        if (!CudaAvailable()) { _output.WriteLine("SKIPPED: no CUDA device"); return; }
        using CudaBackend backend = new CudaBackend();
        // Two cache instances on the same backend — same streams, same context, but
        // different identity. The BackendTag check on the token should reject a token
        // issued by one cache and presented to the other.
        IStreamingWeightCache cacheA = backend.StreamingCache!;
        IStreamingWeightCache cacheB = new CudaStreamingWeightCache(
            backend.Context, backend.Stream.Handle, backend.UploadStream.Handle);

        using Tensor weight = MakeTensorF32(32, seed: 1f);
        StreamingUploadToken tokenFromA = cacheA.BeginUploadAsync(new[] { weight });
        try
        {
            Assert.Throws<InvalidOperationException>(() => cacheB.AwaitWeights(tokenFromA));
        }
        finally
        {
            // Don't leak the event — A still owns it.
            cacheA.AwaitWeights(tokenFromA);
        }
    }

    [Fact]
    public void Await_Of_Empty_Token_Is_Noop()
    {
        if (!CudaAvailable()) { _output.WriteLine("SKIPPED: no CUDA device"); return; }
        using CudaBackend backend = new CudaBackend();
        IStreamingWeightCache cache = backend.StreamingCache!;

        cache.AwaitWeights(StreamingUploadToken.Empty); // does nothing, must not throw
    }

    // ── Budget query ────────────────────────────────────────────────────

    [Fact]
    public void QueryAvailableWeightCacheBytes_Returns_Plausible_Number()
    {
        if (!CudaAvailable()) { _output.WriteLine("SKIPPED: no CUDA device"); return; }
        using CudaBackend backend = new CudaBackend();
        IStreamingWeightCache cache = backend.StreamingCache!;

        long withZeroReserve = cache.QueryAvailableWeightCacheBytes(0);
        Assert.True(withZeroReserve > 0, $"Expected positive free VRAM, got {withZeroReserve}");

        // Asking for a 100 MB reserve should drop the available figure by at least 100 MB.
        long withReserve = cache.QueryAvailableWeightCacheBytes(100L * 1024 * 1024);
        Assert.True(withReserve <= withZeroReserve - 100L * 1024 * 1024 + 1024,
            $"Reserved query ({withReserve}) should be at least 100 MB lower than unreserved ({withZeroReserve}).");

        _output.WriteLine($"Free VRAM available for weight cache: {withZeroReserve / (1024.0 * 1024.0):F1} MB");
    }

    [Fact]
    public void QueryAvailableWeightCacheBytes_Negative_Reserve_Throws()
    {
        if (!CudaAvailable()) { _output.WriteLine("SKIPPED: no CUDA device"); return; }
        using CudaBackend backend = new CudaBackend();
        IStreamingWeightCache cache = backend.StreamingCache!;
        Assert.Throws<ArgumentOutOfRangeException>(() => cache.QueryAvailableWeightCacheBytes(-1));
    }

    // ── Round-trip: upload, op reads cached weight, evict, re-upload ───

    [Fact]
    public void Upload_Then_Op_Reads_Cached_Weight_Without_Reuploading()
    {
        if (!CudaAvailable()) { _output.WriteLine("SKIPPED: no CUDA device"); return; }
        using CudaBackend backend = new CudaBackend();
        IStreamingWeightCache cache = backend.StreamingCache!;

        using Tensor weight = MakeTensorF32(512, seed: 42f);

        StreamingUploadToken token = cache.BeginUploadAsync(new[] { weight });
        cache.AwaitWeights(token);
        ulong dptrFirst = GetCachedDevicePointer(weight);

        // A second CopyToDevice should be a free cache hit and return the *same* dptr —
        // i.e. no re-allocation happened, the streaming-uploaded weight is recognized
        // by the standard backend op fast path.
        ulong dptrSecond = GetCachedDevicePointer(weight);
        Assert.Equal(dptrFirst, dptrSecond);
    }
}
