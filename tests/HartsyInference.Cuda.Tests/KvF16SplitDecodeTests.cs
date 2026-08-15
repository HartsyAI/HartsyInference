using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>Split-K attention over an F16 KV cache, at the geometry real decode actually runs: 32 query heads
/// over 8 KV heads at head-dim 128, one query row, cache lengths a song or a chat actually reaches. The
/// pre-existing coverage in <see cref="KvF16StorageTests"/> uses 4 heads at head-dim 64 with a 5e-3 tolerance,
/// which is wide enough to pass whichever kernel runs — so it cannot tell the split path from the monolithic one.
/// These compare the two paths against each other on identical inputs.</summary>
[Collection("CudaSerial")]
public sealed unsafe class KvF16SplitDecodeTests
{
    private readonly ITestOutputHelper _output;
    public KvF16SplitDecodeTests(ITestOutputHelper output) => _output = output;

    private static string PtxDir()
    {
        string ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(ptxDir))
            ptxDir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        return ptxDir;
    }

    private static uint _rng = 0x2545F491u;
    private static float Rand()
    {
        _rng ^= _rng << 13; _rng ^= _rng >> 17; _rng ^= _rng << 5;
        return (_rng & 0xFFFF) / 65535f - 0.5f;
    }

    private static Tensor Rnd(int a, int b, int c, int d)
    {
        Tensor t = new(new TensorShape(a, b, c, d), DType.F32);
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++) p[i] = Rand();
        return t;
    }

    /// <summary>The split kernel and the monolithic kernel must agree over an F16 cache. Both read the same F16
    /// bytes, so the only difference is how the key axis is partitioned and merged — a floating-point
    /// re-association, nothing more. Cache lengths span the split-eligibility threshold upward.</summary>
    [Theory]
    [InlineData(375)]    // 15 s of MiniMax Music 3 at 25 Hz
    [InlineData(1500)]   // a minute of it, and a routine chat context
    public void SplitAndMonolithic_AgreeOverF16Cache_AtDecodeGeometry(int lk)
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        using CudaBackend backend = new(0, PtxDir());
        IBackend b = backend;

        const int hq = 32, hkv = 8, d = 128, group = hq / hkv;
        int qOffset = lk - 1;
        float scale = 1f / MathF.Sqrt(d);

        using Tensor q = Rnd(1, hq, 1, d);
        using Tensor kF32 = Rnd(1, hkv, lk, d);
        using Tensor vF32 = Rnd(1, hkv, lk, d);

        using Tensor kF16 = new(new TensorShape(1, hkv, lk, d), DType.F16);
        using Tensor vF16 = new(new TensorShape(1, hkv, lk, d), DType.F16);
        b.ResidentAllocateKv(kF16);
        b.ResidentAllocateKv(vF16);
        b.KvCacheAppend(kF16, kF32, offset: 0);
        b.KvCacheAppend(vF16, vF32, offset: 0);
        backend.Sync();

        string? prev = Environment.GetEnvironmentVariable("HARTSY_FLASH_SPLIT_OFF");
        using Tensor splitOut = new(new TensorShape(1, hq, 1, d), DType.F32);
        using Tensor monoOut = new(new TensorShape(1, hq, 1, d), DType.F32);
        try
        {
            Environment.SetEnvironmentVariable("HARTSY_FLASH_SPLIT_OFF", null);
            b.FlashAttention(splitOut, q, kF16, vF16, lk, group, causal: true, qOffset, scale);
            backend.Sync();

            Environment.SetEnvironmentVariable("HARTSY_FLASH_SPLIT_OFF", "1");
            b.FlashAttention(monoOut, q, kF16, vF16, lk, group, causal: true, qOffset, scale);
            backend.Sync();
        }
        finally
        {
            Environment.SetEnvironmentVariable("HARTSY_FLASH_SPLIT_OFF", prev);
        }

        float* s = (float*)splitOut.DataPointer;
        float* m = (float*)monoOut.DataPointer;
        float maxDiff = 0f, maxRef = 0f;
        long differing = 0;
        for (long i = 0; i < splitOut.ElementCount; i++)
        {
            maxDiff = MathF.Max(maxDiff, MathF.Abs(s[i] - m[i]));
            maxRef = MathF.Max(maxRef, MathF.Abs(m[i]));
            if (s[i] != m[i]) differing++;
        }
        _output.WriteLine($"lk={lk}: max |split - monolithic| = {maxDiff:E3} (max |monolithic| = {maxRef:E3}), "
            + $"{differing}/{splitOut.ElementCount} elements differ in the last bits");

        // Guards against the failure this test exists for: if split-K silently did not engage, both arms ran the
        // same kernel and every element would match exactly. Re-association across chunks always perturbs some.
        Assert.True(differing > 0,
            "split and monolithic agreed bit-for-bit, so split-K did not engage — this test is not measuring what it claims.");
        Assert.True(maxDiff <= 1e-4f * MathF.Max(maxRef, 1e-3f),
            $"split-K over an F16 cache diverges from the monolithic kernel by {maxDiff:E3} against a {maxRef:E3} peak.");
    }

    /// <summary>Split-K over an F16 cache must land as close to the F32-cache answer as the monolithic F16 kernel
    /// does. Catches a split path that reads the F16 bytes correctly but merges chunks wrongly — which would stay
    /// invisible above if both paths shared the same merge defect.</summary>
    [Fact]
    public void SplitOverF16Cache_TracksF32Cache_AsCloselyAsMonolithic()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        using CudaBackend backend = new(0, PtxDir());
        IBackend b = backend;

        const int hq = 32, hkv = 8, d = 128, lk = 750, group = hq / hkv;
        int qOffset = lk - 1;
        float scale = 1f / MathF.Sqrt(d);

        using Tensor q = Rnd(1, hq, 1, d);
        using Tensor kF32 = Rnd(1, hkv, lk, d);
        using Tensor vF32 = Rnd(1, hkv, lk, d);

        using Tensor refOut = new(new TensorShape(1, hq, 1, d), DType.F32);
        b.FlashAttention(refOut, q, kF32, vF32, lk, group, causal: true, qOffset, scale);
        backend.Sync();

        using Tensor kF16 = new(new TensorShape(1, hkv, lk, d), DType.F16);
        using Tensor vF16 = new(new TensorShape(1, hkv, lk, d), DType.F16);
        b.ResidentAllocateKv(kF16);
        b.ResidentAllocateKv(vF16);
        b.KvCacheAppend(kF16, kF32, offset: 0);
        b.KvCacheAppend(vF16, vF32, offset: 0);
        backend.Sync();

        string? prev = Environment.GetEnvironmentVariable("HARTSY_FLASH_SPLIT_OFF");
        using Tensor splitOut = new(new TensorShape(1, hq, 1, d), DType.F32);
        using Tensor monoOut = new(new TensorShape(1, hq, 1, d), DType.F32);
        try
        {
            Environment.SetEnvironmentVariable("HARTSY_FLASH_SPLIT_OFF", null);
            b.FlashAttention(splitOut, q, kF16, vF16, lk, group, causal: true, qOffset, scale);
            backend.Sync();
            Environment.SetEnvironmentVariable("HARTSY_FLASH_SPLIT_OFF", "1");
            b.FlashAttention(monoOut, q, kF16, vF16, lk, group, causal: true, qOffset, scale);
            backend.Sync();
        }
        finally
        {
            Environment.SetEnvironmentVariable("HARTSY_FLASH_SPLIT_OFF", prev);
        }

        float* r = (float*)refOut.DataPointer;
        float* s = (float*)splitOut.DataPointer;
        float* m = (float*)monoOut.DataPointer;
        float splitErr = 0f, monoErr = 0f;
        for (long i = 0; i < refOut.ElementCount; i++)
        {
            splitErr = MathF.Max(splitErr, MathF.Abs(s[i] - r[i]));
            monoErr = MathF.Max(monoErr, MathF.Abs(m[i] - r[i]));
        }
        _output.WriteLine($"vs F32 cache: split {splitErr:E3}, monolithic {monoErr:E3}");
        Assert.True(splitErr <= monoErr * 1.5f + 1e-6f,
            $"split-K over F16 is {splitErr:E3} from the F32-cache answer while the monolithic F16 kernel is {monoErr:E3} — "
            + "the split path is losing accuracy the storage change alone does not explain.");
    }
}
