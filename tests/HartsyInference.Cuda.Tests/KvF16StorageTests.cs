using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>F16-storage KV cache (ROADMAP.md §1 / task #15): the K/V projection stays F32, the resident cache
/// buffer is F16 (halves VRAM), and FlashAttention upconverts back to F32 on read — a storage/bandwidth change,
/// not a numerically new kernel (Q, scores, softmax, and the accumulator all stay F32 throughout). Verifies the
/// two new CUDA kernels (<c>lm_kv_append_f16</c>, <c>lm_flash_attn_f16kv_f32</c>) in isolation before trusting
/// the real-weight Llama-3.2-1B token comparison.</summary>
[Collection("CudaSerial")]
public sealed unsafe class KvF16StorageTests
{
    private readonly ITestOutputHelper _output;
    public KvF16StorageTests(ITestOutputHelper output) => _output = output;

    private static string PtxDir()
    {
        string ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(ptxDir))
            ptxDir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        return ptxDir;
    }

    private static uint _rng = 0x9E3Du;
    private static float Rand()
    {
        _rng ^= _rng << 13; _rng ^= _rng >> 17; _rng ^= _rng << 5;
        return (_rng & 0xFFFF) / 65535f - 0.5f;
    }

    private static Tensor Rnd(int a, int b, int c, int d, float scale = 1f)
    {
        Tensor t = new(new TensorShape(a, b, c, d), DType.F32);
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++) p[i] = Rand() * scale;
        return t;
    }

    /// <summary>KvCacheAppend into an F16-typed buffer round-trips through F16 rounding (not bit-identical to
    /// the F32 source — that's expected and the whole point) and lands at the correct offset/addressing, same
    /// as the F32 kernel's layout.</summary>
    [Fact]
    public void KvCacheAppend_F16Dest_RoundTripsWithinF16Rounding()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        using CudaBackend backend = new(0, PtxDir());
        IBackend b = backend;

        const int heads = 3, maxSeq = 5, headDim = 8;
        using Tensor buffer = new(new TensorShape(1, heads, maxSeq, headDim), DType.F16);
        b.ResidentAllocateKv(buffer);

        // First append at offset 0 (tNew=2), second at offset 2 (tNew=1) — exercises the same offset addressing
        // as the F32 kernel, just with an F16 destination.
        using Tensor step1 = Rnd(1, heads, 2, headDim, scale: 4f);
        using Tensor step2 = Rnd(1, heads, 1, headDim, scale: 4f);
        b.KvCacheAppend(buffer, step1, offset: 0);
        b.KvCacheAppend(buffer, step2, offset: 2);
        backend.Sync();

        Half* bufP = (Half*)buffer.DataPointer;
        float* s1 = (float*)step1.DataPointer;
        float* s2 = (float*)step2.DataPointer;
        float maxDiff = 0f;
        for (int h = 0; h < heads; h++)
        {
            for (int t = 0; t < 2; t++)
            {
                for (int d = 0; d < headDim; d++)
                {
                    float expected = (float)(Half)s1[(h * 2 + t) * headDim + d];   // reference F16 rounding
                    float actual = (float)bufP[(h * (long)maxSeq + t) * headDim + d];
                    maxDiff = MathF.Max(maxDiff, MathF.Abs(expected - actual));
                    Assert.Equal(expected, actual);   // F16 rounding must match .NET's Half exactly — same IEEE-754 binary16
                }
            }
            for (int d = 0; d < headDim; d++)
            {
                float expected = (float)(Half)s2[h * headDim + d];
                float actual = (float)bufP[(h * (long)maxSeq + 2) * headDim + d];
                maxDiff = MathF.Max(maxDiff, MathF.Abs(expected - actual));
                Assert.Equal(expected, actual);
            }
        }
        _output.WriteLine($"max F16-rounding diff vs .NET Half reference: {maxDiff:E3} (expect 0 — exact round-trip)");
    }

    /// <summary>FlashAttention with F16-storage K/V matches the F32-storage path within F16 precision (~3
    /// decimal digits) — NOT bit-identical, that bar doesn't apply here (unlike CFG-parallel's same-GPU tests,
    /// which changed dispatch, not numerics). Exercises the monolithic kernel directly; the split-K and
    /// graph-decode paths are asserted OFF for F16 KV separately below.</summary>
    [Theory]
    [InlineData(true)]   // prefill
    [InlineData(false)]  // decode
    public void FlashAttention_F16Kv_MatchesF32Kv_WithinTolerance(bool prefill)
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        using CudaBackend backend = new(0, PtxDir());
        IBackend b = backend;

        const int hq = 8, hkv = 2, d = 64, lk = 37, group = hq / hkv;
        int tq = prefill ? lk : 1;
        int qOffset = prefill ? 0 : lk - 1;
        float scale = 1f / MathF.Sqrt(d);

        using Tensor q = Rnd(1, hq, tq, d);
        using Tensor kF32 = Rnd(1, hkv, lk, d);
        using Tensor vF32 = Rnd(1, hkv, lk, d);

        using Tensor f32Out = new(new TensorShape(1, hq, tq, d), DType.F32);
        b.FlashAttention(f32Out, q, kF32, vF32, lk, group, causal: true, qOffset, scale);
        backend.Sync();

        // Route K/V through the SAME F16-storage path production code uses: append F32 source into an F16
        // resident buffer, then read it back with FlashAttention's F16-KV kernel.
        using Tensor kF16 = new(new TensorShape(1, hkv, lk, d), DType.F16);
        using Tensor vF16 = new(new TensorShape(1, hkv, lk, d), DType.F16);
        b.ResidentAllocateKv(kF16);
        b.ResidentAllocateKv(vF16);
        b.KvCacheAppend(kF16, kF32, offset: 0);
        b.KvCacheAppend(vF16, vF32, offset: 0);
        backend.Sync();

        using Tensor f16Out = new(new TensorShape(1, hq, tq, d), DType.F32);
        b.FlashAttention(f16Out, q, kF16, vF16, lk, group, causal: true, qOffset, scale);
        backend.Sync();

        float* a = (float*)f32Out.DataPointer;
        float* f = (float*)f16Out.DataPointer;
        float maxDiff = 0f, maxRef = 0f;
        for (long i = 0; i < f32Out.ElementCount; i++)
        {
            maxDiff = MathF.Max(maxDiff, MathF.Abs(a[i] - f[i]));
            maxRef = MathF.Max(maxRef, MathF.Abs(a[i]));
        }
        _output.WriteLine($"prefill={prefill}: max |F32-KV - F16-KV| = {maxDiff:E3} (max |F32-KV| = {maxRef:E3})");
        // F16 has ~3 decimal digits of precision; K/V values here are O(1), so an absolute tolerance in that
        // band is the right bar — NOT the CFG-parallel same-GPU tests' bit-identical bar (different kind of
        // change: this narrows storage precision, that only changed dispatch).
        Assert.True(maxDiff <= 5e-3f, $"FlashAttention F16-KV diverges from F32-KV by {maxDiff:E3} (prefill={prefill}) — beyond plausible F16-rounding.");
    }

    /// <summary>Split-K must never engage for F16 KV (LaunchFlashAttentionSplit has no F16-KV variant) — forcing
    /// it via HARTSY_FLASH_SPLIT_FORCE must still produce the monolithic-kernel result, not silently read F16
    /// data through the F32 split kernel's pointer arithmetic (which would corrupt output, not just be slow).</summary>
    [Fact]
    public void FlashAttention_F16Kv_SplitForceEnv_StillMatchesMonolithicF32Kv()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        string? prevForce = Environment.GetEnvironmentVariable("HARTSY_FLASH_SPLIT_FORCE");
        Environment.SetEnvironmentVariable("HARTSY_FLASH_SPLIT_FORCE", "1");
        try
        {
            using CudaBackend backend = new(0, PtxDir());
            IBackend b = backend;
            const int hq = 4, hkv = 1, d = 64, lk = 256, group = hq / hkv;
            float scale = 1f / MathF.Sqrt(d);

            using Tensor q = Rnd(1, hq, 1, d);
            using Tensor kF32 = Rnd(1, hkv, lk, d);
            using Tensor vF32 = Rnd(1, hkv, lk, d);
            using Tensor f32Out = new(new TensorShape(1, hq, 1, d), DType.F32);
            b.FlashAttention(f32Out, q, kF32, vF32, lk, group, causal: true, qOffset: lk - 1, scale);
            backend.Sync();

            using Tensor kF16 = new(new TensorShape(1, hkv, lk, d), DType.F16);
            using Tensor vF16 = new(new TensorShape(1, hkv, lk, d), DType.F16);
            b.ResidentAllocateKv(kF16);
            b.ResidentAllocateKv(vF16);
            b.KvCacheAppend(kF16, kF32, offset: 0);
            b.KvCacheAppend(vF16, vF32, offset: 0);
            backend.Sync();

            using Tensor f16Out = new(new TensorShape(1, hq, 1, d), DType.F32);
            b.FlashAttention(f16Out, q, kF16, vF16, lk, group, causal: true, qOffset: lk - 1, scale);
            backend.Sync();

            float* a = (float*)f32Out.DataPointer;
            float* f = (float*)f16Out.DataPointer;
            float maxDiff = 0f;
            for (long i = 0; i < f32Out.ElementCount; i++) maxDiff = MathF.Max(maxDiff, MathF.Abs(a[i] - f[i]));
            _output.WriteLine($"HARTSY_FLASH_SPLIT_FORCE=1, F16 KV: max diff vs F32-KV monolithic = {maxDiff:E3}");
            Assert.True(maxDiff <= 5e-3f, $"F16-KV output diverged by {maxDiff:E3} under split-force — split-K may have engaged for F16 KV.");
        }
        finally
        {
            Environment.SetEnvironmentVariable("HARTSY_FLASH_SPLIT_FORCE", prevForce);
        }
    }
}
