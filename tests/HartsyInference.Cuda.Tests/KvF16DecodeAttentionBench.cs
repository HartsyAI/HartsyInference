using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using System.Diagnostics;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>Diagnostic bench for "F16 KV decodes slower than F32 KV" (MINIMAX_MUSIC3_PERF.md §2b). Runs the
/// 2x2 of {KV dtype} x {split-K allowed} at MiniMax Music 3's real decode shape so the two candidate causes
/// separate: losing the split-K fast path (F16 is excluded from it by dispatch) versus the monolithic F16
/// kernel being intrinsically slower than the monolithic F32 one. Buffers rotate every call — a fixed pair
/// sits in L2 and flatters both arms unequally.</summary>
[Collection("CudaSerial")]
public sealed unsafe class KvF16DecodeAttentionBench
{
    private readonly ITestOutputHelper _output;
    public KvF16DecodeAttentionBench(ITestOutputHelper output) => _output = output;

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

    private static Tensor Rnd(int a, int b, int c, int d)
    {
        Tensor t = new(new TensorShape(a, b, c, d), DType.F32);
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++) p[i] = Rand();
        return t;
    }

    [Fact]
    public void Bench_F16Kv_Vs_F32Kv_DecodeShape()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        // MiniMax Music 3 global LM: Qwen3-8B geometry, one query row per decode step.
        const int hq = 32, hkv = 8, d = 128, group = hq / hkv, tq = 1;
        const int rotation = 8, warmup = 30, iters = 200, batches = 5;
        float scale = 1f / MathF.Sqrt(d);

        using CudaBackend backend = new(0, PtxDir());
        IBackend b = backend;
        _output.WriteLine($"baseBlocks={1 * hq * tq}");

        foreach (int lk in new[] { 128, 375, 750, 1500 })
        {
            List<Tensor> keep = [];
            Tensor[] kF32 = new Tensor[rotation], vF32 = new Tensor[rotation];
            Tensor[] kF16 = new Tensor[rotation], vF16 = new Tensor[rotation];
            for (int i = 0; i < rotation; i++)
            {
                Tensor ks = Rnd(1, hkv, lk, d), vs = Rnd(1, hkv, lk, d);
                keep.Add(ks); keep.Add(vs);
                kF32[i] = new Tensor(new TensorShape(1, hkv, lk, d), DType.F32);
                vF32[i] = new Tensor(new TensorShape(1, hkv, lk, d), DType.F32);
                kF16[i] = new Tensor(new TensorShape(1, hkv, lk, d), DType.F16);
                vF16[i] = new Tensor(new TensorShape(1, hkv, lk, d), DType.F16);
                foreach (Tensor t in new[] { kF32[i], vF32[i], kF16[i], vF16[i] }) { keep.Add(t); b.ResidentAllocateKv(t); }
                b.KvCacheAppend(kF32[i], ks, 0); b.KvCacheAppend(vF32[i], vs, 0);
                b.KvCacheAppend(kF16[i], ks, 0); b.KvCacheAppend(vF16[i], vs, 0);
            }
            backend.Sync();

            using Tensor q = Rnd(1, hq, tq, d);
            using Tensor outT = new(new TensorShape(1, hq, tq, d), DType.F32);

            foreach (bool f16 in new[] { false, true })
            {
                foreach (bool splitOff in new[] { false, true })
                {
                    Environment.SetEnvironmentVariable("HARTSY_FLASH_SPLIT_OFF", splitOff ? "1" : null);
                    Tensor[] kk = f16 ? kF16 : kF32, vv = f16 ? vF16 : vF32;
                    for (int i = 0; i < warmup; i++)
                        b.FlashAttention(outT, q, kk[i % rotation], vv[i % rotation], lk, group, true, lk - 1, scale);
                    backend.Sync();

                    double[] us = new double[batches];
                    for (int t = 0; t < batches; t++)
                    {
                        Stopwatch sw = Stopwatch.StartNew();
                        for (int i = 0; i < iters; i++)
                            b.FlashAttention(outT, q, kk[i % rotation], vv[i % rotation], lk, group, true, lk - 1, scale);
                        backend.Sync();
                        sw.Stop();
                        us[t] = sw.Elapsed.TotalMilliseconds * 1000.0 / iters;
                    }
                    Array.Sort(us);
                    double mean = us.Average();
                    _output.WriteLine($"kvLen={lk,5}  kv={(f16 ? "F16" : "F32")}  splitOff={splitOff,-5}  "
                        + $"mean={mean,7:F2} us  min={us[0],7:F2}  max={us[^1],7:F2}  spread={(us[^1] - us[0]),6:F2}");
                }
            }
            Environment.SetEnvironmentVariable("HARTSY_FLASH_SPLIT_OFF", null);
            foreach (Tensor t in keep) t.Dispose();
        }
    }
}
