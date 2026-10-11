using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>The tensor-core FlashAttention-2 prefill kernel against the CPU reference: grouped-query heads, causal masking with an
/// absolute query offset, a key buffer whose stride exceeds the valid length, a sliding window and partial query tiles.</summary>
[Collection("CudaSerial")]
public sealed unsafe class CudaFa2AttentionTests
{
    private readonly ITestOutputHelper _output;
    public CudaFa2AttentionTests(ITestOutputHelper output) => _output = output;

    private static string PtxDir()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        return Directory.Exists(dir) ? dir : Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
    }

    private static Tensor Random(Random rng, params long[] dims)
    {
        Tensor t = new(new TensorShape(dims), DType.F32);
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++) p[i] = (float)((rng.NextDouble() - 0.5) * 2.0);
        return t;
    }

    [Trait("Category", "GpuIntegration")]
    [Theory]
    [InlineData(128, 8, 2, 100, 0, 0, 100)]      // partial query tile, no cache slack
    [InlineData(128, 8, 2, 128, 0, 0, 300)]      // key buffer stride larger than the valid length
    [InlineData(128, 8, 2, 70, 197, 0, 500)]     // chunked prefill: queries start mid-sequence
    [InlineData(128, 8, 2, 90, 60, 48, 400)]     // sliding window
    [InlineData(64, 6, 2, 150, 0, 0, 150)]
    [InlineData(64, 4, 4, 33, 31, 20, 128)]      // MHA, window
    [InlineData(128, 4, 1, 16, 5, 0, 64)]        // the smallest block the kernel takes
    public void Fa2_MatchesCpuReference(int d, int hq, int hkv, int tq, int qOffset, int window, int keyStride)
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        Random rng = new(d * 31 + tq);
        int kvLen = qOffset + tq;
        int group = hq / hkv;
        float scale = 1f / MathF.Sqrt(d);
        using Tensor q = Random(rng, 1, hq, tq, d);
        using Tensor k = Random(rng, 1, hkv, keyStride, d);
        using Tensor v = Random(rng, 1, hkv, keyStride, d);
        using Tensor expected = new(new TensorShape(1, hq, tq, d), DType.F32);
        AttentionReference.FlashAttention(expected, q, k, v, kvLen, group, true, qOffset, scale, 0f, null, window, null);

        using CudaBackend cuda = new(0, PtxDir());
        using Tensor actual = new(new TensorShape(1, hq, tq, d), DType.F32);
        cuda.FlashAttention(actual, q, k, v, kvLen, group, true, qOffset, scale, 0f, null, window, null);
        cuda.Sync();

        float* pe = (float*)expected.DataPointer, pa = (float*)actual.DataPointer;
        double diff = 0, norm = 0;
        for (long i = 0; i < expected.ElementCount; i++)
        {
            double e = pe[i], a = pa[i];
            diff += (a - e) * (a - e);
            norm += e * e;
        }
        double rel = Math.Sqrt(diff / Math.Max(norm, 1e-30));
        _output.WriteLine($"d={d} hq={hq} hkv={hkv} tq={tq} off={qOffset} win={window} stride={keyStride}: relative L2 error {rel:E3}");
        Assert.True(rel <= 3e-3, $"FA2 diverges from the reference by {rel:E3}");
    }

    [Trait("Category", "GpuIntegration")]
    [Fact]
    public void Fa2_PrefillThroughput()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        const int d = 128, hq = 16, hkv = 16, tq = 4096;
        Random rng = new(5);
        using Tensor q = Random(rng, 1, hq, tq, d);
        using Tensor k = Random(rng, 1, hkv, tq, d);
        using Tensor v = Random(rng, 1, hkv, tq, d);
        using CudaBackend cuda = new(0, PtxDir());
        using Tensor o = new(new TensorShape(1, hq, tq, d), DType.F32);
        float scale = 1f / MathF.Sqrt(d);
        cuda.FlashAttention(o, q, k, v, tq, 1, true, 0, scale);   // warm-up and upload
        cuda.Sync();
        Stopwatch sw = Stopwatch.StartNew();
        const int reps = 5;
        for (int i = 0; i < reps; i++) cuda.FlashAttention(o, q, k, v, tq, 1, true, 0, scale);
        cuda.Sync();
        double ms = sw.Elapsed.TotalMilliseconds / reps;
        double tflops = 4.0 * tq * (double)tq * d * hq / 2 / (ms * 1e9);
        _output.WriteLine($"FlashAttention tq={tq} d={d} hq={hq}: {ms:F2} ms per call, {tflops:F1} TFLOP/s causal");
    }

    [Trait("Category", "GpuIntegration")]
    [Theory]
    [InlineData(128, 32, 4, 1000, 1200, 0, 0f)]    // Qwen3 / Llama-3 shape
    [InlineData(128, 16, 16, 300, 300, 0, 0f)]     // MHA (OLMoE)
    [InlineData(64, 24, 8, 777, 800, 0, 0f)]       // Granite
    [InlineData(128, 8, 2, 500, 512, 100, 0f)]     // sliding window
    [InlineData(128, 8, 2, 90, 96, 0, 30f)]        // soft-cap
    [InlineData(128, 12, 1, 65, 65, 0, 0f)]        // one KV head for 12 query heads
    public void DecodeGqa_MatchesCpuReference(int d, int hq, int hkv, int kvLen, int keyStride, int window, float softcap)
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        Random rng = new(d + hq * 7 + kvLen);
        int group = hq / hkv;
        int qOffset = kvLen - 1;
        float scale = 1f / MathF.Sqrt(d);
        using Tensor q = Random(rng, 1, hq, 1, d);
        using Tensor k = Random(rng, 1, hkv, keyStride, d);
        using Tensor v = Random(rng, 1, hkv, keyStride, d);
        using Tensor expected = new(new TensorShape(1, hq, 1, d), DType.F32);
        AttentionReference.FlashAttention(expected, q, k, v, kvLen, group, true, qOffset, scale, softcap, null, window, null);

        using CudaBackend cuda = new(0, PtxDir());
        using Tensor actual = new(new TensorShape(1, hq, 1, d), DType.F32);
        cuda.FlashAttention(actual, q, k, v, kvLen, group, true, qOffset, scale, softcap, null, window, null);
        cuda.Sync();
        float* pe = (float*)expected.DataPointer, pa = (float*)actual.DataPointer;
        double diff = 0, norm = 0;
        for (long i = 0; i < expected.ElementCount; i++)
        {
            double e = pe[i], a = pa[i];
            diff += (a - e) * (a - e);
            norm += e * e;
        }
        double rel = Math.Sqrt(diff / Math.Max(norm, 1e-30));
        _output.WriteLine($"decode d={d} hq={hq} hkv={hkv} kvLen={kvLen} win={window} cap={softcap}: relative L2 error {rel:E3}");
        Assert.True(rel <= 1e-5, $"grouped-query decode diverges from the reference by {rel:E3}");
    }

    [Trait("Category", "GpuIntegration")]
    [Fact]
    public void DecodeGqa_Throughput()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        const int d = 128, hq = 32, hkv = 4, kvLen = 4096;
        Random rng = new(9);
        using Tensor q = Random(rng, 1, hq, 1, d);
        using Tensor k = Random(rng, 1, hkv, kvLen, d);
        using Tensor v = Random(rng, 1, hkv, kvLen, d);
        using CudaBackend cuda = new(0, PtxDir());
        using Tensor o = new(new TensorShape(1, hq, 1, d), DType.F32);
        float scale = 1f / MathF.Sqrt(d);
        cuda.FlashAttention(o, q, k, v, kvLen, hq / hkv, true, kvLen - 1, scale);
        cuda.Sync();
        Stopwatch sw = Stopwatch.StartNew();
        const int reps = 50;
        for (int i = 0; i < reps; i++) cuda.FlashAttention(o, q, k, v, kvLen, hq / hkv, true, kvLen - 1, scale);
        cuda.Sync();
        double us = sw.Elapsed.TotalMilliseconds / reps * 1000;
        double gb = 2.0 * hkv * kvLen * d * sizeof(float) / 1e9;
        _output.WriteLine($"decode attention kvLen={kvLen} hq={hq} hkv={hkv}: {us:F1} us per call, {gb / (us * 1e-6):F0} GB/s of F32 KV");
    }

    [Trait("Category", "GpuIntegration")]
    [Theory]
    [InlineData(false)]   // rope + scatter of separate q, k, v
    [InlineData(true)]    // per-head QK-norm + rope + scatter (Qwen3)
    public void F16KvCache_ScatterMatchesF32(bool qkNorm)
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        const int hq = 8, hkv = 2, d = 128, maxSeq = 64, pos = 17;
        Random rng = new(qkNorm ? 3 : 4);
        using CudaBackend cuda = new(0, PtxDir());
        ulong posBuf = CudaMemory.Allocate(2 * sizeof(int));
        int* hostPos = stackalloc int[2] { pos + 1, pos };
        CudaMemory.CopyHostToDevice(posBuf, hostPos, 2 * sizeof(int));
        using Tensor cos = Random(rng, maxSeq, d), sin = Random(rng, maxSeq, d);
        using Tensor q = Random(rng, 1, hq, 1, d), k = Random(rng, 1, hkv, 1, d), v = Random(rng, 1, hkv, 1, d);
        using Tensor qk = Random(rng, 1, 1, (hq + hkv) * d);
        using Tensor qNorm = Random(rng, d), kNorm = Random(rng, d);

        Tensor[] RunWith(DType cacheType)
        {
            Tensor kc = new(new TensorShape(1, hkv, maxSeq, d), cacheType), vc = new(new TensorShape(1, hkv, maxSeq, d), cacheType);
            Tensor qo = new(new TensorShape(1, hq, 1, d), DType.F32);
            if (qkNorm) cuda.QkNormRopeScatterVDecodeStep(qo, kc, vc, qk, v, qNorm, kNorm, 1e-6f, cos, sin, hq, hkv, d, d, false, posBuf);
            else cuda.RopeScatterKvDecodeStep(qo, kc, vc, q, k, v, cos, sin, hq, hkv, d, d, false, posBuf);
            cuda.Sync();
            return [qo, kc, vc];
        }
        Tensor[] f32 = RunWith(DType.F32);
        Tensor[] f16 = RunWith(DType.F16);
        float* q32 = (float*)f32[0].DataPointer, q16 = (float*)f16[0].DataPointer;
        for (int i = 0; i < hq * d; i++) Assert.Equal(q32[i], q16[i]);   // q is untouched by the cache type
        for (int c = 1; c <= 2; c++)
        {
            float* a = (float*)f32[c].DataPointer;
            Half* b = (Half*)f16[c].DataPointer;
            for (int h = 0; h < hkv; h++)
                for (int i = 0; i < d; i++)
                {
                    long at = ((long)h * maxSeq + pos) * d + i;
                    Assert.Equal((float)(Half)a[at], (float)b[at]);   // the F16 cache holds the F32 value rounded to half
                }
        }
        CudaMemory.Free(posBuf);
        foreach (Tensor t in f32.Concat(f16)) t.Dispose();
        _output.WriteLine($"F16 KV scatter matches F32 rounded to half (qkNorm={qkNorm})");
    }

    [Trait("Category", "GpuIntegration")]
    [Theory]
    [InlineData(512, 1)]
    [InlineData(1024, 4)]
    [InlineData(2048, 1)]
    [InlineData(2048, 7)]
    [InlineData(4096, 3)]
    [InlineData(8192, 2)]
    public void FastRmsNormQ8_IsBitIdenticalToTheSharedMemoryKernels(int dim, int rows)
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        Random rng = new(dim + rows);
        using CudaBackend cuda = new(0, PtxDir());
        using Tensor weight = Random(rng, dim);
        using Tensor a = Random(rng, 1, rows, dim), b = Random(rng, 1, rows, dim);
        for (int i = 0; i < rows * dim; i += 97) ((float*)a.DataPointer)[i] *= 40f;   // a few large values stretch the scales

        (byte[] q, float[] d, float[] sum, float[] main, float[] resid) Run(bool addNorm, bool reference)
        {
            cuda.Kernels!.ForceReferenceNorm = reference;
            using Tensor norm = new(new TensorShape(1, rows, dim), DType.F32);
            using Tensor resid = new(new TensorShape(1, rows, dim), DType.F32);
            if (addNorm) cuda.AddRmsNormEmitQ8(resid, norm, a, b, weight, 1e-6f);
            else cuda.RmsNormEmitQ8(norm, a, weight, 1e-6f);
            cuda.Sync();
            ulong xq, xd, xs;
            int blocks = rows * (dim / 32);
            byte[] q = new byte[rows * dim];
            float[] dd = new float[blocks], ss = new float[blocks];
            // The sidecar of a multi-row call is not registered (decode rows only); read it by position from the op's buffers when present.
            bool hasSidecar = GpuTransferHelper.TryGetSidecar(norm, dim, out xq, out xd, out xs);
            if (rows == 1) Assert.True(hasSidecar, "a single-row call must publish its Q8_1 sidecar");
            if (hasSidecar)
            {
                fixed (byte* pq = q) CudaMemory.CopyDeviceToHost(pq, xq, (nuint)q.Length);
                fixed (float* pd = dd) CudaMemory.CopyDeviceToHost(pd, xd, (nuint)(blocks * sizeof(float)));
                fixed (float* ps = ss) CudaMemory.CopyDeviceToHost(ps, xs, (nuint)(blocks * sizeof(float)));
            }
            float[] n = new float[rows * dim], r = new float[rows * dim];
            for (int i = 0; i < n.Length; i++) { n[i] = ((float*)norm.DataPointer)[i]; r[i] = addNorm ? ((float*)resid.DataPointer)[i] : 0f; }
            return (q, dd, ss, n, r);
        }
        foreach (bool addNorm in new[] { false, true })
        {
            (byte[] q, float[] d, float[] sum, float[] main, float[] resid) reference = Run(addNorm, reference: true);
            (byte[] q, float[] d, float[] sum, float[] main, float[] resid) fast = Run(addNorm, reference: false);
            Assert.Equal(reference.main, fast.main);
            Assert.Equal(reference.resid, fast.resid);
            Assert.Equal(reference.q, fast.q);
            Assert.Equal(reference.d, fast.d);
            Assert.Equal(reference.sum, fast.sum);
        }
        cuda.Kernels!.ForceReferenceNorm = false;
        _output.WriteLine($"fast RMSNorm Q8 == reference bit for bit (dim {dim}, rows {rows})");
    }
}
