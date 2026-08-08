using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Cuda;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>Backend-parity tests for the DiT glue kernels (Phase 1): RmsNorm, Tanh, broadcast affine,
/// gated residual, AdaLN modulation split, and CFG+Euler. Each op runs on the CPU backend (the
/// numerical reference, via IBackend default implementations) and the CUDA backend (PTX kernels),
/// then compares element-wise. FP32 PTX-vs-CPU tolerance per KERNEL.md is 1e-5; reductions get 1e-4.
/// Skips cleanly when CUDA is unavailable.</summary>
[Collection("CudaSerial")]
public sealed unsafe class DitGlueKernelTests
{
    private readonly ITestOutputHelper _output;
    public DitGlueKernelTests(ITestOutputHelper output) => _output = output;

    private static string PtxDir()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(dir))
            dir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        return dir;
    }

    private static Tensor Random(TensorShape shape, int seed, float lo = -1f, float hi = 1f)
    {
        Tensor t = new Tensor(shape, DType.F32);
        float* p = (float*)t.DataPointer;
        Random rng = new Random(seed);
        long n = shape.ElementCount;
        for (long i = 0; i < n; i++) p[i] = (float)(rng.NextDouble() * (hi - lo) + lo);
        return t;
    }

    /// <summary>Runs a CUDA op and forces the lazy device-to-host sync of its outputs WHILE the backend
    /// (and CUDA context) is still alive — otherwise reading DataPointer after dispose returns the
    /// unwritten CPU buffer.</summary>
    private static void RunCuda(Action<CudaBackend> op, params Tensor[] outputs)
    {
        using CudaBackend cuda = new CudaBackend(0, PtxDir());
        op(cuda);
        cuda.Sync();
        foreach (Tensor t in outputs)
            _ = *(float*)t.DataPointer;
    }

    private void AssertClose(Tensor cpu, Tensor cuda, float tol, string name)
    {
        float* a = (float*)cpu.DataPointer;
        float* b = (float*)cuda.DataPointer;
        long n = cpu.ElementCount;
        double maxErr = 0;
        for (long i = 0; i < n; i++)
        {
            double e = Math.Abs(a[i] - b[i]);
            if (e > maxErr) maxErr = e;
        }
        _output.WriteLine($"{name}: max_err={maxErr:E3} over {n} elems (tol {tol:E0})");
        Assert.True(maxErr < tol, $"{name}: max_err {maxErr:E3} exceeds tol {tol:E0}");
    }

    [Fact]
    public void RmsNorm_Cpu_Vs_Cuda()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }

        foreach ((int rows, int dim) in new[] { (2, 64), (3, 32), (5, 3072), (1, 256) })
        {
            using Tensor input = Random(new TensorShape(rows, dim), seed: 100 + dim);
            using Tensor weight = Random(new TensorShape(dim), seed: 200 + dim, lo: 0.5f, hi: 1.5f);
            using Tensor cpuOut = new Tensor(new TensorShape(rows, dim), DType.F32);
            using Tensor cudaOut = new Tensor(new TensorShape(rows, dim), DType.F32);

            IBackend cpu = new CpuBackend();
            cpu.RmsNorm(cpuOut, input, weight, 1e-6f);
            cpu.Dispose();

            RunCuda(c => c.RmsNorm(cudaOut, input, weight, 1e-6f), cudaOut);

            AssertClose(cpuOut, cudaOut, 1e-4f, $"RmsNorm[{rows}x{dim}]");
        }
    }

    [Fact]
    public void Tanh_Cpu_Vs_Cuda()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        using Tensor input = Random(new TensorShape(128), seed: 1, lo: -4f, hi: 4f);
        using Tensor cpuOut = new Tensor(new TensorShape(128), DType.F32);
        using Tensor cudaOut = new Tensor(new TensorShape(128), DType.F32);

        IBackend cpu = new CpuBackend();
        cpu.Tanh(cpuOut, input);
        cpu.Dispose();

        RunCuda(c => c.Tanh(cudaOut, input), cudaOut);

        AssertClose(cpuOut, cudaOut, 1e-5f, "Tanh");
    }

    [Fact]
    public void AffineBroadcastLastDim_Cpu_Vs_Cuda()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        const int b = 2, s = 5, d = 16;
        using Tensor input = Random(new TensorShape(b, s, d), seed: 11);
        using Tensor scale = Random(new TensorShape(b, d), seed: 12);
        using Tensor shift = Random(new TensorShape(b, d), seed: 13);

        foreach (bool withShift in new[] { false, true })
        {
            using Tensor cpuOut = new Tensor(new TensorShape(b, s, d), DType.F32);
            using Tensor cudaOut = new Tensor(new TensorShape(b, s, d), DType.F32);
            Tensor? sh = withShift ? shift : null;

            IBackend cpu = new CpuBackend();
            cpu.AffineBroadcastLastDim(cpuOut, input, scale, sh);
            cpu.Dispose();

            RunCuda(c => c.AffineBroadcastLastDim(cudaOut, input, scale, sh), cudaOut);

            AssertClose(cpuOut, cudaOut, 1e-5f, $"Affine(shift={withShift})");
        }
    }

    [Fact]
    public void GatedResidualLastDim_Cpu_Vs_Cuda()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        const int b = 2, s = 7, d = 16;
        using Tensor residual = Random(new TensorShape(b, s, d), seed: 21);
        using Tensor value = Random(new TensorShape(b, s, d), seed: 22);
        using Tensor gate = Random(new TensorShape(b, d), seed: 23);
        using Tensor cpuOut = new Tensor(new TensorShape(b, s, d), DType.F32);
        using Tensor cudaOut = new Tensor(new TensorShape(b, s, d), DType.F32);

        IBackend cpu = new CpuBackend();
        cpu.GatedResidualLastDim(cpuOut, residual, value, gate);
        cpu.Dispose();

        RunCuda(c => c.GatedResidualLastDim(cudaOut, residual, value, gate), cudaOut);

        AssertClose(cpuOut, cudaOut, 1e-5f, "GatedResidual");
    }

    [Fact]
    public void ModulationSplit4_Cpu_Vs_Cuda()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        const int b = 2, d = 16;
        using Tensor proj = Random(new TensorShape(b, 4 * d), seed: 31, lo: -2f, hi: 2f);

        Tensor[] MakeOuts() =>
        [
            new Tensor(new TensorShape(b, d), DType.F32),
            new Tensor(new TensorShape(b, d), DType.F32),
            new Tensor(new TensorShape(b, d), DType.F32),
            new Tensor(new TensorShape(b, d), DType.F32),
        ];

        Tensor[] cpuO = MakeOuts();
        Tensor[] cudaO = MakeOuts();

        IBackend cpu = new CpuBackend();
        cpu.ModulationSplit4(cpuO[0], cpuO[1], cpuO[2], cpuO[3], proj);
        cpu.Dispose();

        RunCuda(c => c.ModulationSplit4(cudaO[0], cudaO[1], cudaO[2], cudaO[3], proj), cudaO[0], cudaO[1], cudaO[2], cudaO[3]);

        string[] names = ["scaleMsa", "gateMsa", "scaleMlp", "gateMlp"];
        for (int i = 0; i < 4; i++)
            AssertClose(cpuO[i], cudaO[i], 1e-5f, $"Modulation:{names[i]}");

        foreach (Tensor t in cpuO) t.Dispose();
        foreach (Tensor t in cudaO) t.Dispose();
    }

    [Fact]
    public void CfgEulerStep_Cpu_Vs_Cuda()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        const int n = 257;
        using Tensor pos = Random(new TensorShape(n), seed: 41);
        using Tensor neg = Random(new TensorShape(n), seed: 42);
        using Tensor zCpu = Random(new TensorShape(n), seed: 43);
        using Tensor zCuda = new Tensor(new TensorShape(n), DType.F32);
        Buffer.MemoryCopy((void*)zCpu.DataPointer, (void*)zCuda.DataPointer, n * sizeof(float), n * sizeof(float));

        const float guidance = 3.5f, delta = 0.12f;
        IBackend cpu = new CpuBackend();
        cpu.CfgEulerStep(zCpu, pos, neg, guidance, delta);
        cpu.Dispose();

        RunCuda(c => c.CfgEulerStep(zCuda, pos, neg, guidance, delta), zCuda);

        AssertClose(zCpu, zCuda, 1e-5f, "CfgEulerStep");
    }

    [Fact]
    public void ApplyRope_Cpu_Vs_Cuda()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        const int b = 1, l = 6, nh = 3, hd = 32;
        using Tensor cos = Random(new TensorShape(b, l, hd), seed: 51);
        using Tensor sin = Random(new TensorShape(b, l, hd), seed: 52);
        using Tensor qSrc = Random(new TensorShape(b, l, nh, hd), seed: 53);
        using Tensor kSrc = Random(new TensorShape(b, l, nh, hd), seed: 54);

        long n = qSrc.ElementCount;
        using Tensor qCpu = new Tensor(new TensorShape(b, l, nh, hd), DType.F32);
        using Tensor kCpu = new Tensor(new TensorShape(b, l, nh, hd), DType.F32);
        using Tensor qCuda = new Tensor(new TensorShape(b, l, nh, hd), DType.F32);
        using Tensor kCuda = new Tensor(new TensorShape(b, l, nh, hd), DType.F32);
        Buffer.MemoryCopy((void*)qSrc.DataPointer, (void*)qCpu.DataPointer, n * 4, n * 4);
        Buffer.MemoryCopy((void*)kSrc.DataPointer, (void*)kCpu.DataPointer, n * 4, n * 4);
        Buffer.MemoryCopy((void*)qSrc.DataPointer, (void*)qCuda.DataPointer, n * 4, n * 4);
        Buffer.MemoryCopy((void*)kSrc.DataPointer, (void*)kCuda.DataPointer, n * 4, n * 4);

        IBackend cpu = new CpuBackend();
        cpu.ApplyRope(qCpu, kCpu, cos, sin);
        cpu.Dispose();

        RunCuda(c => c.ApplyRope(qCuda, kCuda, cos, sin), qCuda, kCuda);

        AssertClose(qCpu, qCuda, 1e-5f, "ApplyRope:q");
        AssertClose(kCpu, kCuda, 1e-5f, "ApplyRope:k");
    }

    [Fact]
    public void SliceLastDim_Cpu_Vs_Cuda()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        const int b = 2, l = 5, h = 16;
        using Tensor fused = Random(new TensorShape(b, l, 3 * h), seed: 61);

        foreach (int chunk in new[] { 0, 1, 2 })
        {
            using Tensor cpuOut = new Tensor(new TensorShape(b, l, h), DType.F32);
            using Tensor cudaOut = new Tensor(new TensorShape(b, l, h), DType.F32);
            int offset = chunk * h;

            IBackend cpu = new CpuBackend();
            cpu.SliceLastDim(cpuOut, fused, offset);
            cpu.Dispose();

            RunCuda(c => c.SliceLastDim(cudaOut, fused, offset), cudaOut);

            AssertClose(cpuOut, cudaOut, 1e-6f, $"SliceLastDim[offset={offset}]");
        }
    }

    [Fact]
    public void MaskRows_Cpu_Vs_Cuda()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        const int b = 2, s = 6, c = 16;
        using Tensor input = Random(new TensorShape(b, s, c), seed: 71);
        using Tensor mask = new Tensor(new TensorShape(b * s), DType.F32);
        float* mp = (float*)mask.DataPointer;
        for (int i = 0; i < b * s; i++) mp[i] = (i % 3 == 0) ? 1f : 0f;
        using Tensor cpuOut = new Tensor(new TensorShape(b, s, c), DType.F32);
        using Tensor cudaOut = new Tensor(new TensorShape(b, s, c), DType.F32);

        IBackend cpu = new CpuBackend();
        cpu.MaskRows(cpuOut, input, mask);
        cpu.Dispose();
        RunCuda(g => g.MaskRows(cudaOut, input, mask), cudaOut);
        AssertClose(cpuOut, cudaOut, 1e-6f, "MaskRows");
    }

    [Fact]
    public void AddScalar_Cpu_Vs_Cuda()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        using Tensor input = Random(new TensorShape(2, 16), seed: 81);
        using Tensor cpuOut = new Tensor(new TensorShape(2, 16), DType.F32);
        using Tensor cudaOut = new Tensor(new TensorShape(2, 16), DType.F32);

        IBackend cpu = new CpuBackend();
        cpu.AddScalar(cpuOut, input, 1.0f);
        cpu.Dispose();
        RunCuda(g => g.AddScalar(cudaOut, input, 1.0f), cudaOut);
        AssertClose(cpuOut, cudaOut, 1e-6f, "AddScalar");
    }

    [Fact]
    public void LayerNormNoAffine_Cpu_Vs_Cuda()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        foreach ((int rows, int dim) in new[] { (3, 64), (2, 3072) })
        {
            using Tensor input = Random(new TensorShape(rows, dim), seed: 90 + dim);
            using Tensor cpuOut = new Tensor(new TensorShape(rows, dim), DType.F32);
            using Tensor cudaOut = new Tensor(new TensorShape(rows, dim), DType.F32);

            IBackend cpu = new CpuBackend();
            cpu.LayerNormNoAffine(cpuOut, input, 1e-6f);
            cpu.Dispose();
            RunCuda(g => g.LayerNormNoAffine(cudaOut, input, 1e-6f), cudaOut);
            AssertClose(cpuOut, cudaOut, 1e-4f, $"LayerNormNoAffine[{rows}x{dim}]");
        }
    }

    [Fact]
    public void IndexAddRows_Cpu_Vs_Cuda()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        const int rows = 7, dim = 16;
        using Tensor table = Random(new TensorShape(2, dim), seed: 95);
        using Tensor indices = new Tensor(new TensorShape(rows), DType.I32);
        int* ip = (int*)indices.DataPointer;
        for (int i = 0; i < rows; i++) ip[i] = i % 2;
        using Tensor hSrc = Random(new TensorShape(rows, dim), seed: 96);

        using Tensor hCpu = new Tensor(new TensorShape(rows, dim), DType.F32);
        using Tensor hCuda = new Tensor(new TensorShape(rows, dim), DType.F32);
        Buffer.MemoryCopy((void*)hSrc.DataPointer, (void*)hCpu.DataPointer, rows * dim * 4L, rows * dim * 4L);
        Buffer.MemoryCopy((void*)hSrc.DataPointer, (void*)hCuda.DataPointer, rows * dim * 4L, rows * dim * 4L);

        IBackend cpu = new CpuBackend();
        cpu.IndexAddRows(hCpu, table, indices);
        cpu.Dispose();
        RunCuda(g => g.IndexAddRows(hCuda, table, indices), hCuda);
        AssertClose(hCpu, hCuda, 1e-6f, "IndexAddRows");
    }

    [Fact]
    public void ScatterAndSliceRows_Cpu_Vs_Cuda()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        const int numText = 5, numImg = 9, dim = 16;
        using Tensor z = Random(new TensorShape(1, numImg, dim), seed: 101);

        using Tensor scCpu = new Tensor(new TensorShape(1, numText + numImg, dim), DType.F32);
        using Tensor scCuda = new Tensor(new TensorShape(1, numText + numImg, dim), DType.F32);
        IBackend cpu = new CpuBackend();
        cpu.ScatterRowsAfter(scCpu, z, numText);
        RunCuda(g => g.ScatterRowsAfter(scCuda, z, numText), scCuda);
        AssertClose(scCpu, scCuda, 1e-6f, "ScatterRowsAfter");

        // Round-trip: slicing image rows back out of the scattered tensor must recover z.
        using Tensor slCpu = new Tensor(new TensorShape(1, numImg, dim), DType.F32);
        using Tensor slCuda = new Tensor(new TensorShape(1, numImg, dim), DType.F32);
        cpu.SliceRows(slCpu, scCpu, numText);
        cpu.Dispose();
        RunCuda(g => g.SliceRows(slCuda, scCuda, numText), slCuda);
        AssertClose(slCpu, slCuda, 1e-6f, "SliceRows");
        AssertClose(z, slCuda, 1e-6f, "SliceRows-recovers-z");
    }

    /// <summary>SliceRowsGeneric on F32 must agree with SliceRows (same result, different implementation path).</summary>
    [Fact]
    public void SliceRowsGeneric_F32_Cpu_Vs_Cuda_MatchesSliceRows()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        const int rows = 12, offset = 5, keep = 4, dim = 8;
        using Tensor src = Random(new TensorShape(rows, dim), seed: 202);

        using Tensor refCpu = new Tensor(new TensorShape(keep, dim), DType.F32);
        IBackend cpu = new CpuBackend();
        cpu.SliceRows(refCpu, src, offset);
        cpu.Dispose();

        using Tensor genCpu = new Tensor(new TensorShape(keep, dim), DType.F32);
        IBackend cpu2 = new CpuBackend();
        cpu2.SliceRowsGeneric(genCpu, src, offset);
        cpu2.Dispose();
        AssertClose(refCpu, genCpu, 1e-6f, "SliceRowsGeneric-vs-SliceRows-cpu");

        using Tensor genCuda = new Tensor(new TensorShape(keep, dim), DType.F32);
        RunCuda(g => g.SliceRowsGeneric(genCuda, src, offset), genCuda);
        AssertClose(refCpu, genCuda, 1e-6f, "SliceRowsGeneric-cuda");
    }

    /// <summary>The trap: a sliced fp8 chunk that loses its parent's per-tensor <see cref="Tensor.Fp8ScaleFactor"/>
    /// silently mis-scales every downstream fp8 GEMM with correct-looking output and no failing test elsewhere —
    /// so this asserts the scale survives explicitly, on both backends, in addition to the sliced bytes.</summary>
    [Fact]
    public unsafe void SliceRowsGeneric_Fp8_CarriesScaleFactor_Cpu_Vs_Cuda()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        const int rows = 6, offset = 2, keep = 3, dim = 4;
        const float scale = 2.5f;
        using Tensor srcF32 = Random(new TensorShape(rows, dim), seed: 303, lo: -4f, hi: 4f);
        using Tensor srcFp8 = srcF32.CastTo(DType.F8E4M3);
        srcFp8.Fp8ScaleFactor = scale;

        using Tensor cpuOut = new Tensor(new TensorShape(keep, dim), DType.F8E4M3);
        IBackend cpu = new CpuBackend();
        cpu.SliceRowsGeneric(cpuOut, srcFp8, offset);
        cpu.Dispose();
        Assert.Equal(scale, cpuOut.Fp8ScaleFactor);

        using Tensor cudaOut = new Tensor(new TensorShape(keep, dim), DType.F8E4M3);
        using CudaBackend gpu = new CudaBackend(0, PtxDir());
        gpu.SliceRowsGeneric(cudaOut, srcFp8, offset);
        gpu.Sync();
        Assert.Equal(scale, cudaOut.Fp8ScaleFactor);

        byte* pSrc = (byte*)srcFp8.DataPointer;
        byte* pCpu = (byte*)cpuOut.DataPointer;
        byte* pCuda = (byte*)cudaOut.DataPointer;
        long byteOffset = (long)offset * dim;
        for (long i = 0; i < keep * dim; i++)
        {
            Assert.Equal(pSrc[byteOffset + i], pCpu[i]);
            Assert.Equal(pSrc[byteOffset + i], pCuda[i]);
        }
    }

    [Fact]
    public void ArgMaxLastDim_Cpu_Vs_Cuda()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }

        // Cover the Kyutai shapes (text head C=8000, depformer C=2048) plus small / tie cases.
        foreach ((int rows, int c) in new[] { (1, 8000), (1, 2048), (4, 2048), (3, 17), (1, 1), (2, 257) })
        {
            using Tensor input = Random(new TensorShape(rows, c), seed: 700 + c);
            using Tensor cpuIdx = new Tensor(new TensorShape(rows), DType.I32);
            using Tensor cudaIdx = new Tensor(new TensorShape(rows), DType.I32);

            IBackend cpu = new CpuBackend();
            cpu.ArgMaxLastDim(cpuIdx, input);
            cpu.Dispose();

            // I32 output: force the device→host sync by reading the int buffer while the context is alive.
            using (CudaBackend cuda = new CudaBackend(0, PtxDir()))
            {
                cuda.ArgMaxLastDim(cudaIdx, input);
                cuda.Sync();
                _ = *(int*)cudaIdx.DataPointer;
            }

            int* a = (int*)cpuIdx.DataPointer;
            int* b = (int*)cudaIdx.DataPointer;
            for (int r = 0; r < rows; r++)
                Assert.True(a[r] == b[r], $"ArgMaxLastDim[{rows}x{c}] row {r}: cpu={a[r]} cuda={b[r]}");
            _output.WriteLine($"ArgMaxLastDim[{rows}x{c}]: matched ({rows} rows).");
        }

        // Explicit tie: lowest index must win on both backends.
        using (Tensor tie = new Tensor(new TensorShape(1, 8), DType.F32))
        {
            float* p = (float*)tie.DataPointer;
            for (int i = 0; i < 8; i++) p[i] = 0.5f;
            p[3] = 0.9f; p[6] = 0.9f;   // tie at index 3 and 6 → expect 3
            using Tensor cpuIdx = new Tensor(new TensorShape(1), DType.I32);
            using Tensor cudaIdx = new Tensor(new TensorShape(1), DType.I32);
            IBackend cpu = new CpuBackend();
            cpu.ArgMaxLastDim(cpuIdx, tie);
            cpu.Dispose();
            using (CudaBackend cuda = new CudaBackend(0, PtxDir()))
            {
                cuda.ArgMaxLastDim(cudaIdx, tie);
                cuda.Sync();
                _ = *(int*)cudaIdx.DataPointer;
            }
            Assert.Equal(3, ((int*)cpuIdx.DataPointer)[0]);
            Assert.Equal(3, ((int*)cudaIdx.DataPointer)[0]);
        }
    }
}
