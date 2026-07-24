using System.Diagnostics;
using System.Linq;
using HartsyInference.Core.Tensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>Stage-3 gates for the W8A8 Linear integration (<c>CudaBackend.EnableW8A8</c>): (1) parity —
/// <c>Linear</c> with W8A8 on vs off at a DiT shape stays in the int8 error regime; (2) the weight cache
/// runs the host quant exactly once (second call reuses the persistent buffer); (3) wall-time A/B of the
/// integrated path vs the F16 GEMM path at the Chroma proj_mlp shape. Run explicitly:
///   CUDA_VISIBLE_DEVICES=1 dotnet test --filter "FullyQualifiedName~W8A8LinearTests"
/// (Category=W8A8Bench, excluded from sweeps; CVD=1 = the 3060, the IMMA target class.)</summary>
[Collection("CudaSerial")]
[Trait("Category", "W8A8Bench")]
public sealed unsafe class W8A8LinearTests
{
    private readonly ITestOutputHelper _output;
    public W8A8LinearTests(ITestOutputHelper output) => _output = output;

    private static string PtxDir()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(dir))
            dir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        return dir;
    }

    private static Tensor Rand(int seed, DType dt, params long[] shape)
    {
        Tensor f32 = new Tensor(new TensorShape(shape), DType.F32);
        Random rng = new Random(seed);
        float* p = (float*)f32.DataPointer;
        for (long i = 0; i < f32.ElementCount; i++) p[i] = (float)(rng.NextDouble() * 0.08 - 0.04);
        if (dt == DType.F32) return f32;
        Tensor cast = f32.CastTo(dt);
        f32.Dispose();
        return cast;
    }

    private static double RelL2(Tensor a, Tensor b)
    {
        Tensor a32 = a.DType == DType.F32 ? a : a.CastTo(DType.F32);
        Tensor b32 = b.DType == DType.F32 ? b : b.CastTo(DType.F32);
        float* pa = (float*)a32.DataPointer;
        float* pb = (float*)b32.DataPointer;
        double num = 0, den = 0;
        for (long i = 0; i < a32.ElementCount; i++)
        {
            double d = pa[i] - pb[i];
            num += d * d;
            den += (double)pb[i] * pb[i];
        }
        if (!ReferenceEquals(a32, a)) a32.Dispose();
        if (!ReferenceEquals(b32, b)) b32.Dispose();
        return Math.Sqrt(num / Math.Max(den, 1e-30));
    }

    [Theory]
    [InlineData(true)]   // F16 activations (DiT F16 loop)
    [InlineData(false)]  // F32 activations (classic path)
    public void Linear_W8A8_MatchesBaseline(bool f16Act)
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        const int M = 512, N = 1024, K = 768;

        using CudaBackend cuda = new CudaBackend(0, PtxDir());
        if (!cuda.Kernels!.HasW8A8Kernels) { _output.WriteLine("SKIPPED: w8a8.ptx missing"); return; }

        DType act = f16Act ? DType.F16 : DType.F32;
        using Tensor input = Rand(1, act, 1, M, K);
        using Tensor weight = Rand(2, DType.F16, N, K);
        using Tensor bias = Rand(3, DType.F32, N);
        using Tensor outBase = new Tensor(new TensorShape(1, M, N), act);
        using Tensor outW8 = new Tensor(new TensorShape(1, M, N), act);

        cuda.EnableW8A8 = false;
        cuda.Linear(outBase, input, weight, bias);
        cuda.Sync();
        _ = outBase.DataPointer; // drain to host

        cuda.EnableW8A8 = true;
        cuda.Linear(outW8, input, weight, bias);
        cuda.Sync();
        _ = outW8.DataPointer;

        double rel = RelL2(outW8, outBase);
        _output.WriteLine($"[{(f16Act ? "F16" : "F32")} act] Linear W8A8 vs baseline relL2 = {rel:e2}");
        Assert.True(rel < 3e-2, $"W8A8 Linear diverged: relL2={rel}");
    }

    /// <summary>End-to-end integration gate for SmoothQuant (W8A8_HANDOFF.md item 1, offline-gate-confirmed
    /// 2026-07-24 on real Kandinsky5 layers): exercises the FULL production path — <c>SetW8A8SmoothingScale</c>
    /// → weight-side fold in <c>QuantizeWeightForW8A8</c> → activation-side <c>invScale</c> in the
    /// w8a8_quant_rowwise kernel → <c>Linear</c> — not just the kernel in isolation. Builds a synthetic
    /// activation with a deliberate per-channel outlier structure (the pathology SmoothQuant targets) and
    /// confirms setting a matching smoothing scale reduces relL2 vs the EXACT F32 reference, not just vs
    /// the F16 baseline (so the comparison isn't contaminated by F16 rounding).</summary>
    [Fact]
    public void Linear_W8A8_SmoothingScale_ReducesErrorVsExactReference()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        const int M = 512, N = 256, K = 512;
        const int OutlierChannels = 8;

        using CudaBackend cuda = new CudaBackend(0, PtxDir());
        if (!cuda.Kernels!.HasW8A8Kernels) { _output.WriteLine("SKIPPED: w8a8.ptx missing"); return; }

        Random rng = new Random(99);
        float[] inputF32 = new float[(long)M * K];
        for (int i = 0; i < inputF32.Length; i++) inputF32[i] = (float)(rng.NextDouble() * 0.06 - 0.03);
        // A handful of channels get a much larger, per-row-varying magnitude — the classic activation-
        // outlier pathology (a few fixed channels dominate every row's absmax, starving the other K-8
        // channels' int8 precision under a single shared per-row scale).
        int[] outlierIdx = Enumerable.Range(0, K).OrderBy(_ => rng.Next()).Take(OutlierChannels).ToArray();
        for (int r = 0; r < M; r++)
            foreach (int c in outlierIdx)
                inputF32[r * K + c] = (float)((rng.NextDouble() * 2 - 1) * (8.0 + 4.0 * rng.NextDouble()));

        float[] weightF32 = new float[(long)N * K];
        for (int i = 0; i < weightF32.Length; i++) weightF32[i] = (float)(rng.NextDouble() * 0.08 - 0.04);

        using Tensor input = new Tensor(new TensorShape(1, M, K), DType.F32);
        fixed (float* p = inputF32) Buffer.MemoryCopy(p, (void*)input.DataPointer, inputF32.Length * 4, inputF32.Length * 4);
        using Tensor inputF16 = input.CastTo(DType.F16);

        using Tensor w32 = new Tensor(new TensorShape(N, K), DType.F32);
        fixed (float* p = weightF32) Buffer.MemoryCopy(p, (void*)w32.DataPointer, weightF32.Length * 4, weightF32.Length * 4);
        using Tensor weight = w32.CastTo(DType.F16);

        // Exact F32 reference (CPU), no bias.
        float[] reference = new float[(long)M * N];
        System.Threading.Tasks.Parallel.For(0, M, mi =>
        {
            for (int ni = 0; ni < N; ni++)
            {
                double sum = 0;
                for (int ki = 0; ki < K; ki++) sum += (double)inputF32[mi * K + ki] * weightF32[ni * K + ki];
                reference[mi * N + ni] = (float)sum;
            }
        });

        double RelL2VsRef(Tensor outT)
        {
            using Tensor o32 = outT.DType == DType.F32 ? outT : outT.CastTo(DType.F32);
            float* p = (float*)o32.DataPointer;
            double num = 0, den = 0;
            for (long i = 0; i < reference.Length; i++)
            {
                double d = p[i] - reference[i];
                num += d * d;
                den += (double)reference[i] * reference[i];
            }
            return Math.Sqrt(num / Math.Max(den, 1e-30));
        }

        using Tensor outUnsmoothed = new Tensor(new TensorShape(1, M, N), DType.F16);
        cuda.EnableW8A8 = true;
        cuda.Linear(outUnsmoothed, inputF16, weight, null);
        cuda.Sync();
        double relUnsmoothed = RelL2VsRef(outUnsmoothed);

        // Calibrate s from THIS exact activation/weight (deterministic synthetic data, no timestep drift
        // to worry about here) using the same alpha=0.7 the real-checkpoint offline gate found best.
        float[] actMax = new float[K];
        for (int r = 0; r < M; r++)
            for (int c = 0; c < K; c++)
                actMax[c] = MathF.Max(actMax[c], MathF.Abs(inputF32[r * K + c]));
        float[] wMax = new float[K];
        for (int ni = 0; ni < N; ni++)
            for (int c = 0; c < K; c++)
                wMax[c] = MathF.Max(wMax[c], MathF.Abs(weightF32[ni * K + c]));
        const double alpha = 0.7;
        float[] s = new float[K];
        for (int c = 0; c < K; c++)
        {
            double sv = actMax[c] > 0 && wMax[c] > 0 ? Math.Pow(actMax[c], alpha) / Math.Pow(wMax[c], 1.0 - alpha) : 1.0;
            s[c] = (float)Math.Clamp(sv, 1e-3, 1e3);
        }
        cuda.SetW8A8SmoothingScale(weight, s);

        using Tensor outSmoothed = new Tensor(new TensorShape(1, M, N), DType.F16);
        cuda.Linear(outSmoothed, inputF16, weight, null);
        cuda.Sync();
        double relSmoothed = RelL2VsRef(outSmoothed);

        _output.WriteLine($"unsmoothed relL2 vs exact F32 ref: {relUnsmoothed:e3}");
        _output.WriteLine($"smoothed (alpha={alpha:F1})   relL2 vs exact F32 ref: {relSmoothed:e3}");
        Assert.True(relSmoothed < relUnsmoothed,
            $"SmoothQuant should reduce error on an outlier-channel activation: smoothed={relSmoothed:e3} >= unsmoothed={relUnsmoothed:e3}");
    }

    [Fact]
    public void Linear_W8A8_WeightCacheReused_And_Perf()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        const int M = 4608, N = 12288, K = 3072;   // Chroma proj_mlp class

        using CudaBackend cuda = new CudaBackend(0, PtxDir());
        if (!cuda.Kernels!.HasW8A8Kernels) { _output.WriteLine("SKIPPED: w8a8.ptx missing"); return; }

        using Tensor input = Rand(10, DType.F16, 1, M, K);
        using Tensor weight = Rand(11, DType.F16, N, K);
        using Tensor bias = Rand(12, DType.F32, N);
        using Tensor outT = new Tensor(new TensorShape(1, M, N), DType.F16);

        // First W8A8 call pays the one-time host quant; the second must not (cache hit) — assert via timing
        // ratio (host quant of 37M params dwarfs a 4.3 ms GEMM chain).
        cuda.EnableW8A8 = true;
        Stopwatch sw = Stopwatch.StartNew();
        cuda.Linear(outT, input, weight, bias);
        cuda.Sync();
        sw.Stop();
        double firstMs = sw.Elapsed.TotalMilliseconds;
        sw.Restart();
        cuda.Linear(outT, input, weight, bias);
        cuda.Sync();
        sw.Stop();
        double secondMs = sw.Elapsed.TotalMilliseconds;
        _output.WriteLine($"first (quant+upload+chain): {firstMs:F1} ms | second (cache hit): {secondMs:F2} ms");
        Assert.True(secondMs < firstMs, "second call should reuse the cached int8 weight");

        // Interleaved A/B: integrated W8A8 Linear vs baseline F16 Linear.
        const int Trials = 8;
        double[] w8Ms = new double[Trials];
        double[] f16Ms = new double[Trials];
        cuda.EnableW8A8 = false;
        cuda.Linear(outT, input, weight, bias);   // warm the F16 path (weight upload)
        cuda.Sync();
        for (int t = 0; t < Trials; t++)
        {
            cuda.EnableW8A8 = true;
            sw.Restart(); cuda.Linear(outT, input, weight, bias); cuda.Sync(); sw.Stop();
            w8Ms[t] = sw.Elapsed.TotalMilliseconds;
            cuda.EnableW8A8 = false;
            sw.Restart(); cuda.Linear(outT, input, weight, bias); cuda.Sync(); sw.Stop();
            f16Ms[t] = sw.Elapsed.TotalMilliseconds;
        }
        double MeanOf(double[] xs) { double s = 0; foreach (double v in xs) s += v; return s / xs.Length; }
        double StdOf(double[] xs, double mean) { double s = 0; foreach (double v in xs) s += (v - mean) * (v - mean); return Math.Sqrt(s / (xs.Length - 1)); }
        double mW = MeanOf(w8Ms), mF = MeanOf(f16Ms);
        double sW = StdOf(w8Ms, mW), sF = StdOf(f16Ms, mF);
        double welch = (mF - mW) / Math.Sqrt(sW * sW / Trials + sF * sF / Trials);
        _output.WriteLine($"[{M}x{N}x{K}] W8A8 Linear: {mW:F3} ± {sW:F3} ms | F16 Linear: {mF:F3} ± {sF:F3} ms | " +
            $"speedup {mF / mW:F3}x | Welch t={welch:F1}");
    }
}
