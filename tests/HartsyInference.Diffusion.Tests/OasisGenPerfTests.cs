using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.ModelHandler.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Interactive gen-perf harness for the Oasis-500m DiT (the per-frame compute of the AR world-model
/// loop: one DDIM step = one DiT forward over the sliding window). Times repeated forwards on a fixed
/// reference latent and reports ms/forward + effective FPS (10 DDIM steps/frame). Doubles as a parity guard:
/// asserts corr vs the reference <c>dit_v</c> stays &gt;0.9999 so residency ports can't silently break numerics.
/// Gated on <c>OASIS_DIT</c>/<c>OASIS_REF</c> + <c>OASIS_PERF=1</c>; <c>PARITY_BACKEND=cuda</c> runs on GPU.</summary>
public sealed unsafe class OasisGenPerfTests
{
    private readonly ITestOutputHelper _out;
    public OasisGenPerfTests(ITestOutputHelper o) => _out = o;

    private static bool Enabled => Environment.GetEnvironmentVariable("OASIS_PERF") == "1"
        && Environment.GetEnvironmentVariable("OASIS_DIT") is { Length: > 0 } d && File.Exists(d)
        && Environment.GetEnvironmentVariable("OASIS_REF") is { Length: > 0 } r && File.Exists(r);
    private static bool IsCuda => string.Equals(Environment.GetEnvironmentVariable("PARITY_BACKEND"), "cuda", StringComparison.OrdinalIgnoreCase);

    private static IBackend MakeBackend()
    {
        if (!IsCuda) return new CpuBackend();
        string? d = AppContext.BaseDirectory;
        for (int i = 0; i < 8 && d is not null; i++, d = Path.GetDirectoryName(d))
        {
            string cand = Path.Combine(d, "src", "HartsyInference.Cuda", "Ptx");
            if (Directory.Exists(cand)) return new HartsyInference.Cuda.CudaBackend(0, cand);
        }
        return new HartsyInference.Cuda.CudaBackend(0, Path.Combine(AppContext.BaseDirectory, "Ptx"));
    }

    [Fact]
    public void DitForward_Fps()
    {
        if (!Enabled) return; // gated
        string dp = Environment.GetEnvironmentVariable("OASIS_DIT")!;
        string rp = Environment.GetEnvironmentVariable("OASIS_REF")!;
        int warm = int.TryParse(Environment.GetEnvironmentVariable("OASIS_WARM"), out int wv) ? wv : 4;
        int iters = int.TryParse(Environment.GetEnvironmentVariable("OASIS_ITERS"), out int iv) ? iv : 30;

        SafeTensorsLoader dit = new(); dit.Load(dp);
        SafeTensorsLoader refIo = new(); refIo.Load(rp);
        using (dit) using (refIo)
        {
            using IBackend backend = MakeBackend();
            OasisDit model = new(OasisDitConfig.S2); model.LoadWeights(dit.GetAllTensors());

            Tensor xB = refIo.GetTensor("dit_x");        // [1,T,16,18,32]
            Tensor tB = refIo.GetTensor("dit_t");        // [1,T]
            Tensor condB = refIo.GetTensor("dit_cond");  // [1,T,25]
            int T = (int)xB.Shape[1];
            long per = (long)16 * 18 * 32;
            Tensor latents = new(new TensorShape(T, 16, 18, 32), DType.F32);
            new ReadOnlySpan<float>((float*)xB.DataPointer, (int)(T * per)).CopyTo(new Span<float>((float*)latents.DataPointer, (int)(T * per)));
            int[] noiseIdx = new int[T];
            float* tp = (float*)tB.DataPointer;
            for (int f = 0; f < T; f++) noiseIdx[f] = (int)MathF.Round(tp[f]);
            Tensor actions = new(new TensorShape(T, 25), DType.F32);
            new ReadOnlySpan<float>((float*)condB.DataPointer, T * 25).CopyTo(new Span<float>((float*)actions.DataPointer, T * 25));
            Tensor refV = refIo.GetTensor("dit_v");

            _out.WriteLine($"OASIS DiT-S/2 forward: T={T} window, grid 18x32 (576 tokens), dim 1024, 16 blocks, backend={(IsCuda ? "cuda" : "cpu")}");

            double corr = 0;
            for (int f = 0; f < warm; f++) { Tensor v = model.Forward(backend, latents, noiseIdx, actions); if (f == 0) corr = Corr(v, refV); v.Dispose(); }
            backend.Sync();

            double[] ms = new double[iters];
            double lastCorr = corr;
            Stopwatch sw = new();
            for (int f = 0; f < iters; f++)
            {
                sw.Restart();
                Tensor v = model.Forward(backend, latents, noiseIdx, actions);
                backend.Sync();
                sw.Stop();
                ms[f] = sw.Elapsed.TotalMilliseconds;
                if (f == iters - 1) lastCorr = Corr(v, refV);   // graph-REPLAY output when HARTSY_DIT_GRAPH=1
                v.Dispose();
            }
            double mean = ms.Average(), p50 = Percentile(ms, 50), p99 = Percentile(ms, 99), min = ms.Min();
            bool graph = Environment.GetEnvironmentVariable("HARTSY_DIT_GRAPH") == "1";
            _out.WriteLine($"per-forward ms: mean={mean:F2} p50={p50:F2} p99={p99:F2} min={min:F2}  (graph={graph})");
            _out.WriteLine($"=> 1 DDIM step/forward: {1000 / mean:F1} fwd/s;  10-step frame: {1000 / (10 * mean):F2} FPS;  parity corr warm={corr:F8} last={lastCorr:F8}");
            Assert.True(corr > 0.9999, $"DiT parity broke: corr={corr}");
            Assert.True(lastCorr > 0.9999, $"DiT replay parity broke: corr={lastCorr}");
            latents.Dispose(); actions.Dispose();
        }
    }

    private static double Corr(Tensor a, Tensor b)
    {
        long n = a.ElementCount; float* ap = (float*)a.DataPointer, bp = (float*)b.DataPointer;
        double ma = 0, mb = 0; for (long i = 0; i < n; i++) { ma += ap[i]; mb += bp[i]; } ma /= n; mb /= n;
        double sab = 0, saa = 0, sbb = 0;
        for (long i = 0; i < n; i++) { double da = ap[i] - ma, db = bp[i] - mb; sab += da * db; saa += da * da; sbb += db * db; }
        return sab / (Math.Sqrt(saa * sbb) + 1e-12);
    }

    private static double Percentile(double[] xs, double p)
    {
        double[] s = (double[])xs.Clone(); Array.Sort(s);
        double idx = (p / 100.0) * (s.Length - 1); int lo = (int)Math.Floor(idx), hi = (int)Math.Ceiling(idx);
        return lo == hi ? s[lo] : s[lo] + (s[hi] - s[lo]) * (idx - lo);
    }
}
