using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers.Diamond;
using HartsyInference.ModelHandler.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Interactive AR-rollout gen-perf harness for DIAMOND (the world-model FPS metric, not s/clip).
/// Rolls a 4-frame history forward, generating each next frame with the 3-step EDM sampler, and reports
/// per-frame latency + FPS (mean/p50/p99 over a timed window after warmup). Gated on <c>DIAMOND_WEIGHTS</c>
/// (breakout_inner.safetensors) + <c>DIAMOND_PERF=1</c>; <c>PARITY_BACKEND=cuda</c> runs on the GPU.
/// Coherence-checks the rendered rollout (finite, non-flat). Skips cleanly when unset.</summary>
public sealed unsafe class DiamondGenPerfTests
{
    private readonly ITestOutputHelper _out;
    public DiamondGenPerfTests(ITestOutputHelper o) => _out = o;

    private static bool Enabled => Environment.GetEnvironmentVariable("DIAMOND_PERF") == "1"
        && Environment.GetEnvironmentVariable("DIAMOND_WEIGHTS") is { Length: > 0 } p && File.Exists(p);
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
    public void ArRollout_Fps()
    {
        if (!Enabled) return; // gated
        string wp = Environment.GetEnvironmentVariable("DIAMOND_WEIGHTS")!;
        int warm = int.TryParse(Environment.GetEnvironmentVariable("DIAMOND_WARM"), out int wv) ? wv : 5;
        int frames = int.TryParse(Environment.GetEnvironmentVariable("DIAMOND_FRAMES"), out int fv) ? fv : 60;

        SafeTensorsLoader wl = new(); wl.Load(wp);
        using (wl)
        {
            using IBackend backend = MakeBackend();
            DiamondConfig cfg = DiamondConfig.Atari(4);
            DiamondDenoiser den = new(cfg);
            den.LoadWeights(wl.GetAllTensors(), "");
            DiamondSampler sampler = new(den);

            int H = cfg.ImgSize, W = cfg.ImgSize, C = cfg.ImgChannels, K = cfg.NumStepsConditioning;
            long plane = (long)H * W, frameLen = C * plane;
            // Seed a plausible 4-frame history (deterministic gradient-ish content in [-1,1]).
            Tensor obs = new(new TensorShape(1, C * K, H, W), DType.F32);
            float* op = (float*)obs.DataPointer;
            for (long i = 0; i < obs.ElementCount; i++) op[i] = MathF.Sin(i * 0.013f) * 0.6f;
            int[] act = new int[K];
            for (int t = 0; t < K; t++) act[t] = t % cfg.NumActions;

            Random rng = new(1234);
            Tensor xInit = new(new TensorShape(1, C, H, W), DType.F32);

            void FillNoise() { float* xp = (float*)xInit.DataPointer; for (long i = 0; i < xInit.ElementCount; i++) xp[i] = (float)(rng.NextDouble() * 2 - 1); }
            // Roll obs: drop oldest frame, append `next` [1,C,H,W].
            void Roll(Tensor next)
            {
                Buffer.MemoryCopy(op + frameLen, op, (long)(C * (K - 1)) * plane * 4, (long)(C * (K - 1)) * plane * 4);
                Buffer.MemoryCopy((float*)next.DataPointer, op + (long)(C * (K - 1)) * plane, frameLen * 4, frameLen * 4);
            }

            _out.WriteLine($"DIAMOND AR rollout: {H}x{W} {K}-frame ctx, {cfg.NumStepsDenoising}-step EDM, backend={(IsCuda ? "cuda" : "cpu")}");

            // Warmup (JIT/PTX + weight upload + pool warm).
            for (int f = 0; f < warm; f++) { FillNoise(); Tensor nx = sampler.Sample(backend, xInit, obs, act); Roll(nx); nx.Dispose(); }
            backend.Sync();

            double[] ms = new double[frames];
            double lastMean = 0, lastMax = 0;
            Stopwatch sw = new();
            for (int f = 0; f < frames; f++)
            {
                FillNoise();
                sw.Restart();
                Tensor nx = sampler.Sample(backend, xInit, obs, act);
                backend.Sync();
                sw.Stop();
                ms[f] = sw.Elapsed.TotalMilliseconds;
                // Coherence tap on the last generated frame.
                if (f == frames - 1)
                {
                    float* np = (float*)nx.DataPointer; double sum = 0, sqsum = 0; long n = nx.ElementCount;
                    for (long i = 0; i < n; i++) { float v = np[i]; Assert.True(float.IsFinite(v)); sum += v; sqsum += (double)v * v; }
                    lastMean = sum / n; lastMax = Math.Sqrt(sqsum / n - lastMean * lastMean);
                }
                Roll(nx); nx.Dispose();
            }

            double mean = ms.Average();
            double p50 = Percentile(ms, 50), p99 = Percentile(ms, 99), min = ms.Min();
            _out.WriteLine($"per-frame ms: mean={mean:F2} p50={p50:F2} p99={p99:F2} min={min:F2}  =>  FPS mean={1000/mean:F1} p50={1000/p50:F1}");
            _out.WriteLine($"last-frame coherence: mean={lastMean:F4} std={lastMax:F4} (non-flat => std>0)");
            Assert.True(lastMax > 1e-4, $"rendered frame is flat (std={lastMax})");
            xInit.Dispose(); obs.Dispose();
        }
    }

    private static double Percentile(double[] xs, double p)
    {
        double[] s = (double[])xs.Clone(); Array.Sort(s);
        double idx = (p / 100.0) * (s.Length - 1); int lo = (int)Math.Floor(idx); int hi = (int)Math.Ceiling(idx);
        return lo == hi ? s[lo] : s[lo] + (s[hi] - s[lo]) * (idx - lo);
    }
}
