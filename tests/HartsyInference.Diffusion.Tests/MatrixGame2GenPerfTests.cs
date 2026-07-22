using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.ModelAssets.CheckpointConverters;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Real-weight gen-perf harness for the Matrix-Game 2.0 DiT (Wan-backbone + per-block ActionModule) — the
/// interactive per-DDIM-step compute. Loads the Skywork distilled checkpoint, runs the transformer forward over one
/// 3-frame latent block with live mouse/keyboard action streams, and times ms/forward. The ActionModule carries the
/// cuDNN-fused-SDPA + residency ports. Gated on <c>MG2_DIT</c> + <c>MG2_PERF=1</c>; <c>PARITY_BACKEND=cuda</c> on GPU.
/// Coherence = finite v-prediction (numeric parity vs reference is a separate gated test).</summary>
public sealed unsafe class MatrixGame2GenPerfTests
{
    private readonly ITestOutputHelper _out;
    public MatrixGame2GenPerfTests(ITestOutputHelper o) => _out = o;

    private static bool Enabled => Environment.GetEnvironmentVariable("MG2_PERF") == "1"
        && Environment.GetEnvironmentVariable("MG2_DIT") is { Length: > 0 } p && File.Exists(p);
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

    private static Tensor Rand(int seed, params int[] shape)
    {
        long[] s = new long[shape.Length];
        for (int i = 0; i < shape.Length; i++) s[i] = shape[i];
        Tensor t = new(new TensorShape(s), DType.F32);
        Random r = new(seed);
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++) p[i] = (float)(r.NextDouble() * 2 - 1);
        return t;
    }

    [Fact]
    public void DitForward_Fps()
    {
        if (!Enabled) return; // gated
        string dit = Environment.GetEnvironmentVariable("MG2_DIT")!;
        int warm = int.TryParse(Environment.GetEnvironmentVariable("MG2_WARM"), out int wv) ? wv : 3;
        int iters = int.TryParse(Environment.GetEnvironmentVariable("MG2_ITERS"), out int iv) ? iv : 20;
        int gtF = int.TryParse(Environment.GetEnvironmentVariable("MG2_T"), out int tv) ? tv : 3;
        int hw = int.TryParse(Environment.GetEnvironmentVariable("MG2_HW"), out int hv) ? hv : 32;

        (MatrixGame3CheckpointConverter.ConvertedWeights weights, HartsyInference.ModelAssets.SafeTensors.SafeTensorsLoader loader) = MatrixGame2CheckpointConverter.LoadAndConvert(dit);
        using (loader)
        {
            using IBackend backend = MakeBackend();
            MatrixGame2Config cfg = MatrixGame2Config.Universal;
            MatrixGame2Transformer model = new(cfg);
            model.LoadWeights(weights.Transformer);

            int c = cfg.InChannels, ratio = 4;
            Tensor latent = Rand(1, 1, c, gtF, hw, hw);                 // [1, 36, T, H, W]
            Tensor clip = Rand(2, cfg.ClipContextTokens, cfg.ClipContextDim);   // [257, 1280]
            float[] ts = new float[gtF]; for (int f = 0; f < gtF; f++) ts[f] = 500f;
            int[] ropeIdx = new int[gtF]; for (int f = 0; f < gtF; f++) ropeIdx[f] = f;
            int rawFrames = gtF * ratio;
            Tensor mouse = Rand(3, rawFrames, 2);
            Tensor? keyboard = Environment.GetEnvironmentVariable("MG2_NOACTION") == "1" ? null : Rand(4, rawFrames, cfg.KeyboardDim);

            _out.WriteLine($"MG2 DiT: latent[1,{c},{gtF},{hw},{hw}] grid gt={gtF} gh={hw / 2} gw={hw / 2}, dim={cfg.InnerDim}, {cfg.NumLayers} blocks, action-blocks + cuDNN-SDPA, backend={(IsCuda ? "cuda" : "cpu")}");

            double lastStd = 0;
            for (int f = 0; f < warm; f++) { Tensor v = model.Forward(backend, latent, clip, ts, ropeIdx, gtF, mouse, keyboard); v.Dispose(); }
            backend.Sync();

            double[] ms = new double[iters];
            Stopwatch sw = new();
            for (int f = 0; f < iters; f++)
            {
                sw.Restart();
                Tensor v = model.Forward(backend, latent, clip, ts, ropeIdx, gtF, mouse, keyboard);
                backend.Sync();
                sw.Stop();
                ms[f] = sw.Elapsed.TotalMilliseconds;
                if (f == iters - 1)
                {
                    float* vp = (float*)v.DataPointer; double s = 0, sq = 0; long n = v.ElementCount;
                    for (long i = 0; i < n; i++) { float x = vp[i]; Assert.True(float.IsFinite(x)); s += x; sq += (double)x * x; }
                    double m = s / n; lastStd = Math.Sqrt(sq / n - m * m);
                }
                v.Dispose();
            }
            double mean = ms.Average(), p50 = Percentile(ms, 50), min = ms.Min();
            _out.WriteLine($"per-forward ms: mean={mean:F2} p50={p50:F2} min={min:F2}  =>  {1000 / mean:F1} fwd/s;  output std={lastStd:F4} (finite, non-flat)");
            Assert.True(lastStd > 1e-4, $"velocity flat (std={lastStd})");
            latent.Dispose(); clip.Dispose(); mouse.Dispose(); keyboard?.Dispose();
        }
    }

    private static double Percentile(double[] xs, double p)
    {
        double[] s = (double[])xs.Clone(); Array.Sort(s);
        double idx = (p / 100.0) * (s.Length - 1); int lo = (int)Math.Floor(idx), hi = (int)Math.Ceiling(idx);
        return lo == hi ? s[lo] : s[lo] + (s[hi] - s[lo]) * (idx - lo);
    }
}
