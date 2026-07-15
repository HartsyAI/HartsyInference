using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.ModelHandler.CheckpointConverters;
using HartsyInference.ModelHandler.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Real-weight gen-perf harness for the Matrix-Game 3.0 DiT (Wan2.2 backbone + per-block ActionModule +
/// optional FOV memory frames + optional Plücker camera injection) — the interactive per-DMD-step compute. Loads the
/// Skywork <c>base_model</c>, runs one segment forward with live mouse/keyboard streams, times ms/forward. Carries the
/// campaign's levers: cuDNN fused SDPA (allowF16) + F16 FFN (<c>HARTSY_DIT_F16</c>). Gated on <c>MG3_DIT</c> +
/// <c>MG3_PERF=1</c>; <c>PARITY_BACKEND=cuda</c> on GPU. Coherence = finite/non-flat v (parity is a separate test).</summary>
public sealed unsafe class MatrixGame3GenPerfTests
{
    private readonly ITestOutputHelper _out;
    public MatrixGame3GenPerfTests(ITestOutputHelper o) => _out = o;

    private static bool Enabled => Environment.GetEnvironmentVariable("MG3_PERF") == "1"
        && Environment.GetEnvironmentVariable("MG3_DIT") is { Length: > 0 } p && File.Exists(p);
    private static bool IsCuda => string.Equals(Environment.GetEnvironmentVariable("PARITY_BACKEND"), "cuda", StringComparison.OrdinalIgnoreCase);
    private static int EnvI(string k, int d) => int.TryParse(Environment.GetEnvironmentVariable(k), out int v) ? v : d;

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
        string dit = Environment.GetEnvironmentVariable("MG3_DIT")!;
        int warm = EnvI("MG3_WARM", 3), iters = EnvI("MG3_ITERS", 20);
        int predF = EnvI("MG3_T", 5), H = EnvI("MG3_H", 32), W = EnvI("MG3_W", 32);
        int mem = EnvI("MG3_MEM", 0);                       // FOV memory frames prepended (0 = none)
        bool plucker = Environment.GetEnvironmentVariable("MG3_PLUCKER") == "1";
        int ratio = 4;

        (MatrixGame3CheckpointConverter.ConvertedWeights weights, List<SafeTensorsLoader> loaders) =
            Directory.Exists(dit) ? MatrixGame3CheckpointConverter.LoadFolder(dit) : LoadFile(dit);
        try
        {
            using IBackend backend = MakeBackend();
            MatrixGame3Config cfg = MatrixGame3Config.Base5B with
            {
                ActionBlocks = weights.ActionBlocks.Length > 0 ? weights.ActionBlocks : MatrixGame3Config.Base5B.ActionBlocks,
                // MG3_SIGMA0=1: force sigma_theta=0 (shared-cos rank-2 rope → GPU WanRopeInterleaved) — perf probe only,
                // breaks parity (the real model uses 0.8 per-head rope); isolates the host rope cost.
                SigmaTheta = Environment.GetEnvironmentVariable("MG3_SIGMA0") == "1" ? 0f : MatrixGame3Config.Base5B.SigmaTheta,
            };
            MatrixGame3Transformer model = new(cfg);
            model.LoadWeights(weights.Transformer);

            int c = cfg.InChannels, totF = mem + predF;
            int gh = H / 2, gw = W / 2, s = totF * gh * gw;
            Tensor latent = Rand(1, 1, c, totF, H, W);              // [1, 48, mem+pred, H, W]
            Tensor encoder = Rand(2, 226, cfg.TextDim);            // umT5 context (padded to text_len=512 inside)
            float[] ts = new float[totF];
            for (int f = mem; f < totF; f++) ts[f] = 500f;        // memory frames clean (0), pred frames noised
            int[] ropeIdx = new int[totF]; for (int f = 0; f < totF; f++) ropeIdx[f] = f;
            int rawFrames = (predF - 1) * ratio + 1;
            bool noAction = Environment.GetEnvironmentVariable("MG3_NOACTION") == "1";
            Tensor mouse = Rand(3, rawFrames, 2);
            Tensor? keyboard = noAction ? null : Rand(4, rawFrames, 6);
            Tensor? plk = plucker ? Rand(5, s, cfg.PluckerPatchDim) : null;

            _out.WriteLine($"MG3 DiT: latent[1,{c},{totF}(mem{mem}+pred{predF}),{H},{W}] grid {totF}x{gh}x{gw}={s} tokens, " +
                $"dim={cfg.InnerDim}, {cfg.NumLayers} blocks ({cfg.ActionBlocks?.Length ?? 0} action), plucker={plucker}, " +
                $"FFN=F32(MG3), backend={(IsCuda ? "cuda" : "cpu")}");

            double lastStd = 0;
            for (int f = 0; f < warm; f++)
            { Tensor v = model.Forward(backend, latent, encoder, ts, ropeIdx, mem, predF, mouse, keyboard, plk); v.Dispose(); }
            backend.Sync();

            double[] ms = new double[iters];
            Stopwatch sw = new();
            for (int f = 0; f < iters; f++)
            {
                sw.Restart();
                Tensor v = model.Forward(backend, latent, encoder, ts, ropeIdx, mem, predF, mouse, keyboard, plk);
                backend.Sync();
                sw.Stop();
                ms[f] = sw.Elapsed.TotalMilliseconds;
                if (f == iters - 1)
                {
                    float* vp = (float*)v.DataPointer; double sum = 0, sq = 0; long n = v.ElementCount;
                    for (long i = 0; i < n; i++) { float x = vp[i]; Assert.True(float.IsFinite(x)); sum += x; sq += (double)x * x; }
                    double m = sum / n; lastStd = Math.Sqrt(sq / n - m * m);
                }
                v.Dispose();
            }
            double mean = ms.Average(), p50 = Percentile(ms, 50), min = ms.Min();
            _out.WriteLine($"per-forward ms: mean={mean:F2} p50={p50:F2} min={min:F2}  =>  {1000 / mean:F1} fwd/s;  output std={lastStd:F4}");
            Assert.True(lastStd > 1e-4, $"velocity flat (std={lastStd})");
            latent.Dispose(); encoder.Dispose(); mouse.Dispose(); keyboard?.Dispose(); plk?.Dispose();
        }
        finally { foreach (SafeTensorsLoader l in loaders) l.Dispose(); }
    }

    private static (MatrixGame3CheckpointConverter.ConvertedWeights, List<SafeTensorsLoader>) LoadFile(string path)
    {
        SafeTensorsLoader loader = new(); loader.Load(path);
        return (MatrixGame3CheckpointConverter.Convert(loader.GetAllTensors()), new List<SafeTensorsLoader> { loader });
    }

    private static double Percentile(double[] xs, double p)
    {
        double[] srt = (double[])xs.Clone(); Array.Sort(srt);
        double idx = (p / 100.0) * (srt.Length - 1); int lo = (int)Math.Floor(idx), hi = (int)Math.Ceiling(idx);
        return lo == hi ? srt[lo] : srt[lo] + (srt[hi] - srt[lo]) * (idx - lo);
    }
}
