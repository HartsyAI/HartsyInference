using System.Diagnostics;
using Xunit;
using Xunit.Abstractions;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.ModelHandler.SafeTensors;

namespace HartsyInference.Diffusion.Tests;

/// <summary>GPU micro-benchmark for the ACE-Step v1.5 turbo DiT after the GPU-residency rewrite. Loads only the
/// <c>decoder.*</c> keys of the real turbo checkpoint (BF16, mmap-borrowed — no big host copy), preloads them
/// resident, and times <see cref="AceStep15Dit.Forward"/> at a realistic 10 s sequence (250 latent frames → 125
/// tokens). Isolates the optimized DiT (no VAE / text encoder), so host RAM stays low. Gated behind
/// <c>HARTSY_ACE15_GPU_BENCH=1</c> and the checkpoint path so a normal run skips it.
/// <code>HARTSY_ACE15_GPU_BENCH=1 dotnet test tests/HartsyInference.Diffusion.Tests --filter AceStep15DitGpuBench</code>
/// Env: <c>HARTSY_DIT_F16</c> / <c>HARTSY_DIT_GRAPH</c> toggle the F16 / CUDA-graph paths for A/B timing.</summary>
public unsafe class AceStep15DitGpuBench
{
    private const string TurboPath = "/home/kalebbroo/Desktop/Projects/SwarmUI/Models/audio/music/AceStep/acestep-v15-turbo.safetensors";
    private readonly ITestOutputHelper _output;
    public AceStep15DitGpuBench(ITestOutputHelper output) => _output = output;

    [Fact]
    public void Turbo_Dit_Forward_Timed()
    {
        if (Environment.GetEnvironmentVariable("HARTSY_ACE15_GPU_BENCH") != "1")
        { _output.WriteLine("SKIPPED: set HARTSY_ACE15_GPU_BENCH=1 to run."); return; }
        if (!File.Exists(TurboPath))
        { _output.WriteLine($"SKIPPED: turbo checkpoint not found at {TurboPath}."); return; }

        HartsyInference.Cuda.CudaBackend cuda;
        try { cuda = new HartsyInference.Cuda.CudaBackend(0, Path.Combine(AppContext.BaseDirectory, "Ptx")); }
        catch (Exception ex) { _output.WriteLine($"SKIPPED: no usable CUDA device ({ex.Message})."); return; }

        using (cuda)
        {
            AceStep15Config cfg = AceStep15Config.Turbo;
            using SafeTensorsLoader loader = new();
            loader.Load(TurboPath);
            System.Collections.Generic.Dictionary<string, Tensor> all = loader.GetAllTensors();
            System.Collections.Generic.Dictionary<string, Tensor> dec = new(StringComparer.Ordinal);
            foreach (System.Collections.Generic.KeyValuePair<string, Tensor> kv in all)
                if (kv.Key.StartsWith("decoder.", StringComparison.Ordinal)) dec[kv.Key] = kv.Value;
            _output.WriteLine($"loaded {dec.Count} decoder tensors (bf16 borrowed) from {Path.GetFileName(TurboPath)}");

            using AceStep15Dit dit = new(cfg);
            dit.LoadWeights(dec);

            const double seconds = 10.0;
            int frames = cfg.FrameCount(seconds);          // 250 latent frames at 25 Hz
            int condLen = 256;
            Tensor noisy = Rand(new TensorShape(1, frames, cfg.LatentChannels), 1);
            Tensor context = Rand(new TensorShape(1, frames, cfg.InChannels - cfg.LatentChannels), 2);
            Tensor conditions = Rand(new TensorShape(1, condLen, cfg.HiddenSize), 3);
            float[] ts = AceStep15Config.GetTimesteps(cfg.FlowShift);

            cuda.PreloadWeights(dit.EnumerateWeights());
            bool f16 = Environment.GetEnvironmentVariable("HARTSY_DIT_F16") == "1";
            bool graph = Environment.GetEnvironmentVariable("HARTSY_DIT_GRAPH") == "1";
            _output.WriteLine($"config: {frames} frames ({frames / cfg.PatchSize} tokens), condLen {condLen}, F16={f16}, GRAPH={graph}");

            // Warmup (JIT kernels, allocate workspaces, capture graph if enabled).
            for (int i = 0; i < 3; i++)
            { Tensor v = dit.Forward(cuda, noisy, context, conditions, ts[0], ts[0]); cuda.Sync(); v.Dispose(); }

            // Timed: 8 turbo steps.
            Stopwatch sw = Stopwatch.StartNew();
            float last = 0f;
            Tensor? vLast = null;
            for (int i = 0; i < ts.Length; i++)
            {
                vLast?.Dispose();
                vLast = dit.Forward(cuda, noisy, context, conditions, ts[i], ts[i]);
            }
            cuda.Sync();
            sw.Stop();
            double perStep = sw.Elapsed.TotalMilliseconds / ts.Length;
            _output.WriteLine($"DiT forward: {perStep:F1} ms/step  ->  {perStep * ts.Length / 1000.0:F2}s for {ts.Length}-step turbo ({seconds:F0}s audio)");

            // Finiteness check on the last velocity.
            long n = vLast!.Shape.ElementCount;
            float* p = (float*)vLast.DataPointer;   // one D2H at the very end (pipeline reads velocity host-side too)
            bool finite = true;
            for (long i = 0; i < n; i++) { float a = p[i]; if (!float.IsFinite(a)) { finite = false; break; } last = MathF.Max(last, MathF.Abs(a)); }
            _output.WriteLine($"velocity finite={finite}, maxAbs={last:E3}");
            Assert.True(finite, "DiT produced non-finite velocity on CUDA");

            vLast.Dispose(); noisy.Dispose(); context.Dispose(); conditions.Dispose();
        }
    }

    private static Tensor Rand(TensorShape shape, int seed)
    {
        Tensor t = new(shape, DType.F32);
        float* p = (float*)t.DataPointer;
        long n = shape.ElementCount;
        uint s = (uint)(seed * 2654435761u + 1u);
        for (long i = 0; i < n; i++) { s = s * 1664525u + 1013904223u; p[i] = ((s >> 8) / (float)(1 << 24) - 0.5f) * 2.0f; }
        return t;
    }
}
