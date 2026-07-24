using Xunit;
using Xunit.Abstractions;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Schedulers;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Tests.Common;
using HartsyInference.Video.Pipelines;

namespace HartsyInference.Video.Tests;

/// <summary>W8A8 stage 4a/4b measure-first gate (docs/Checklists/W8A8_HANDOFF.md "What's left" #1-2) on
/// Kandinsky-5.0 T2V Lite (2B, real checkpoint): does the current per-row/per-channel INT8 quant (no
/// SmoothQuant channel migration) stay inside the near-lossless relL2 budget on REAL trained weights +
/// REAL text conditioning, and does the error drift across the flow-match timestep schedule (which
/// would motivate NDTC-style calibration)? The existing W8A8 accuracy numbers (relL2 3-5.5e-3) are all
/// on synthetic Gaussian activations/weights — this is the first check against a real checkpoint's
/// actual channel statistics, per the house "measure before building" rule (H2/H3/R4 precedent).
/// Latent input per probe step is fresh unit-ish Gaussian noise, not a real trajectory sample — a
/// reasonable proxy for flow-match latents (which interpolate data/noise) since this test isolates
/// weight/activation quantization error, not trajectory fidelity. Run explicitly:
///   CUDA_VISIBLE_DEVICES=1 dotnet test --filter "FullyQualifiedName~Kandinsky5W8A8MeasurementTests"
/// (Category=W8A8Bench, excluded from sweeps; CVD=1 = the 3060, the IMMA target class.)</summary>
[Trait("Category", "W8A8Bench")]
public sealed unsafe class Kandinsky5W8A8MeasurementTests
{
    private readonly ITestOutputHelper _output;
    public Kandinsky5W8A8MeasurementTests(ITestOutputHelper output) => _output = output;

    private static string T2VDir => Environment.GetEnvironmentVariable("KANDINSKY5_T2V_DIR")
        ?? Path.Combine(TestPaths.ModelsDir, "Stable-Diffusion", "Kandinsky5", "Kandinsky-5.0-T2V-Lite-sft-5s-Diffusers");

    [Fact]
    public void W8A8_RelL2_Vs_Timestep_RealWeights()
    {
        string transformerDir = Path.Combine(T2VDir, "transformer");
        if (!Directory.Exists(transformerDir)) { _output.WriteLine($"SKIPPED: T2V transformer dir not found: {transformerDir} (set KANDINSKY5_T2V_DIR)."); return; }
        if (!File.Exists(TestPaths.Kandinsky5.PromptQwenEmbeds) || !File.Exists(TestPaths.Kandinsky5.PromptClipPooled))
        { _output.WriteLine("SKIPPED: pre-computed Qwen/CLIP embeddings missing (see dump_kandinsky5_embeddings.py)."); return; }
        string ptxDir = Path.Combine(Path.GetDirectoryName(typeof(Kandinsky5W8A8MeasurementTests).Assembly.Location)!, "Ptx");
        if (!Directory.Exists(ptxDir)) { _output.WriteLine($"SKIPPED: PTX dir not found: {ptxDir}"); return; }
        if (!File.Exists(Path.Combine(ptxDir, "w8a8.ptx"))) { _output.WriteLine("SKIPPED: w8a8.ptx missing"); return; }
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }

        (Kandinsky5CheckpointConverter.ConvertedWeights converted, List<SafeTensorsLoader> loaders) =
            Kandinsky5CheckpointConverter.LoadDiffusersFolder(transformerDir);
        try
        {
            Dictionary<string, Tensor> tw = new(converted.Transformer.Count);
            foreach ((string k, Tensor v) in converted.Transformer)
                tw[k] = v.DType == DType.BF16 ? v.CastTo(DType.F16) : v;
            Kandinsky5Config config = Kandinsky5Config.VideoLite2B;
            using Kandinsky5Transformer transformer = new(config);
            transformer.LoadWeights(tw);

            using CudaBackend backend = new(deviceOrdinal: 0, ptxDir: ptxDir);
            _output.WriteLine($"Device: {backend.Capabilities.Name} (CUDA_VISIBLE_DEVICES=" +
                $"{Environment.GetEnvironmentVariable("CUDA_VISIBLE_DEVICES") ?? "unset"})");

            using Tensor qwen = LoadF32Tensor(TestPaths.Kandinsky5.PromptQwenEmbeds, config.InTextDim);
            using Tensor clip = LoadPooled(TestPaths.Kandinsky5.PromptClipPooled, config.InTextDim2);

            const int width = 512, height = 512, numFrames = 25;
            int tLat = (numFrames - 1) / 4 + 1, hLat = height / 8, wLat = width / 8;
            int latCh = config.InVisualDim;
            TensorShape latentShape = new TensorShape([1L, latCh, tLat, hLat, wLat]);
            TensorShape maskShape = new TensorShape([1L, 1, tLat, hLat, wLat]);

            const int steps = 30;
            FlowMatchEulerDiscreteScheduler scheduler = new(5.0f);
            scheduler.SetTimesteps(steps);
            ReadOnlySpan<float> timesteps = scheduler.Timesteps;
            int[] probeSteps = [0, steps / 4, steps / 2, 3 * steps / 4, steps - 1];
            (float scaleT, float scaleH, float scaleW) = Kandinsky5VideoPipeline.GetRopeScaleFactor(height, width);

            // ONE fixed noise sample reused at every probe step — t is the only variable, not per-point
            // input variance. condLatent/condMask are T2V-mode zeros (no image conditioning), constant too.
            using Tensor noisy = SeedGenerator.CreateNoise(latentShape, seed: 42);
            using Tensor condLatent = Zeros(latentShape);
            using Tensor condMask = Zeros(maskShape);
            using Tensor packed = PackVisualCond(noisy, condLatent, condMask, latCh);

            backend.PreloadWeights(transformer.EnumerateWeights());

            // ── Determinism-floor control: two IDENTICAL baseline (W8A8 off) forwards at the same input.
            // cuDNN fused SDPA is run-to-run nondeterministic on some paths (LLM-decode-grind round-8
            // finding: "byte-A/Bs on qwen3 are meaningless"); this establishes the noise floor BELOW
            // which a W8A8-vs-baseline relL2 can't be attributed to quantization at all. ──
            backend.EnableW8A8 = false;
            Tensor floorA = transformer.ForwardVideo(backend, packed, timesteps[probeSteps[0]], qwen, clip, scaleT, scaleH, scaleW);
            backend.Sync();
            _ = floorA.DataPointer;
            backend.FreeActivations(trimPool: false);
            backend.EnableW8A8 = false;
            Tensor floorB = transformer.ForwardVideo(backend, packed, timesteps[probeSteps[0]], qwen, clip, scaleT, scaleH, scaleW);
            backend.Sync();
            _ = floorB.DataPointer;
            double floorRel = RelL2(floorB, floorA);
            floorA.Dispose();
            floorB.Dispose();
            backend.FreeActivations(trimPool: false);
            _output.WriteLine($"determinism floor (baseline vs baseline, t={timesteps[probeSteps[0]]:F1}): relL2={floorRel:e3}");

            _output.WriteLine("step\tt\trelL2");
            double maxRel = 0, minRel = double.MaxValue;
            foreach (int stepIdx in probeSteps)
            {
                float t = timesteps[stepIdx];

                backend.EnableW8A8 = false;
                Tensor baseline = transformer.ForwardVideo(backend, packed, t, qwen, clip, scaleT, scaleH, scaleW);
                backend.Sync();
                _ = baseline.DataPointer;

                backend.EnableW8A8 = true;
                Tensor candidate = transformer.ForwardVideo(backend, packed, t, qwen, clip, scaleT, scaleH, scaleW);
                backend.Sync();
                _ = candidate.DataPointer;

                double rel = RelL2(candidate, baseline);
                maxRel = Math.Max(maxRel, rel);
                minRel = Math.Min(minRel, rel);
                _output.WriteLine($"{stepIdx}\t{t:F1}\t{rel:e3}");

                baseline.Dispose();
                candidate.Dispose();
                backend.FreeActivations(trimPool: false);
            }
            backend.FreeWeights(transformer.EnumerateWeights());

            double drift = maxRel - minRel;
            _output.WriteLine($"relL2 range across schedule: [{minRel:e3}, {maxRel:e3}] drift={drift:e3} " +
                $"(determinism floor {floorRel:e3})");
            Assert.True(maxRel < 3e-2, $"W8A8 velocity relL2 exceeds the near-lossless budget at some timestep: max={maxRel:e3}");
        }
        finally { foreach (SafeTensorsLoader l in loaders) l.Dispose(); }
    }

    private static Tensor Zeros(TensorShape shape)
    {
        Tensor t = new Tensor(shape, DType.F32);
        new Span<float>((float*)t.DataPointer, checked((int)shape.ElementCount)).Clear();
        return t;
    }

    /// <summary>Ports <c>Kandinsky5VideoPipeline.PackVisualCond</c> (private): concat <c>[noisy(16), cond(16), mask(1)]</c> along the channel axis.</summary>
    private static Tensor PackVisualCond(Tensor noisy, Tensor condLatent, Tensor condMask, int latCh)
    {
        long b = noisy.Shape[0], t = noisy.Shape[2], h = noisy.Shape[3], w = noisy.Shape[4];
        Tensor packed = new Tensor(new TensorShape([b, 2L * latCh + 1, t, h, w]), DType.F32);
        long per = t * h * w;
        float* dst = (float*)packed.DataPointer;
        float* pn = (float*)noisy.DataPointer;
        float* pc = (float*)condLatent.DataPointer;
        float* pm = (float*)condMask.DataPointer;
        long chOut = 2L * latCh + 1;
        for (long bi = 0; bi < b; bi++)
        {
            long dstBase = bi * chOut * per;
            Buffer.MemoryCopy(pn + bi * latCh * per, dst + dstBase, latCh * per * 4, latCh * per * 4);
            Buffer.MemoryCopy(pc + bi * latCh * per, dst + dstBase + latCh * per, latCh * per * 4, latCh * per * 4);
            Buffer.MemoryCopy(pm + bi * per, dst + dstBase + 2L * latCh * per, per * 4, per * 4);
        }
        return packed;
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

    /// <summary>Raw headerless F32 <c>[seq, dim]</c> → <c>[1, seq, dim]</c>.</summary>
    private static Tensor LoadF32Tensor(string path, int embedDim)
    {
        byte[] data = File.ReadAllBytes(path);
        long totalFloats = data.Length / sizeof(float);
        if (totalFloats % embedDim != 0)
            throw new InvalidOperationException($"{path}: {totalFloats} floats not a multiple of {embedDim}.");
        int seqLen = (int)(totalFloats / embedDim);
        Tensor result = new Tensor(new TensorShape(1, seqLen, embedDim), DType.F32);
        fixed (byte* src = data) Buffer.MemoryCopy(src, (void*)result.DataPointer, data.Length, data.Length);
        return result;
    }

    /// <summary>Raw headerless F32 <c>[dim]</c> → <c>[1, dim]</c>.</summary>
    private static Tensor LoadPooled(string path, int embedDim)
    {
        byte[] data = File.ReadAllBytes(path);
        if (data.Length / sizeof(float) != embedDim)
            throw new InvalidOperationException($"{path}: expected {embedDim} floats.");
        Tensor result = new Tensor(new TensorShape(1, embedDim), DType.F32);
        fixed (byte* src = data) Buffer.MemoryCopy(src, (void*)result.DataPointer, data.Length, data.Length);
        return result;
    }
}
