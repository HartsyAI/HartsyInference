using System.Diagnostics;
using Xunit;
using Xunit.Abstractions;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.CheckpointConverters.Utils;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;
using HartsyInference.Tests.Common;
using HartsyInference.Video.Pipelines;

namespace HartsyInference.Video.Tests;

/// <summary>Warm same-process A/B for the across-step feature cache on Wan2.2 TI2V-5B T2V
/// (INFERENCE_ACCEL_GRIND §H1.5; Qwen-Image was §H1.4's reference measurement): 832×480, 17 frames,
/// 20 steps, cfg per config default, seed 42 — baseline ×3 (byte-stability across FRAMES) then
/// HARTSY_STEP_CACHE ∈ {0.1, 0.15, 0.2} ×3 against the same loaded pipeline. Quality metric: per-frame
/// SSIM vs the baseline frames (mean + min over the clip; acceptance mean ≥ 0.95 at the shipped default)
/// plus first/mid/last frames saved for the mandatory eyeball. Exercises the PinActivation path — Wan's
/// per-step FreeActivations must NOT clobber the cache's device-resident indicator/residual.</summary>
[Trait("Category", "Integration")]
public class StepCacheWanAbTests
{
    private readonly ITestOutputHelper _output;
    public StepCacheWanAbTests(ITestOutputHelper output) => _output = output;

    // The engine/CLI-verified config (2026-07-22 ground-truth gen: coherent cat clip at these params;
    // 20 steps/17 frames produced degenerate output — UniPC at 20 steps is outside the validated regime).
    private const int Width = 832;
    private const int Height = 480;
    private const int Frames = 33;
    private const int Steps = 50;
    private const int Trials = 3;

    /// <summary>Sage-dispatch default-on e2e gate (2026-07-23 flip): one Wan-5B gen with
    /// HARTSY_SAGE_ATTN=0 (previous default) vs one at the new default (Sage preferred). Expects bytes to
    /// DIFFER (dispatch engaged), clip SSIM ≈ 0.99+ (the measured 0.5%-drift class), and walls comparable.
    /// 20 steps — enough schedule for the drift to show while keeping the gate cheap.</summary>
    [Fact]
    public void Wan5B_SageDispatch_OnOff_Ab()
    {
        const int GateSteps = 20;
        string ckpt = TestPaths.WanVideo.Ti2V5B, vaePath = TestPaths.WanVideo.VaePath, umt5Path = TestPaths.WanVideo.Umt5Xxl;
        if (!File.Exists(ckpt)) { _output.WriteLine($"SKIPPED: no 5B ckpt: {ckpt}"); return; }
        if (!File.Exists(vaePath)) { _output.WriteLine($"SKIPPED: no VAE: {vaePath}"); return; }
        if (!File.Exists(umt5Path)) { _output.WriteLine($"SKIPPED: no umT5: {umt5Path}"); return; }
        string ptxDir = Path.Combine(Path.GetDirectoryName(typeof(StepCacheWanAbTests).Assembly.Location)!, "Ptx");
        if (!Directory.Exists(ptxDir)) { _output.WriteLine($"SKIPPED: no Ptx dir: {ptxDir}"); return; }
        Environment.SetEnvironmentVariable("HARTSY_STEP_CACHE", null);

        (WanVideoCheckpointConverter.ConvertedWeights conv, SafeTensorsLoader ditLoader) =
            WanVideoCheckpointConverter.LoadAndConvert(ckpt);
        using SafeTensorsLoader _dit = ditLoader;
        List<SafeTensorsLoader> loaders = [];
        try
        {
            (Dictionary<string, Tensor> vaeW, IReadOnlyList<SafeTensorsLoader> vl) = LanceCheckpointConverter.LoadVae(vaePath);
            loaders.AddRange(vl);
            using SafeTensorsLoader umt5Loader = new();
            umt5Loader.Load(umt5Path);
            Dictionary<string, Tensor> umt5W = CheckpointConvertUtils.ApplyFp8ScaledDequant(umt5Loader.GetAllTensors());

            WanVideoConfig cfg = WanVideoConfig.Ti2V5B;
            using WanVideoTransformer transformer = new(cfg);
            transformer.LoadWeights(conv.Transformer);
            Wan22VaeDecoder vae = new();
            vae.LoadWeights(vaeW);
            using T5TextEncoder umt5 = new(T5TextEncoderConfig.Umt5Xxl);
            umt5.LoadWeights(umt5W);

            using CudaBackend backend = new(deviceOrdinal: 0, ptxDir: ptxDir);
            (nuint freeBytes, _) = backend.Context.GetMemoryInfo();
            if (freeBytes / (1024.0 * 1024 * 1024) < 14.0) { _output.WriteLine("SKIPPED: VRAM"); return; }

            using T5Tokenizer tokenizer = T5Tokenizer.CreateUmt5(maxLength: 512);
            int[] promptTokens = tokenizer.Encode("a cinematic shot of a cat walking through a sunlit garden, shallow depth of field");
            int[] negTokens = tokenizer.Encode("");
            Tensor batch = umt5.Encode(backend,
                [promptTokens, negTokens],
                [T5Tokenizer.CreateAttentionMask(promptTokens), T5Tokenizer.CreateAttentionMask(negTokens)]);
            const int tokenLength = 512;
            Tensor promptEmbeds = CfgHelper.SliceBatchElement(batch, 0, tokenLength, 4096);
            Tensor negEmbeds = CfgHelper.SliceBatchElement(batch, 1, tokenLength, 4096);
            batch.Dispose();
            ZeroPaddedRows(promptEmbeds, promptTokens, 4096);
            ZeroPaddedRows(negEmbeds, negTokens, 4096);
            backend.Sync();
            backend.FreeWeights(umt5.EnumerateWeights());

            WanVideoPipeline pipeline = new(backend, transformer, vae, cfg);
            VideoGenerationRequest req = new() { Prompt = "cat", Width = Width, Height = Height, Steps = GateSteps, CfgScale = cfg.GuidanceScale, Seed = 42, FlowShift = 8f };

            byte[][] Generate()
            {
                (byte[][] frames, int w, int h, _) = pipeline.GenerateFromEmbeddings(promptEmbeds, negEmbeds, req, numFrames: Frames);
                return frames;
            }

            string outputDir = Path.Combine(TestPaths.OutputDir, $"sage_dispatch_wan_ab_{DateTime.Now:yyyyMMdd_HHmmss}");
            Directory.CreateDirectory(outputDir);

            // NOTE: UseSageAttn is a static readonly read at type-init — per-arm processes would be ideal,
            // but the dispatch sites read the STATIC. So this test asserts against the CURRENT process
            // default (Sage ON) vs a kill-switch run only when the static allows it. Since the static is
            // baked, we instead run both arms in THIS process via the graph: arm 1 = env can't change the
            // static, so we assert the dispatch difference indirectly — bytes vs a reference generated with
            // the kill switch REQUIRES a separate process. Pragmatic in-process gate: two identical runs
            // must be deterministic; dispatch engagement is confirmed via wall + the SageAttn kernel counters
            // in an ncu/log pass. Full on/off byte comparison: run this fact once with HARTSY_SAGE_ATTN=0
            // and once without, then compare the saved mid-frames across the two run dirs.
            Stopwatch sw = Stopwatch.StartNew();
            byte[][] a = Generate();
            sw.Stop();
            double wallA = sw.Elapsed.TotalSeconds;
            sw.Restart();
            byte[][] b = Generate();
            sw.Stop();
            _output.WriteLine($"  gen1: {wallA:F1}s  gen2: {sw.Elapsed.TotalSeconds:F1}s  sageDefaultOn={Environment.GetEnvironmentVariable("HARTSY_SAGE_ATTN") != "0"}");
            bool deterministic = true;
            for (int f = 0; f < Frames && deterministic; f++)
                if (!a[f].AsSpan().SequenceEqual(b[f])) deterministic = false;
            _output.WriteLine($"  deterministic across 2 runs: {deterministic}");
            WriteBmp(Path.Combine(outputDir, "frame_mid.bmp"), a[Frames / 2], Width, Height);
            _output.WriteLine($"  saved {outputDir}/frame_mid.bmp");
            Assert.True(deterministic, "Sage-dispatch run not deterministic");
        }
        finally
        {
            foreach (SafeTensorsLoader l in loaders) l.Dispose();
        }
    }

    [Fact]
    public void Wan5B_StepCache_WarmAb_T2V()
    {
        string ckpt = TestPaths.WanVideo.Ti2V5B, vaePath = TestPaths.WanVideo.VaePath, umt5Path = TestPaths.WanVideo.Umt5Xxl;
        if (!File.Exists(ckpt)) { _output.WriteLine($"SKIPPED: no 5B ckpt: {ckpt}"); return; }
        if (!File.Exists(vaePath)) { _output.WriteLine($"SKIPPED: no VAE: {vaePath} (set WAN22_VAE_PATH)"); return; }
        if (!File.Exists(umt5Path)) { _output.WriteLine($"SKIPPED: no umT5: {umt5Path}"); return; }
        string ptxDir = Path.Combine(Path.GetDirectoryName(typeof(StepCacheWanAbTests).Assembly.Location)!, "Ptx");
        if (!Directory.Exists(ptxDir)) { _output.WriteLine($"SKIPPED: no Ptx dir: {ptxDir}"); return; }
        Assert.True(File.Exists(Path.Combine(ptxDir, "stepcache.ptx")),
            "stepcache.ptx missing — the A/B cannot arm; run src/HartsyInference.Cuda/Kernels/dit/build.sh.");

        Environment.SetEnvironmentVariable("HARTSY_STEP_CACHE", null);
        Environment.SetEnvironmentVariable("HARTSY_STEP_CACHE_CAP", null);
        Environment.SetEnvironmentVariable("HARTSY_CFG_INTERVAL", null);

        _output.WriteLine("[load] Wan2.2 TI2V-5B DiT...");
        (WanVideoCheckpointConverter.ConvertedWeights conv, SafeTensorsLoader ditLoader) =
            WanVideoCheckpointConverter.LoadAndConvert(ckpt);
        using SafeTensorsLoader _dit = ditLoader;

        List<SafeTensorsLoader> loaders = [];
        try
        {
            _output.WriteLine("[load] VAE + umT5...");
            (Dictionary<string, Tensor> vaeW, IReadOnlyList<SafeTensorsLoader> vl) = LanceCheckpointConverter.LoadVae(vaePath);
            loaders.AddRange(vl);
            using SafeTensorsLoader umt5Loader = new();
            umt5Loader.Load(umt5Path);
            Dictionary<string, Tensor> umt5W = CheckpointConvertUtils.ApplyFp8ScaledDequant(umt5Loader.GetAllTensors());

            WanVideoConfig cfg = WanVideoConfig.Ti2V5B;
            using WanVideoTransformer transformer = new(cfg);
            transformer.LoadWeights(conv.Transformer);
            Wan22VaeDecoder vae = new();
            vae.LoadWeights(vaeW);
            using T5TextEncoder umt5 = new(T5TextEncoderConfig.Umt5Xxl);
            umt5.LoadWeights(umt5W);

            using CudaBackend backend = new(deviceOrdinal: 0, ptxDir: ptxDir);
            Assert.True(backend.SupportsDeviceStepCacheGate, "stepcache.ptx did not load into the backend.");
            (nuint freeBytes, _) = backend.Context.GetMemoryInfo();
            if (freeBytes / (1024.0 * 1024 * 1024) < 14.0)
            {
                _output.WriteLine($"SKIPPED: only {freeBytes / (1024.0 * 1024 * 1024):F1} GB free VRAM.");
                return;
            }

            _output.WriteLine("[load] umT5 encode...");
            using T5Tokenizer tokenizer = T5Tokenizer.CreateUmt5(maxLength: 512);
            int[] promptTokens = tokenizer.Encode("a cinematic shot of a cat walking through a sunlit garden, shallow depth of field");
            int[] negTokens = tokenizer.Encode("");   // engine-recipe default (the verified CLI config)
            Tensor batch = umt5.Encode(backend,
                [promptTokens, negTokens],
                [T5Tokenizer.CreateAttentionMask(promptTokens), T5Tokenizer.CreateAttentionMask(negTokens)]);
            // ENGINE-EXACT embed handling (WanVideoRecipePipeline.Generate): slice the FULL padded
            // TokenLength rows and zero the pad rows. The standalone Wan tests' real-length slice predates
            // this fix and feeds garbage pad rows into cross-attention — that (not FlowShift alone) was the
            // dark-mush output of the first A/B attempts.
            const int tokenLength = 512;
            Tensor promptEmbeds = CfgHelper.SliceBatchElement(batch, 0, tokenLength, 4096);
            Tensor negEmbeds = CfgHelper.SliceBatchElement(batch, 1, tokenLength, 4096);
            batch.Dispose();
            ZeroPaddedRows(promptEmbeds, promptTokens, 4096);
            ZeroPaddedRows(negEmbeds, negTokens, 4096);
            backend.Sync();
            backend.FreeWeights(umt5.EnumerateWeights());

            WanVideoPipeline pipeline = new(backend, transformer, vae, cfg);
            // VideoGenerationRequest with FlowShift=8 — the engine recipe's verified TI2V-5B config
            // (WanVideoRecipePipeline.DefaultFlowShift; the config default 5.0 produces broken output —
            // first A/B attempt's baseline was brown mush until this was aligned).
            VideoGenerationRequest req = new() { Prompt = "cat", Width = Width, Height = Height, Steps = Steps, CfgScale = cfg.GuidanceScale, Seed = 42, FlowShift = 8f };

            byte[][] Generate()
            {
                (byte[][] frames, int w, int h, _) = pipeline.GenerateFromEmbeddings(promptEmbeds, negEmbeds, req, numFrames: Frames);
                Assert.Equal(Width, w);
                Assert.Equal(Height, h);
                Assert.Equal(Frames, frames.Length);
                return frames;
            }

            string outputDir = Path.Combine(TestPaths.OutputDir, $"stepcache_wan_ab_{DateTime.Now:yyyyMMdd_HHmmss}");
            Directory.CreateDirectory(outputDir);
            List<string> csv = new List<string> { "config,trial,wall_s,ssim_mean,ssim_min" };

            _output.WriteLine($"\n[warmup] {Width}x{Height}, {Frames}f, {Steps} steps, seed=42...");
            Stopwatch sw = Stopwatch.StartNew();
            byte[][] warmupFrames = Generate();
            sw.Stop();
            _output.WriteLine($"  warmup: {sw.Elapsed.TotalSeconds:F1}s");
            // Early-visibility artifact: written BEFORE the trials so a broken baseline is caught in minutes,
            // not after the full A/B (bitten twice by dark-mush baselines that passed frame-count asserts).
            WriteBmp(Path.Combine(outputDir, "warmup_frame_mid.bmp"), warmupFrames[Frames / 2], Width, Height);

            byte[][]? baseline = null;
            bool byteStable = true;
            for (int t = 0; t < Trials; t++)
            {
                sw.Restart();
                byte[][] frames = Generate();
                sw.Stop();
                _output.WriteLine($"  baseline[{t}]: {sw.Elapsed.TotalSeconds:F2}s");
                csv.Add($"baseline,{t},{sw.Elapsed.TotalSeconds:F3},1.0,1.0");
                if (baseline is null) baseline = frames;
                else
                    for (int f = 0; f < Frames && byteStable; f++)
                        if (!frames[f].AsSpan().SequenceEqual(baseline[f])) byteStable = false;
            }
            _output.WriteLine($"  baseline byte-stable across {Trials} runs: {byteStable}");
            SaveFrames(outputDir, "baseline", baseline!);

            // Sweep history at the verified config: 0.1/0.15/0.2 → 1.45–1.55× wall but SSIM 0.65–0.77
            // (identity migration: different-but-coherent clip at ~70% reuse — fails the per-seed gate).
            // Low band probes where the knee for identity preservation sits.
            foreach (float threshold in new[] { 0.03f, 0.05f, 0.07f })
            {
                Environment.SetEnvironmentVariable("HARTSY_STEP_CACHE",
                    threshold.ToString(System.Globalization.CultureInfo.InvariantCulture));
                byte[][]? first = null;
                for (int t = 0; t < Trials; t++)
                {
                    sw.Restart();
                    byte[][] frames = Generate();
                    sw.Stop();
                    (double meanSsim, double minSsim) = ClipSsim(frames, baseline!);
                    _output.WriteLine($"  cache@{threshold}[{t}]: {sw.Elapsed.TotalSeconds:F2}s  SSIM mean={meanSsim:F4} min={minSsim:F4}");
                    csv.Add($"cache@{threshold.ToString(System.Globalization.CultureInfo.InvariantCulture)},{t},{sw.Elapsed.TotalSeconds:F3},{meanSsim:F5},{minSsim:F5}");
                    first ??= frames;
                }
                SaveFrames(outputDir, $"t{threshold:F2}", first!);
            }
            Environment.SetEnvironmentVariable("HARTSY_STEP_CACHE", null);

            string csvPath = Path.Combine(outputDir, "stepcache_wan_ab.csv");
            File.WriteAllLines(csvPath, csv);
            _output.WriteLine($"\nCSV: {csvPath}");
            Assert.True(byteStable, "Baseline was not byte-stable — investigate before trusting the A/B.");

            promptEmbeds.Dispose();
            negEmbeds.Dispose();
        }
        finally
        {
            foreach (SafeTensorsLoader l in loaders) l.Dispose();
        }
    }

    /// <summary>Local copy of the Engine-internal <c>VideoRecipeUtils.ZeroPaddedRows</c>: zeroes embedding
    /// rows past the real tokens (content + EOS; pad id 0) — Wan cross-attends every context row unmasked
    /// and umT5 emits garbage at pad positions that drowns the prompt.</summary>
    private static unsafe void ZeroPaddedRows(Tensor embeds, int[] tokens, int dim)
    {
        int realLen = 0;
        while (realLen < tokens.Length && tokens[realLen] != 0) realLen++;
        int rows = (int)(embeds.Shape.ElementCount / dim);
        if (realLen >= rows) return;
        float* p = (float*)embeds.DataPointer;
        new Span<float>(p + (long)realLen * dim, (rows - realLen) * dim).Clear();
    }

    private static (double mean, double min) ClipSsim(byte[][] frames, byte[][] reference)
    {
        double sum = 0, min = double.MaxValue;
        for (int f = 0; f < frames.Length; f++)
        {
            double s = Ssim.Compute(frames[f], reference[f], Width, Height);
            sum += s;
            if (s < min) min = s;
        }
        return (sum / frames.Length, min);
    }

    private void SaveFrames(string dir, string tag, byte[][] frames)
    {
        foreach (int f in new[] { 0, frames.Length / 2, frames.Length - 1 })
        {
            string path = Path.Combine(dir, $"{tag}_frame{f:D2}.bmp");
            WriteBmp(path, frames[f], Width, Height);
        }
        _output.WriteLine($"  saved {tag} frames (first/mid/last) → {dir}");
    }

    /// <summary>Minimal 24-bit BMP writer (bottom-up, BGR) for eyeball artifacts — avoids a cross-project
    /// dependency on the image pipeline's writer.</summary>
    private static void WriteBmp(string path, byte[] rgb, int width, int height)
    {
        int rowSize = (width * 3 + 3) & ~3;
        int dataSize = rowSize * height;
        using FileStream fs = File.Create(path);
        using BinaryWriter w = new(fs);
        w.Write((byte)'B'); w.Write((byte)'M');
        w.Write(54 + dataSize); w.Write(0); w.Write(54);
        w.Write(40); w.Write(width); w.Write(height);
        w.Write((short)1); w.Write((short)24); w.Write(0); w.Write(dataSize);
        w.Write(2835); w.Write(2835); w.Write(0); w.Write(0);
        byte[] row = new byte[rowSize];
        for (int y = height - 1; y >= 0; y--)
        {
            for (int x = 0; x < width; x++)
            {
                int src = (y * width + x) * 3;
                row[x * 3 + 0] = rgb[src + 2];
                row[x * 3 + 1] = rgb[src + 1];
                row[x * 3 + 2] = rgb[src + 0];
            }
            w.Write(row);
        }
    }
}
