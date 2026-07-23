using System.Diagnostics;
using Xunit;
using Xunit.Abstractions;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Schedulers;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Tests.Common;
using HartsyInference.ModelAssets.Tokenizers;
using HartsyInference.Diffusion.Tests.Helpers;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Warm same-process A/B for the across-step First-Block cache on Ideogram 4 (INFERENCE_ACCEL_GRIND
/// §H1.4 replication): 1024², Default20 preset, seed 42 — baseline ×3 for byte-stability + wall, then each
/// config ×3 against the SAME loaded pipeline (both 9.3B DiTs resident under HARTSY_KEEP_MODELS, prompt/RoPE
/// caches warm — the house warm-A/B rule). Knobs are read per-Generate, so in-process env flips give a clean
/// A/B. Every config image is SSIM'd against baseline (acceptance ≥ 0.95) and saved for eyeballing.
/// The Calibrate fact logs (indicator drift → residual drift) pairs for the TeaCache-style polynomial fit
/// (the §H1.4 recipe: near-zero threshold = observe mode, all steps compute, pairs logged at zero extra cost).
/// Skips cleanly when weights or VRAM are absent (dev-box test, not CI). Ideogram runs TWO transformers per
/// step, so each stream's cache lives on its own model — reuse counts are reported per stream.</summary>
[Trait("Category", "Integration")]
public class StepCacheIdeogramAbTests
{
    private readonly ITestOutputHelper _output;
    public StepCacheIdeogramAbTests(ITestOutputHelper output) => _output = output;

    private const int Width = 1024;
    private const int Height = 1024;
    private const int Trials = 3;
    private const string Prompt = "A photograph of an astronaut riding a horse on the moon";

    /// <summary>Observe-mode calibration run: threshold ≈ 0 forces every step to compute while logging
    /// indicator→residual drift pairs (both streams append to one CSV). Two seeds for pair diversity.
    /// Fit degree-3 LSQ offline, ship via HARTSY_STEP_CACHE_POLY.</summary>
    [Fact]
    public void Ideogram4_StepCache_Calibrate() => RunWithPipeline((generate, generateSeed, outputDir, stamp) =>
    {
        string calibPath = Path.Combine(outputDir, $"stepcache_calib_ideogram_{stamp}.csv");
        Environment.SetEnvironmentVariable("HARTSY_STEP_CACHE", "0.000001");
        Environment.SetEnvironmentVariable("HARTSY_STEP_CACHE_CALIB", calibPath);
        try
        {
            generateSeed(42);
            generateSeed(1234);
        }
        finally
        {
            Environment.SetEnvironmentVariable("HARTSY_STEP_CACHE", null);
            Environment.SetEnvironmentVariable("HARTSY_STEP_CACHE_CALIB", null);
        }
        _output.WriteLine($"Calibration pairs: {calibPath} ({File.ReadAllLines(calibPath).Length} rows)");
    });

    /// <summary>One fresh-process generation (the warmup), image saved — the minimal ground-truth probe for
    /// "is the current tree's default config producing a correct image at all". Honors ambient HARTSY_DIT_F16.</summary>
    [Fact]
    public void Ideogram4_Baseline_SingleGen() => RunWithPipeline((generate, generateSeed, outputDir, stamp) =>
        _output.WriteLine("single-gen probe complete (warmup image saved)"));

    [Fact]
    public void Ideogram4_StepCache_WarmAb() => RunWithPipeline((generate, generateSeed, outputDir, stamp) =>
    {
        // Fixed-harness round 2: full-schedule raw budgets FAILED the eyeball (0.15 → 1.86× but murky/dark,
        // SSIM 0.73 — calibration says residual drift is 0.72 early vs 0.12 late while the indicator is flat,
        // so uniform reuse spends its budget exactly where it hurts). These configs confine reuse to the late
        // window (HARTSY_STEP_CACHE_LATE) where the drift floor lives.
        (string label, string stepCache, string? late)[] configs =
        {
            ("late0.5@0.15", "0.15", "0.5"),
            ("late0.5@0.3", "0.3", "0.5"),
            ("late0.3@0.15", "0.15", "0.3"),
        };

        List<string> csv = new List<string> { "config,trial,wall_s,ssim_vs_baseline" };
        Stopwatch sw = new Stopwatch();

        byte[]? baselineRgb = null;
        bool byteStable = true;
        for (int t = 0; t < Trials; t++)
        {
            sw.Restart();
            byte[] rgb = generate();
            sw.Stop();
            _output.WriteLine($"  baseline[{t}]: {sw.Elapsed.TotalSeconds:F2}s");
            csv.Add($"baseline,{t},{sw.Elapsed.TotalSeconds:F3},1.0");
            if (baselineRgb is null) baselineRgb = rgb;
            else if (!rgb.AsSpan().SequenceEqual(baselineRgb)) byteStable = false;
        }
        _output.WriteLine($"  baseline byte-stable across {Trials} runs: {byteStable}");
        SaveBmp(outputDir, $"stepcache_ab_ideogram_baseline_{stamp}", baselineRgb!);

        foreach ((string label, string stepCache, string? late) in configs)
        {
            Environment.SetEnvironmentVariable("HARTSY_STEP_CACHE", stepCache);
            Environment.SetEnvironmentVariable("HARTSY_STEP_CACHE_LATE", late);
            byte[]? firstRgb = null;
            for (int t = 0; t < Trials; t++)
            {
                sw.Restart();
                byte[] rgb = generate();
                sw.Stop();
                double ssim = Ssim.Compute(rgb, baselineRgb!, Width, Height);
                _output.WriteLine($"  {label}[{t}]: {sw.Elapsed.TotalSeconds:F2}s  SSIM={ssim:F4}");
                csv.Add($"{label},{t},{sw.Elapsed.TotalSeconds:F3},{ssim:F5}");
                firstRgb ??= rgb;
            }
            SaveBmp(outputDir, $"stepcache_ab_ideogram_{label.Replace('@', '_')}_{stamp}", firstRgb!);
        }
        Environment.SetEnvironmentVariable("HARTSY_STEP_CACHE", null);
        Environment.SetEnvironmentVariable("HARTSY_STEP_CACHE_LATE", null);

        string csvPath = Path.Combine(outputDir, $"stepcache_ab_ideogram_{stamp}.csv");
        File.WriteAllLines(csvPath, csv);
        _output.WriteLine($"\nCSV: {csvPath}");
        Assert.True(byteStable, "Baseline was not byte-stable across trials — investigate before trusting the A/B.");
    });

    /// <summary>Loads the full Ideogram 4 stack once (skip-guarded), runs one excluded warmup, then hands the
    /// body a seed-42 generate closure (plus an any-seed variant for calibration) against the warm pipeline.</summary>
    private void RunWithPipeline(Action<Func<byte[]>, Action<int>, string, string> body)
    {
        string dir = TestPaths.Ideogram4.Dir;
        if (!Directory.Exists(dir) || !Directory.Exists(Path.Combine(dir, "vae")))
        {
            _output.WriteLine($"SKIPPED: Ideogram 4 folder not found/incomplete: {dir}");
            return;
        }
        string vocab = TestPaths.Tokenizers.Qwen3Vocab;
        string merges = TestPaths.Tokenizers.Qwen3Merges;
        if (!File.Exists(vocab) || !File.Exists(merges))
        {
            _output.WriteLine($"SKIPPED: Qwen3 tokenizer not found ({vocab}). Set QWEN3_TOKENIZER_DIR.");
            return;
        }
        string ptxDir = Path.Combine(Path.GetDirectoryName(typeof(StepCacheIdeogramAbTests).Assembly.Location)!, "Ptx");
        if (!Directory.Exists(ptxDir)) { _output.WriteLine($"SKIPPED: no Ptx dir: {ptxDir}"); return; }
        Assert.True(File.Exists(Path.Combine(ptxDir, "stepcache.ptx")),
            "stepcache.ptx missing — run native/cuda/dit/build.sh before the A/B.");

        // Guard against ambient knobs polluting the baseline.
        Environment.SetEnvironmentVariable("HARTSY_STEP_CACHE", null);
        Environment.SetEnvironmentVariable("HARTSY_STEP_CACHE_CAP", null);
        Environment.SetEnvironmentVariable("HARTSY_STEP_CACHE_CALIB", null);
        Environment.SetEnvironmentVariable("HARTSY_STEP_CACHE_LATE", null);

        Stopwatch sw = Stopwatch.StartNew();
        List<SafeTensorsLoader> allLoaders = [];
        try
        {
            _output.WriteLine("[load] conditional + unconditional transformers + TE + VAE...");
            (Dictionary<string, Tensor> condW, IReadOnlyList<SafeTensorsLoader> condL) =
                Ideogram4CheckpointConverter.LoadTransformer(dir, unconditional: false);
            allLoaders.AddRange(condL);
            (Dictionary<string, Tensor> uncondW, IReadOnlyList<SafeTensorsLoader> uncondL) =
                Ideogram4CheckpointConverter.LoadTransformer(dir, unconditional: true);
            allLoaders.AddRange(uncondL);
            (Dictionary<string, Tensor> teW, IReadOnlyList<SafeTensorsLoader> teL) =
                Ideogram4CheckpointConverter.LoadTextEncoder(dir);
            allLoaders.AddRange(teL);
            (Dictionary<string, Tensor> vaeW, IReadOnlyList<SafeTensorsLoader> vaeL) =
                Ideogram4CheckpointConverter.LoadVae(dir);
            allLoaders.AddRange(vaeL);

            Ideogram4Config config = Ideogram4Config.V4;
            using Ideogram4Transformer conditional = new(config);
            conditional.LoadWeights(condW);
            using Ideogram4Transformer unconditional = new(config);
            unconditional.LoadWeights(uncondW);
            using LlamaStyleEncoder textEncoder = new(LlamaStyleEncoderConfig.Qwen3_VL_8B);
            textEncoder.LoadWeights(teW);
            VaeDecoder vae = new(VaeConfig.Flux2);
            vae.LoadWeights(vaeW);

            using CudaBackend backend = new(deviceOrdinal: 0, ptxDir: ptxDir);
            (nuint freeBytes, nuint totalBytes) = backend.Context.GetMemoryInfo();
            double freeGb = freeBytes / (1024.0 * 1024.0 * 1024.0);
            if (freeGb < 21.0)
            {
                _output.WriteLine($"SKIPPED: only {freeGb:F1} GB free VRAM; Ideogram 4 needs ≥21 GB.");
                return;
            }
            Assert.True(backend.SupportsDeviceStepCacheGate,
                "CudaBackend.SupportsDeviceStepCacheGate is false — stepcache.ptx didn't load.");

            // Mirror the ENGINE recipe exactly (Ideogram4RecipePipeline): no <think> block (Qwen3-VL-8B-
            // Instruct template) and TRIM the right-pad — EncodeChat pads to maxLength with BOS(151643) and
            // Ideogram runs unmasked attention, so ~2020 pad rows DROWN the prompt (deterministic degenerate
            // texture output — root-caused 2026-07-22; the engine/Swarm path was never affected).
            using Qwen3Tokenizer tokenizer = new(vocab, merges, maxLength: config.MaxTextTokens);
            int[] paddedTokens = tokenizer.EncodeChat(Prompt, includeThinkBlock: false);
            int end = paddedTokens.Length;
            while (end > 1 && paddedTokens[end - 1] == Qwen3Tokenizer.BosTokenId) end--;
            int[] promptTokens = paddedTokens[..end];
            using Ideogram4Pipeline pipeline = new(backend, textEncoder, conditional, unconditional, vae, config);
            Ideogram4SamplerPreset preset = Ideogram4SamplerPreset.Default20;

            byte[] GenerateSeed(int seed)
            {
                TextToImageRequest request = new()
                {
                    Prompt = Prompt,
                    NegativePrompt = "",
                    Width = Width,
                    Height = Height,
                    Steps = preset.NumSteps,
                    CfgScale = 7.0f,
                    Seed = seed,
                };
                (byte[] rgb, int w, int h, _) = pipeline.GenerateFromTokens(promptTokens, request, preset);
                Assert.Equal(Width, w);
                Assert.Equal(Height, h);
                return rgb;
            }
            byte[] Generate() => GenerateSeed(42);

            string outputDir = TestPaths.OutputDir;
            Directory.CreateDirectory(outputDir);
            string stamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");

            _output.WriteLine($"\n[warmup] {Width}x{Height}, preset={preset.Name}, seed=42...");
            sw.Restart();
            byte[] warmupRgb = Generate();
            sw.Stop();
            _output.WriteLine($"  warmup: {sw.Elapsed.TotalSeconds:F1}s");
            // The warmup is the only FRESH-process first generation in this harness — save it so warm-path
            // regressions (gen 2+ vs gen 1) are distinguishable from config-wide ones.
            SaveBmp(outputDir, $"stepcache_ab_ideogram_warmup_{stamp}", warmupRgb);

            body(Generate, s => GenerateSeed(s), outputDir, stamp);
        }
        finally
        {
            foreach (SafeTensorsLoader l in allLoaders) l.Dispose();
        }
    }

    private void SaveBmp(string dir, string name, byte[] rgb)
    {
        string path = Path.Combine(dir, name + ".bmp");
        ImagePostProcessor.SaveBmp(path, rgb, Width, Height);
        _output.WriteLine($"  saved {path}");
    }
}
