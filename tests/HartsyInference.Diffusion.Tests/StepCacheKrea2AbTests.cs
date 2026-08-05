using System.Diagnostics;
using Xunit;
using Xunit.Abstractions;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Models.Vae.QwenImage;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Tests.Common;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Step-cache calibration + warm A/B for Krea 2 Turbo (INFERENCE_ACCEL_GRIND §H1.5): 1024², the
/// production 8 distilled guidance-free steps, seed 42. An 8-step distilled sampler is the EXPECTED-NEGATIVE
/// case for residual caching (each step covers 2.5× the schedule of a 20-step model) — the calibration run
/// makes that expectation measurable instead of assumed, and the A/B is the proof either way. Conditioning
/// mirrors the verified generation test exactly (Qwen-Image ChatML template + drop-34, NO extra padding).
/// Armed caches force the eager path (step-graph topology can't vary per step); the baseline here runs
/// whatever the ambient HARTSY_DIT_GRAPH default is, so baseline wall = production wall.</summary>
[Trait("Category", "Integration")]
public class StepCacheKrea2AbTests
{
    private readonly ITestOutputHelper _output;
    public StepCacheKrea2AbTests(ITestOutputHelper output) => _output = output;

    private const int Width = 1024;
    private const int Height = 1024;
    private const int Steps = 8;
    private const int Trials = 3;
    private const string Prompt = "A photograph of an astronaut riding a horse";
    private const string QwenImageSystem =
        "Describe the image by detailing the color, shape, size, texture, quantity, text, " +
        "spatial relationships of the objects and background:";

    [Fact]
    public void Krea2Turbo_StepCache_Calibrate() => RunWithPipeline((generate, generateSeed, outputDir, stamp) =>
    {
        string calibPath = Path.Combine(outputDir, $"stepcache_calib_krea2turbo_{stamp}.csv");
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

    [Fact]
    public void Krea2Turbo_StepCache_WarmAb() => RunWithPipeline((generate, generateSeed, outputDir, stamp) =>
    {
        (string label, string stepCache, string? late)[] configs =
        {
            ("late0.5@0.15", "0.15", "0.5"),
            ("late0.25@0.15", "0.15", "0.25"),
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
        SaveBmp(outputDir, $"stepcache_ab_krea2turbo_baseline_{stamp}", baselineRgb!);

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
            SaveBmp(outputDir, $"stepcache_ab_krea2turbo_{label.Replace('@', '_')}_{stamp}", firstRgb!);
        }
        Environment.SetEnvironmentVariable("HARTSY_STEP_CACHE", null);
        Environment.SetEnvironmentVariable("HARTSY_STEP_CACHE_LATE", null);

        string csvPath = Path.Combine(outputDir, $"stepcache_ab_krea2turbo_{stamp}.csv");
        File.WriteAllLines(csvPath, csv);
        _output.WriteLine($"\nCSV: {csvPath}");
        Assert.True(byteStable, "Baseline was not byte-stable across trials — investigate before trusting the A/B.");
    });

    private void RunWithPipeline(Action<Func<byte[]>, Action<int>, string, string> body)
    {
        string rootDir = TestPaths.Krea2.TurboDir;
        if (!Directory.Exists(rootDir)) { _output.WriteLine($"SKIPPED: Krea2 Turbo dir not found: {rootDir}"); return; }
        string ptxDir = Path.Combine(Path.GetDirectoryName(typeof(StepCacheKrea2AbTests).Assembly.Location)!, "Ptx");
        if (!Directory.Exists(ptxDir)) { _output.WriteLine($"SKIPPED: no Ptx dir: {ptxDir}"); return; }
        Assert.True(File.Exists(Path.Combine(ptxDir, "stepcache.ptx")),
            "stepcache.ptx missing — run src/HartsyInference.Cuda/Kernels/dit/build.sh before the A/B.");

        Environment.SetEnvironmentVariable("HARTSY_STEP_CACHE", null);
        Environment.SetEnvironmentVariable("HARTSY_STEP_CACHE_CAP", null);
        Environment.SetEnvironmentVariable("HARTSY_STEP_CACHE_CALIB", null);
        Environment.SetEnvironmentVariable("HARTSY_STEP_CACHE_LATE", null);

        _output.WriteLine($"[load] Krea2 Turbo from {rootDir}...");
        (Dictionary<string, Tensor> txWeights, IReadOnlyList<SafeTensorsLoader> txLoaders) = Krea2CheckpointConverter.LoadTransformer(rootDir);
        (Dictionary<string, Tensor> teWeights, IReadOnlyList<SafeTensorsLoader> teLoaders) = Krea2CheckpointConverter.LoadTextEncoder(rootDir);
        (Dictionary<string, Tensor> vaeWeights, IReadOnlyList<SafeTensorsLoader> vaeLoaders) = Krea2CheckpointConverter.LoadVae(rootDir);
        try
        {
            Krea2Config config = Krea2Config.Turbo;
            Krea2Transformer transformer = new(config);
            transformer.LoadWeights(txWeights);
            using LlamaStyleEncoder textEncoder = new(LlamaStyleEncoderConfig.Qwen3_VL_4B);
            textEncoder.LoadWeights(teWeights);
            QwenImageVaeDecoder vae = new(VaeConfig.QwenImage);
            vae.LoadWeights(CastF32(vaeWeights));

            using Qwen2Tokenizer tokenizer = new();
            int[] promptTokens = tokenizer.EncodeChat(Prompt, systemPrompt: QwenImageSystem, addGenerationPrompt: true);

            using CudaBackend backend = new(deviceOrdinal: 0, ptxDir: ptxDir);
            backend.CacheWeightCasts = false;
            (nuint freeB, nuint totalB) = backend.Context.GetMemoryInfo();
            double freeGb = freeB / (1024.0 * 1024.0 * 1024.0);
            if (freeGb < 16.0) { _output.WriteLine($"SKIPPED: {freeGb:F1} GB free; need ≥16 GB."); transformer.Dispose(); return; }
            Assert.True(backend.SupportsDeviceStepCacheGate,
                "CudaBackend.SupportsDeviceStepCacheGate is false — stepcache.ptx didn't load.");

            using Krea2Pipeline pipeline = new(backend, textEncoder, transformer, vae, config);

            byte[] GenerateSeed(int seed)
            {
                TextToImageRequest request = new()
                {
                    Prompt = Prompt,
                    Width = Width,
                    Height = Height,
                    Steps = Steps,
                    CfgScale = 1.0f,
                    Seed = seed,
                };
                (byte[] rgb, int w, int h, _) = pipeline.GenerateFromTokens(promptTokens, null, request);
                Assert.Equal(Width, w);
                Assert.Equal(Height, h);
                return rgb;
            }
            byte[] Generate() => GenerateSeed(42);

            string outputDir = TestPaths.OutputDir;
            Directory.CreateDirectory(outputDir);
            string stamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");

            _output.WriteLine($"\n[warmup] {Width}x{Height}, {Steps} steps, no CFG, seed=42...");
            Stopwatch sw = Stopwatch.StartNew();
            byte[] warmupRgb = Generate();
            sw.Stop();
            _output.WriteLine($"  warmup: {sw.Elapsed.TotalSeconds:F1}s");
            SaveBmp(outputDir, $"stepcache_ab_krea2turbo_warmup_{stamp}", warmupRgb);

            body(Generate, s => GenerateSeed(s), outputDir, stamp);
            transformer.Dispose();
        }
        finally
        {
            foreach (SafeTensorsLoader l in txLoaders) l.Dispose();
            foreach (SafeTensorsLoader l in teLoaders) l.Dispose();
            foreach (SafeTensorsLoader l in vaeLoaders) l.Dispose();
        }
    }

    private void SaveBmp(string dir, string name, byte[] rgb)
    {
        string path = Path.Combine(dir, name + ".bmp");
        ImagePostProcessor.SaveBmp(path, rgb, Width, Height);
        _output.WriteLine($"  saved {path}");
    }

    private static Dictionary<string, Tensor> CastF32(Dictionary<string, Tensor> w)
    {
        Dictionary<string, Tensor> f32 = new Dictionary<string, Tensor>(w.Count);
        foreach (KeyValuePair<string, Tensor> kvp in w)
        {
            f32[kvp.Key] = (kvp.Value.DType == DType.F16 || kvp.Value.DType == DType.BF16)
                ? kvp.Value.CastTo(DType.F32)
                : kvp.Value;
        }
        return f32;
    }
}
