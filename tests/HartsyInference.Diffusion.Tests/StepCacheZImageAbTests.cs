using System.Diagnostics;
using Xunit;
using Xunit.Abstractions;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Recipes.Image;
using HartsyInference.Engine.Requests;
using HartsyInference.Tests.Common;
using HartsyInference.Diffusion.Tests.Helpers;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Step-cache calibration + warm A/B for Z-Image Turbo, driven through the ENGINE recipe
/// (`ZImageRecipe.Construct` — the Flux.2 harness pattern; never re-implement conditioning). 1024², the
/// production 8 distilled guidance-free steps, seed 42. Like Krea2-Turbo this is the distilled few-step
/// case: expect at most 1–2 quality-free late reuses; the calibration decides. The checkpoint lives in the
/// Swarm tree (ZIMAGE_CKPT overrides); side models (Qwen3-4B TE + Flux VAE) resolve via
/// HARTSYINFERENCE_MODELS.</summary>
[Trait("Category", "Integration")]
public class StepCacheZImageAbTests
{
    private readonly ITestOutputHelper _output;
    public StepCacheZImageAbTests(ITestOutputHelper output) => _output = output;

    private const int Width = 1024;
    private const int Height = 1024;
    private const int Steps = 8;
    private const int Trials = 3;
    private const string Prompt = "A photograph of an astronaut riding a horse on the moon";

    private static string CheckpointPath =>
        Environment.GetEnvironmentVariable("ZIMAGE_CKPT")
        ?? "/home/hartsy/Desktop/Swarm/SwarmUI.not too old/Models/Stable-Diffusion/ZImage/SwarmUI_Z-Image-Turbo-FP8Mix.safetensors";

    [Fact]
    public void ZImageTurbo_StepCache_Calibrate() => RunWithPipeline((generate, generateSeed, outputDir, stamp) =>
    {
        string calibPath = Path.Combine(outputDir, $"stepcache_calib_zimageturbo_{stamp}.csv");
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
    public void ZImageTurbo_StepCache_WarmAb() => RunWithPipeline((generate, generateSeed, outputDir, stamp) =>
    {
        (string label, string stepCache, string? late)[] configs =
        {
            ("late0.5@0.15", "0.15", "0.5"),
            ("late0.5@0.3", "0.3", "0.5"),
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
        SaveBmp(outputDir, $"stepcache_ab_zimageturbo_baseline_{stamp}", baselineRgb!);

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
            SaveBmp(outputDir, $"stepcache_ab_zimageturbo_{label.Replace('@', '_')}_{stamp}", firstRgb!);
        }
        Environment.SetEnvironmentVariable("HARTSY_STEP_CACHE", null);
        Environment.SetEnvironmentVariable("HARTSY_STEP_CACHE_LATE", null);

        string csvPath = Path.Combine(outputDir, $"stepcache_ab_zimageturbo_{stamp}.csv");
        File.WriteAllLines(csvPath, csv);
        _output.WriteLine($"\nCSV: {csvPath}");
        Assert.True(byteStable, "Baseline was not byte-stable across trials — investigate before trusting the A/B.");
    });

    private void RunWithPipeline(Action<Func<byte[]>, Action<int>, string, string> body)
    {
        string ckpt = CheckpointPath;
        if (!File.Exists(ckpt)) { _output.WriteLine($"SKIPPED: no Z-Image checkpoint: {ckpt}"); return; }
        string ptxDir = Path.Combine(Path.GetDirectoryName(typeof(StepCacheZImageAbTests).Assembly.Location)!, "Ptx");
        if (!Directory.Exists(ptxDir)) { _output.WriteLine($"SKIPPED: no Ptx dir: {ptxDir}"); return; }
        Assert.True(File.Exists(Path.Combine(ptxDir, "stepcache.ptx")),
            "stepcache.ptx missing — run native/cuda/dit/build.sh before the A/B.");

        Environment.SetEnvironmentVariable("HARTSY_STEP_CACHE", null);
        Environment.SetEnvironmentVariable("HARTSY_STEP_CACHE_CAP", null);
        Environment.SetEnvironmentVariable("HARTSY_STEP_CACHE_CALIB", null);
        Environment.SetEnvironmentVariable("HARTSY_STEP_CACHE_LATE", null);
        Environment.SetEnvironmentVariable("HARTSYINFERENCE_MODELS", TestPaths.ModelsDir);

        using CudaBackend backend = new(deviceOrdinal: 0, ptxDir: ptxDir);
        (nuint freeB, nuint totalB) = backend.Context.GetMemoryInfo();
        double freeGb = freeB / (1024.0 * 1024.0 * 1024.0);
        if (freeGb < 16.0) { _output.WriteLine($"SKIPPED: {freeGb:F1} GB free; need ≥16 GB."); return; }
        Assert.True(backend.SupportsDeviceStepCacheGate,
            "CudaBackend.SupportsDeviceStepCacheGate is false — stepcache.ptx didn't load.");

        _output.WriteLine($"[load] Z-Image Turbo via engine recipe: {Path.GetFileName(ckpt)}...");
        ZImageRecipe recipe = new ZImageRecipe();
        using IRecipePipeline pipeline = recipe.Construct(new RecipeContext { CheckpointPath = ckpt, Backend = backend });

        byte[] GenerateSeed(int seed)
        {
            ImageResult result = pipeline.Generate(new ImageRequest
            {
                Prompt = Prompt,
                Width = Width,
                Height = Height,
                Steps = Steps,
                CfgScale = 1.0f,
                Seed = seed,
            }, null, CancellationToken.None);
            Assert.Equal(Width, result.Width);
            Assert.Equal(Height, result.Height);
            return result.Rgb;
        }
        byte[] Generate() => GenerateSeed(42);

        string outputDir = TestPaths.OutputDir;
        Directory.CreateDirectory(outputDir);
        string stamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");

        _output.WriteLine($"\n[warmup] {Width}x{Height}, {Steps} steps, seed=42...");
        Stopwatch sw = Stopwatch.StartNew();
        byte[] warmupRgb = Generate();
        sw.Stop();
        _output.WriteLine($"  warmup: {sw.Elapsed.TotalSeconds:F1}s");
        SaveBmp(outputDir, $"stepcache_ab_zimageturbo_warmup_{stamp}", warmupRgb);

        body(Generate, s => GenerateSeed(s), outputDir, stamp);
    }

    private void SaveBmp(string dir, string name, byte[] rgb)
    {
        string path = Path.Combine(dir, name + ".bmp");
        HartsyInference.Diffusion.Utilities.ImagePostProcessor.SaveBmp(path, rgb, Width, Height);
        _output.WriteLine($"  saved {path}");
    }
}
