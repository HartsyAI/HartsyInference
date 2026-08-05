using System.Diagnostics;
using Xunit;
using Xunit.Abstractions;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Recipes.Image;
using HartsyInference.Engine.Requests;
using HartsyInference.Tests.Common;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Step-cache calibration + warm A/B for Flux.2 Dev, driven through the ENGINE recipe
/// (`Flux2Recipe.Construct` + `IRecipePipeline.Generate`) rather than a hand-built harness — the Ideogram
/// round proved harness-side conditioning divergence produces silently-broken baselines, and Flux.2 Dev's
/// Mistral-tekken marker splicing is exactly the kind of path not to duplicate. 1024², the production 50
/// steps / guidance 3.5 embedded (no CFG — ONE cache), seed 42. Q4_K_S GGUF (the catalog pin) keeps the
/// step-graph OFF by default for Dev, so the armed-cache eager path and the baseline share topology.
/// Skips cleanly when the GGUF or VRAM is absent.</summary>
[Trait("Category", "Integration")]
public class StepCacheFlux2AbTests
{
    private readonly ITestOutputHelper _output;
    public StepCacheFlux2AbTests(ITestOutputHelper output) => _output = output;

    private const int Width = 1024;
    private const int Height = 1024;
    private const int Steps = 50;
    private const int Trials = 3;
    private const string Prompt = "A photograph of an astronaut riding a horse on the moon";

    [Fact]
    public void Flux2Dev_StepCache_Calibrate() => RunWithPipeline((generate, generateSeed, outputDir, stamp) =>
    {
        string calibPath = Path.Combine(outputDir, $"stepcache_calib_flux2dev_{stamp}.csv");
        Environment.SetEnvironmentVariable("HARTSY_STEP_CACHE", "0.000001");
        Environment.SetEnvironmentVariable("HARTSY_STEP_CACHE_CALIB", calibPath);
        try
        {
            generateSeed(42);
        }
        finally
        {
            Environment.SetEnvironmentVariable("HARTSY_STEP_CACHE", null);
            Environment.SetEnvironmentVariable("HARTSY_STEP_CACHE_CALIB", null);
        }
        _output.WriteLine($"Calibration pairs: {calibPath} ({File.ReadAllLines(calibPath).Length} rows)");
    });

    /// <summary>Baseline determinism probe: 3 uncached gens, pairwise SSIM + max byte delta. The first A/B
    /// found the baseline NOT byte-stable (unique among the five models A/B'd so far) — this quantifies
    /// whether the wobble is ULP-noise (SSIM ≈ 1, gate numbers stand) or real nondeterminism.</summary>
    [Fact]
    public void Flux2Dev_Baseline_Stability() => RunWithPipeline((generate, generateSeed, outputDir, stamp) =>
    {
        byte[] a = generate();
        byte[] b = generate();
        byte[] c = generate();
        double ab = Ssim.Compute(a, b, Width, Height);
        double ac = Ssim.Compute(a, c, Width, Height);
        double bc = Ssim.Compute(b, c, Width, Height);
        int diffAb = 0, maxDelta = 0;
        for (int i = 0; i < a.Length; i++)
        {
            int d = Math.Abs(a[i] - b[i]);
            if (d != 0) { diffAb++; if (d > maxDelta) maxDelta = d; }
        }
        _output.WriteLine($"pairwise SSIM: a-b={ab:F6} a-c={ac:F6} b-c={bc:F6}");
        _output.WriteLine($"a-b bytes differing: {diffAb}/{a.Length} ({100.0 * diffAb / a.Length:F3}%), max delta={maxDelta}");
        SaveBmp(outputDir, $"stability_a_{stamp}", a);
        SaveBmp(outputDir, $"stability_b_{stamp}", b);
    });

    [Fact]
    public void Flux2Dev_StepCache_WarmAb()
    {
        // Calibration 2026-07-22 (49 pairs, R²=0.70): V-shaped residual drift — falls into a cheap
        // MID-schedule valley (0.05) then rises late (0.34), and the indicator tracks it. Late-window
        // would be wrong here (the tail is expensive again); the poly places reuses in the valley.
        Environment.SetEnvironmentVariable("HARTSY_STEP_CACHE_POLY", "0.616388,-29.5305,456.02,-1890.27");
        try { Flux2Dev_WarmAbBody(); }
        finally { Environment.SetEnvironmentVariable("HARTSY_STEP_CACHE_POLY", null); }
    }

    private void Flux2Dev_WarmAbBody() => RunWithPipeline((generate, generateSeed, outputDir, stamp) =>
    {
        (string label, string stepCache, string? late)[] configs =
        {
            ("poly0.15", "0.15", null),
            ("poly0.25", "0.25", null),
            ("poly0.4", "0.4", null),
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
        SaveBmp(outputDir, $"stepcache_ab_flux2dev_baseline_{stamp}", baselineRgb!);

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
            SaveBmp(outputDir, $"stepcache_ab_flux2dev_{label.Replace('@', '_')}_{stamp}", firstRgb!);
        }
        Environment.SetEnvironmentVariable("HARTSY_STEP_CACHE", null);
        Environment.SetEnvironmentVariable("HARTSY_STEP_CACHE_LATE", null);

        string csvPath = Path.Combine(outputDir, $"stepcache_ab_flux2dev_{stamp}.csv");
        File.WriteAllLines(csvPath, csv);
        _output.WriteLine($"\nCSV: {csvPath}");
        Assert.True(byteStable, "Baseline was not byte-stable across trials — investigate before trusting the A/B.");
    });

    private void RunWithPipeline(Action<Func<byte[]>, Action<int>, string, string> body)
    {
        string ggufPath = Path.Combine(TestPaths.ModelsDir, "Stable-Diffusion", "Flux2", "flux2-dev-Q4_K_S.gguf");
        if (!File.Exists(ggufPath)) { _output.WriteLine($"SKIPPED: no Flux.2 Dev GGUF: {ggufPath}"); return; }
        string ptxDir = Path.Combine(Path.GetDirectoryName(typeof(StepCacheFlux2AbTests).Assembly.Location)!, "Ptx");
        if (!Directory.Exists(ptxDir)) { _output.WriteLine($"SKIPPED: no Ptx dir: {ptxDir}"); return; }
        Assert.True(File.Exists(Path.Combine(ptxDir, "stepcache.ptx")),
            "stepcache.ptx missing — run src/HartsyInference.Cuda/Kernels/dit/build.sh before the A/B.");

        Environment.SetEnvironmentVariable("HARTSY_STEP_CACHE", null);
        Environment.SetEnvironmentVariable("HARTSY_STEP_CACHE_CAP", null);
        Environment.SetEnvironmentVariable("HARTSY_STEP_CACHE_CALIB", null);
        Environment.SetEnvironmentVariable("HARTSY_STEP_CACHE_LATE", null);
        // Engine side-model resolution (Mistral TE + tokenizer + Flux2 VAE) roots here.
        Environment.SetEnvironmentVariable("HARTSYINFERENCE_MODELS", TestPaths.ModelsDir);

        using CudaBackend backend = new(deviceOrdinal: 0, ptxDir: ptxDir);
        (nuint freeB, nuint totalB) = backend.Context.GetMemoryInfo();
        double freeGb = freeB / (1024.0 * 1024.0 * 1024.0);
        if (freeGb < 20.0) { _output.WriteLine($"SKIPPED: {freeGb:F1} GB free; need ≥20 GB."); return; }
        Assert.True(backend.SupportsDeviceStepCacheGate,
            "CudaBackend.SupportsDeviceStepCacheGate is false — stepcache.ptx didn't load.");

        _output.WriteLine($"[load] Flux.2 Dev via engine recipe: {Path.GetFileName(ggufPath)}...");
        Flux2Recipe recipe = new Flux2Recipe();
        using IRecipePipeline pipeline = recipe.Construct(new RecipeContext { CheckpointPath = ggufPath, Backend = backend });

        byte[] GenerateSeed(int seed)
        {
            ImageResult result = pipeline.Generate(new ImageRequest
            {
                Prompt = Prompt,
                Width = Width,
                Height = Height,
                Steps = Steps,
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
        SaveBmp(outputDir, $"stepcache_ab_flux2dev_warmup_{stamp}", warmupRgb);

        body(Generate, s => GenerateSeed(s), outputDir, stamp);
    }

    private void SaveBmp(string dir, string name, byte[] rgb)
    {
        string path = Path.Combine(dir, name + ".bmp");
        HartsyInference.Diffusion.Utilities.ImagePostProcessor.SaveBmp(path, rgb, Width, Height);
        _output.WriteLine($"  saved {path}");
    }
}
