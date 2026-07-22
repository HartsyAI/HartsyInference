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
using HartsyInference.ModelAssets.Gguf;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Tests.Common;
using HartsyInference.ModelAssets.Tokenizers;
using HartsyInference.Diffusion.Tests.Helpers;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Warm same-process A/B for the across-step feature cache (INFERENCE_ACCEL_GRIND §H1.4): Qwen-Image
/// GGUF at 1024², 20 steps, cfg 4, seed 42 — baseline (knob unset) ×3 for byte-stability + wall, then
/// HARTSY_STEP_CACHE ∈ {0.1, 0.15, 0.2} ×3 each, all against the SAME loaded pipeline (model load and weight
/// caches amortized exactly once, matching the house warm-A/B rule). The knob is read per-Generate, so
/// in-process env flips give a clean A/B without a process or SwarmUI restart. Every threshold image is
/// SSIM'd against the baseline (acceptance ≥ 0.95 at the shipped default) and saved for eyeballing.
/// Wall-clock + SSIM + per-stream compute/reuse counts (from pipeline logs) land in a CSV for
/// benchmarks/results/. Skips cleanly when weights or GPU are absent (dev-box test, not CI).</summary>
[Trait("Category", "Integration")]
public class StepCacheQwenAbTests
{
    private readonly ITestOutputHelper _output;

    public StepCacheQwenAbTests(ITestOutputHelper output) => _output = output;

    private const int Width = 1024;
    private const int Height = 1024;
    private const int Steps = 20;
    private const float Cfg = 4.0f;
    private const int Trials = 3;

    [Fact]
    public void QwenImage_StepCache_WarmAb_Gguf() => RunWarmAb(
        csvTag: "stepcache",
        configs: new (string label, string? stepCache, string? cfgInterval)[]
        {
            ("cache@0.1", "0.1", null),
            ("cache@0.15", "0.15", null),
            ("cache@0.2", "0.2", null),
        });

    /// <summary>H2: limited-interval CFG A/B (arXiv:2404.07724) + the H2.2 composability run — interval and
    /// step cache together (interval halves the gated steps, the cache skips block stacks on the rest; the
    /// skipped uncond steps stale the uncond cache and self-heal via the drift spike on re-entry).</summary>
    [Fact]
    public void QwenImage_CfgInterval_WarmAb_Gguf() => RunWarmAb(
        csvTag: "cfginterval",
        configs: new (string label, string? stepCache, string? cfgInterval)[]
        {
            ("interval@0.1,0.85", null, "0.1,0.85"),
            ("interval@0.15,0.9", null, "0.15,0.9"),
            ("compose@cache0.1+int0.1,0.85", "0.1", "0.1,0.85"),
        });

    /// <summary>H2 follow-up after the full-band negative result (early-skip flips Qwen-Image from photo to
    /// illustration — SSIM 0.35, prompt-fidelity regression): late-only bands keep guidance on the composition-
    /// setting early steps and skip the uncond forward only at low noise, where the paper's rationale (guidance
    /// barely changes the trajectory) should hold WITHOUT changing image identity.</summary>
    [Fact]
    public void QwenImage_CfgIntervalLateOnly_WarmAb_Gguf() => RunWarmAb(
        csvTag: "cfglate",
        configs: new (string label, string? stepCache, string? cfgInterval)[]
        {
            ("late@0.05,1", null, "0.05,1"),
            ("late@0.1,1", null, "0.1,1"),
            ("late@0.15,1", null, "0.15,1"),
        });

    private void RunWarmAb(string csvTag, (string label, string? stepCache, string? cfgInterval)[] configs)
    {
        string transformerPath = TestPaths.QwenImage.V1Gguf;
        string textEncoderPath = TestPaths.QwenImage.TextEncoder;
        string vaePath = TestPaths.QwenImage.Vae;
        if (!File.Exists(transformerPath)) { _output.WriteLine($"SKIPPED: no GGUF: {transformerPath}"); return; }
        if (!File.Exists(textEncoderPath)) { _output.WriteLine($"SKIPPED: no TE: {textEncoderPath}"); return; }
        if (!File.Exists(vaePath)) { _output.WriteLine($"SKIPPED: no VAE: {vaePath}"); return; }

        string ptxDir = Path.Combine(Path.GetDirectoryName(typeof(StepCacheQwenAbTests).Assembly.Location)!, "Ptx");
        if (!Directory.Exists(ptxDir)) { _output.WriteLine($"SKIPPED: no Ptx dir: {ptxDir}"); return; }

        // The A/B is meaningless if the knob can't arm — fail loud rather than silently measuring baseline twice.
        Assert.True(File.Exists(Path.Combine(ptxDir, "stepcache.ptx")),
            "stepcache.ptx missing — run native/cuda/dit/build.sh before the A/B.");

        // Guard against ambient knobs polluting the baseline (e.g. leftovers from a Swarm launcher env).
        Environment.SetEnvironmentVariable("HARTSY_STEP_CACHE", null);
        Environment.SetEnvironmentVariable("HARTSY_STEP_CACHE_CAP", null);
        Environment.SetEnvironmentVariable("HARTSY_CFG_INTERVAL", null);

        _output.WriteLine("[load] transformer (GGUF Q4_K, lazy)...");
        GgufModelLoader.LoadedGgufModel ggufHandle = GgufModelLoader.Load(transformerPath);
        Dictionary<string, Tensor> ggufRawDict = GgufModelLoader.RelabelRank2ToPyTorchOrder(ggufHandle.Weights);
        QwenImageCheckpointConverter.ConvertedWeights converted = QwenImageCheckpointConverter.Convert(ggufRawDict);

        using (ggufHandle)
        {
            using Qwen2Tokenizer tokenizer = new();
            const string qwenImageSystem =
                "Describe the image by detailing the color, shape, size, texture, quantity, text, " +
                "spatial relationships of the objects and background:";
            const int templatePrefixDrop = 34;
            const string prompt = "A photograph of an astronaut riding a horse";
            int[] promptTokens = tokenizer.EncodeChat(prompt, systemPrompt: qwenImageSystem, addGenerationPrompt: true);
            int[] negTokens = tokenizer.EncodeChat("", systemPrompt: qwenImageSystem, addGenerationPrompt: true);

            _output.WriteLine("[load] Qwen2.5-VL-7B text encoder...");
            LlamaStyleEncoder textEncoder = new LlamaStyleEncoder(LlamaStyleEncoderConfig.Qwen2_5_VL_7B);
            textEncoder.LoadWeights(LoadStandalone(textEncoderPath));

            _output.WriteLine("[load] transformer weights...");
            QwenImageConfig config = QwenImageConfig.V1;
            QwenImageTransformer transformer = new QwenImageTransformer(config);
            transformer.LoadWeights(converted.Transformer);

            _output.WriteLine("[load] VAE...");
            QwenImageVaeDecoder vae = new QwenImageVaeDecoder(VaeConfig.QwenImage);
            vae.LoadWeights(CastWeightsToF32(LoadStandalone(vaePath)));

            using CudaBackend backend = new CudaBackend(deviceOrdinal: 0, ptxDir: ptxDir);
            backend.CacheWeightCasts = false; // ~20B Q4_K: transient per-GEMM dequant, quantized weights stay resident

            Assert.True(backend.SupportsDeviceStepCacheGate,
                "CudaBackend.SupportsDeviceStepCacheGate is false — stepcache.ptx didn't load.");

            (nuint freeBytes, nuint totalBytes) = backend.Context.GetMemoryInfo();
            double freeGb = freeBytes / (1024.0 * 1024.0 * 1024.0);
            if (freeGb < 14.0)
            {
                _output.WriteLine($"SKIPPED: only {freeGb:F1} GB free VRAM; the GGUF path needs ≥14 GB.");
                transformer.Dispose();
                textEncoder.Dispose();
                return;
            }

            using QwenImagePipeline pipeline = new QwenImagePipeline(backend, textEncoder, transformer, vae, config);
            TextToImageRequest request = new TextToImageRequest
            {
                Prompt = prompt,
                NegativePrompt = "",
                Width = Width,
                Height = Height,
                Steps = Steps,
                CfgScale = Cfg,
                Seed = 42,
            };

            byte[] Generate()
            {
                (byte[] rgb, int w, int h, _) = pipeline.GenerateFromTokens(
                    promptTokens, negTokens, request, onProgress: null,
                    promptDropIndex: templatePrefixDrop, negativeDropIndex: templatePrefixDrop);
                Assert.Equal(Width, w);
                Assert.Equal(Height, h);
                return rgb;
            }

            string outputDir = TestPaths.OutputDir;
            Directory.CreateDirectory(outputDir);
            string stamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
            List<string> csv = new List<string> { "config,trial,wall_s,ssim_vs_baseline" };

            // Warmup: first gen pays one-time costs (dequant paths, cuDNN plan search, pool growth) — excluded.
            _output.WriteLine($"\n[warmup] {Width}x{Height}, {Steps} steps, cfg={Cfg}, seed=42...");
            Stopwatch sw = Stopwatch.StartNew();
            Generate();
            sw.Stop();
            _output.WriteLine($"  warmup: {sw.Elapsed.TotalSeconds:F1}s");

            // Baseline ×3: wall + byte-stability (the determinism precondition for everything downstream).
            byte[]? baselineRgb = null;
            bool byteStable = true;
            for (int t = 0; t < Trials; t++)
            {
                sw.Restart();
                byte[] rgb = Generate();
                sw.Stop();
                _output.WriteLine($"  baseline[{t}]: {sw.Elapsed.TotalSeconds:F2}s");
                csv.Add($"baseline,{t},{sw.Elapsed.TotalSeconds:F3},1.0");
                if (baselineRgb is null) baselineRgb = rgb;
                else if (!rgb.AsSpan().SequenceEqual(baselineRgb)) byteStable = false;
            }
            _output.WriteLine($"  baseline byte-stable across {Trials} runs: {byteStable}");
            SaveBmp(outputDir, $"{csvTag}_ab_baseline_{stamp}", baselineRgb!);

            // Knob configs ×3 each, same process, env read per-Generate.
            foreach ((string label, string? stepCache, string? cfgInterval) in configs)
            {
                Environment.SetEnvironmentVariable("HARTSY_STEP_CACHE", stepCache);
                Environment.SetEnvironmentVariable("HARTSY_CFG_INTERVAL", cfgInterval);
                byte[]? firstRgb = null;
                for (int t = 0; t < Trials; t++)
                {
                    sw.Restart();
                    byte[] rgb = Generate();
                    sw.Stop();
                    double ssim = Ssim.Compute(rgb, baselineRgb!, Width, Height);
                    _output.WriteLine($"  {label}[{t}]: {sw.Elapsed.TotalSeconds:F2}s  SSIM={ssim:F4}");
                    csv.Add($"{label},{t},{sw.Elapsed.TotalSeconds:F3},{ssim:F5}");
                    firstRgb ??= rgb;
                }
                string slug = label.Replace('@', '_').Replace(',', '-').Replace('+', '_');
                SaveBmp(outputDir, $"{csvTag}_ab_{slug}_{stamp}", firstRgb!);
            }
            Environment.SetEnvironmentVariable("HARTSY_STEP_CACHE", null);
            Environment.SetEnvironmentVariable("HARTSY_CFG_INTERVAL", null);

            string csvPath = Path.Combine(outputDir, $"{csvTag}_ab_qwen_{stamp}.csv");
            File.WriteAllLines(csvPath, csv);
            _output.WriteLine($"\nCSV: {csvPath}");
            Assert.True(byteStable, "Baseline was not byte-stable across trials — investigate before trusting the A/B.");

            transformer.Dispose();
            textEncoder.Dispose();
        }
    }

    private void SaveBmp(string dir, string name, byte[] rgb)
    {
        string path = Path.Combine(dir, name + ".bmp");
        ImagePostProcessor.SaveBmp(path, rgb, Width, Height);
        _output.WriteLine($"  saved {path}");
    }

    private static Dictionary<string, Tensor> LoadStandalone(string path)
    {
        SafeTensorsLoader loader = new SafeTensorsLoader();
        loader.Load(path);
        return loader.GetAllTensors();
    }

    private static Dictionary<string, Tensor> CastWeightsToF32(Dictionary<string, Tensor> weights)
    {
        Dictionary<string, Tensor> f32 = new Dictionary<string, Tensor>(weights.Count);
        foreach (KeyValuePair<string, Tensor> kvp in weights)
        {
            f32[kvp.Key] = (kvp.Value.DType == DType.F16 || kvp.Value.DType == DType.BF16)
                ? kvp.Value.CastTo(DType.F32)
                : kvp.Value;
        }
        return f32;
    }
}
