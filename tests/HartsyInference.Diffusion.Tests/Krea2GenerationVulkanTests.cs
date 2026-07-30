using System.Diagnostics;
using Xunit;
using Xunit.Abstractions;
using HartsyInference.Core.Tensors;
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
using HartsyInference.Vulkan;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Vulkan counterpart of <see cref="Krea2GenerationTests"/> — same weights, same prompt, same
/// seed, same request shape, only the backend differs. Written for the Vulkan bring-up's end-of-plan
/// real-weight end-to-end check (per <c>docs/Checklists/ROADMAP.md</c> §3 Phase 7's diffusion-domain gate:
/// "at least one representative real-weight model passes A/B against CUDA"). Run pinned to the 4090
/// (<c>MESA_VK_DEVICE_SELECT=10de:2684</c>) — the fp8 checkpoint (~13 GB) plus transient dequant does not
/// fit the 3060's 12 GB. Skips cleanly when weights or a Vulkan device are absent.</summary>
[Trait("Category", "Integration")]
public class Krea2GenerationVulkanTests
{
    private readonly ITestOutputHelper _output;
    public Krea2GenerationVulkanTests(ITestOutputHelper output) => _output = output;

    private const string QwenImageSystem =
        "Describe the image by detailing the color, shape, size, texture, quantity, text, " +
        "spatial relationships of the objects and background:";

    private static bool VulkanAvailable()
    {
        try { using VulkanInstance i = new(); return i.EnumeratePhysicalDevices().Length > 0; }
        catch { return false; }
    }

    [Fact]
    public void Krea2_Turbo_Vulkan_1024_NoCfg() =>
        RunGenerationTest(TestPaths.Krea2.TurboDir, Krea2Config.Turbo, "krea2_turbo_1024_vulkan", steps: 8, cfgScale: 1.0f);

    private void RunGenerationTest(string rootDir, Krea2Config config, string outputName, int steps, float cfgScale)
    {
        if (!VulkanAvailable()) { _output.WriteLine("SKIPPED: no Vulkan device"); return; }
        if (!Directory.Exists(rootDir)) { _output.WriteLine($"SKIPPED: Krea2 dir not found: {rootDir}"); return; }
        string spvDir = Path.Combine(Path.GetDirectoryName(typeof(Krea2GenerationVulkanTests).Assembly.Location)!, "Spirv");
        if (!Directory.Exists(spvDir)) { _output.WriteLine($"SKIPPED: Spirv dir not found: {spvDir}"); return; }

        Stopwatch sw = Stopwatch.StartNew();
        _output.WriteLine($"[1/6] Loading + converting Krea2 from {rootDir}...");
        (Dictionary<string, Tensor> txWeights, IReadOnlyList<SafeTensorsLoader> txLoaders) = Krea2CheckpointConverter.LoadTransformer(rootDir);
        (Dictionary<string, Tensor> teWeights, IReadOnlyList<SafeTensorsLoader> teLoaders) = Krea2CheckpointConverter.LoadTextEncoder(rootDir);
        (Dictionary<string, Tensor> vaeWeights, IReadOnlyList<SafeTensorsLoader> vaeLoaders) = Krea2CheckpointConverter.LoadVae(rootDir);
        _output.WriteLine($"  transformer={txWeights.Count} te={teWeights.Count} vae={vaeWeights.Count} keys");

        try
        {
            _output.WriteLine("[2/6] Building transformer + TE (Qwen3-VL-4B) + VAE (Qwen-Image)...");
            Krea2Transformer transformer = new(config);
            transformer.LoadWeights(txWeights);
            LlamaStyleEncoder textEncoder = new(LlamaStyleEncoderConfig.Qwen3_VL_4B);
            textEncoder.LoadWeights(teWeights);
            QwenImageVaeDecoder vae = new(VaeConfig.QwenImage);
            vae.LoadWeights(CastF32(vaeWeights));

            _output.WriteLine("[3/6] Tokenizing (Qwen-Image ChatML template, drop 34)...");
            using Qwen2Tokenizer tokenizer = new();
            string prompt = "A photograph of an astronaut riding a horse";
            int[] promptTokens = tokenizer.EncodeChat(prompt, systemPrompt: QwenImageSystem, addGenerationPrompt: true);
            int[] negTokens = tokenizer.EncodeChat("", systemPrompt: QwenImageSystem, addGenerationPrompt: true);

            _output.WriteLine("[4/6] Vulkan backend...");
            using VulkanBackend backend = new(deviceOrdinal: 0, spvDir: spvDir);
            backend.CacheWeightCasts = false;   // 13 GB fp8 → transient dequant (matches the CUDA test)
            _output.WriteLine($"  Device: {backend.Vk.DeviceName}");

            using Krea2Pipeline pipeline = new(backend, textEncoder, transformer, vae, config);
            TextToImageRequest request = new() { Prompt = prompt, Width = 1024, Height = 1024, Steps = steps, CfgScale = cfgScale, Seed = 42 };

            _output.WriteLine($"[5/6] Generating 1024x1024, {steps} steps, cfg={cfgScale}...");
            Stopwatch gen = Stopwatch.StartNew();
            (byte[] rgb, int w, int h, int seed) = pipeline.GenerateFromTokens(promptTokens, cfgScale > 1.0f ? negTokens : null, request,
                p => _output.WriteLine($"  Step {p.Step}/{p.TotalSteps} ({p.ElapsedMs:F0}ms)"));
            gen.Stop();
            _output.WriteLine($"  Generated in {gen.Elapsed.TotalSeconds:F1}s (seed={seed})");

            Assert.Equal(w * h * 3, rgb.Length);
            int nonZero = rgb.Count(b => b != 0), nonFF = rgb.Count(b => b != 255);
            _output.WriteLine($"  Non-zero {nonZero / (float)rgb.Length * 100:F1}%, Non-255 {nonFF / (float)rgb.Length * 100:F1}%");
            Assert.True(nonZero > rgb.Length * 0.1, "all black");
            Assert.True(nonFF > rgb.Length * 0.1, "all white");

            string outDir = TestPaths.OutputDir; Directory.CreateDirectory(outDir);
            string outPath = Path.Combine(outDir, $"{outputName}_{DateTime.Now:yyyyMMdd_HHmmss}.bmp");
            ImagePostProcessor.SaveBmp(outPath, rgb, w, h);
            _output.WriteLine($"  Saved: {outPath} ({sw.Elapsed.TotalSeconds:F1}s total)");
            transformer.Dispose();
            textEncoder.Dispose();
        }
        finally
        {
            foreach (SafeTensorsLoader l in txLoaders) l.Dispose();
            foreach (SafeTensorsLoader l in teLoaders) l.Dispose();
            foreach (SafeTensorsLoader l in vaeLoaders) l.Dispose();
        }
    }

    private static Dictionary<string, Tensor> CastF32(Dictionary<string, Tensor> w)
    {
        Dictionary<string, Tensor> o = new(w.Count);
        foreach (KeyValuePair<string, Tensor> kv in w)
            o[kv.Key] = (kv.Value.DType == DType.F16 || kv.Value.DType == DType.BF16) ? kv.Value.CastTo(DType.F32) : kv.Value;
        return o;
    }
}
