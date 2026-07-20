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
using HartsyInference.Diffusion.Utilities;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Tests.Common;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Diffusion.Tests;

/// <summary>End-to-end SD3.5 image generation tests against single-file safetensor checkpoints. Auto-detects MMDiT-X dual-attention layers and QK-norm from the converted weights, then preloads the transformer + VAE to GPU before the denoise loop. Tests skip cleanly when checkpoints, tokenizers, or the PTX directory are missing — they're intended to run on a developer GPU box, not CI.</summary>
[Trait("Category", "Integration")]
public class Sd35GenerationTests
{
    private static string ClipVocab => TestPaths.Tokenizers.ClipVocab;
    private static string ClipMerges => TestPaths.Tokenizers.ClipMerges;
    private static string T5Spiece => TestPaths.Tokenizers.T5Spiece;
    private static string OutputDir => TestPaths.OutputDir;

    private readonly ITestOutputHelper _output;

    public Sd35GenerationTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void Sd35_Medium_Gpu_512_NoT5() =>
        RunGenerationTest(TestPaths.Sd35.Medium, "sd35_medium_512_noT5", width: 512, height: 512, steps: 28, cfgScale: 4.0f, useT5: false);

    [Fact]
    public void Sd35_Medium_Gpu_512_WithT5() =>
        RunGenerationTest(TestPaths.Sd35.Medium, "sd35_medium_512_withT5", width: 512, height: 512, steps: 28, cfgScale: 4.0f, useT5: true);

    [Fact]
    public void Sd35_Large_Gpu_512_NoT5() =>
        RunGenerationTest(TestPaths.Sd35.Large, "sd35_large_512_noT5", width: 512, height: 512, steps: 28, cfgScale: 4.0f, useT5: false);

    [Fact]
    public void Sd35_LargeTurbo_Gpu_512() =>
        RunGenerationTest(TestPaths.Sd35.LargeTurbo, "sd35_large_turbo_512", width: 512, height: 512, steps: 4, cfgScale: 1.0f, useT5: false);

    /// <summary>Diagnostic: SD3.5 Medium with cfg=1.0 (no CFG) and 4 steps. Used to isolate whether the patch-grid bug comes from CFG vs the base pipeline.</summary>
    [Fact]
    public void Sd35_Medium_Gpu_512_NoCfg_4Steps() =>
        RunGenerationTest(TestPaths.Sd35.Medium, "sd35_medium_512_nocfg_4steps", width: 512, height: 512, steps: 4, cfgScale: 1.0f, useT5: false);

    /// <summary>Smaller-resolution diagnostic that can fit alongside other GPU tenants (e.g. SwarmUI). cfg=1, 4 steps, 256×256 — minimal VRAM footprint.</summary>
    [Fact]
    public void Sd35_Medium_Gpu_256_NoCfg_4Steps() =>
        RunGenerationTest(TestPaths.Sd35.Medium, "sd35_medium_256_nocfg_4steps", width: 256, height: 256, steps: 4, cfgScale: 1.0f, useT5: false);

    private void RunGenerationTest(string checkpointPath, string outputName, int width, int height, int steps, float cfgScale, bool useT5)
    {
        if (!File.Exists(checkpointPath))
        {
            _output.WriteLine($"SKIPPED: Checkpoint not found: {checkpointPath}");
            return;
        }
        if (!File.Exists(ClipVocab) || !File.Exists(ClipMerges))
        {
            _output.WriteLine("SKIPPED: CLIP tokenizer files not found");
            return;
        }

        string assemblyDir = Path.GetDirectoryName(typeof(Sd35GenerationTests).Assembly.Location)!;
        string ptxDir = Path.Combine(assemblyDir, "Ptx");
        if (!Directory.Exists(ptxDir))
        {
            _output.WriteLine($"SKIPPED: PTX directory not found: {ptxDir}");
            return;
        }

        Stopwatch totalSw = Stopwatch.StartNew();
        Stopwatch sw = Stopwatch.StartNew();

        _output.WriteLine($"[1/8] Loading + converting checkpoint: {Path.GetFileName(checkpointPath)}");
        (Sd3CheckpointConverter.ConvertedWeights converted, SafeTensorsLoader loader) =
            Sd3CheckpointConverter.LoadAndConvert(checkpointPath);
        sw.Stop();
        _output.WriteLine($"  Converted in {sw.ElapsedMilliseconds}ms");

        using (loader)
        {
            // FP8 SD3.5 checkpoint: transformer + T5 stored at FP8 (1 byte / param). Casting to F32 would
            // explode the 2.5B-param Medium transformer to ~10GB and the T5 to ~20GB and OOM a 32GB system.
            // Pass as-is — `LoadWeights` auto-casts the few sites that read raw `float*` (QkNorm, norms,
            // embeddings); Linear weights stay native FP8 and CudaBackend casts F8→F16 per GEMM.
            // VAE must never be FP8 — cast to F32. CLIPs in this checkpoint are F16; pass through.
            Dictionary<string, Tensor> transformerWeights = converted.Transformer;
            Dictionary<string, Tensor> clipLWeights = converted.ClipL;
            Dictionary<string, Tensor> clipGWeights = converted.ClipG;
            Dictionary<string, Tensor> vaeF32 = CastWeightsToF32(converted.Vae);
            Dictionary<string, Tensor> t5Weights = useT5 ? converted.T5 : new Dictionary<string, Tensor>();

            // Auto-detect SD3.5 architecture (depth, dual-attn layers, QK-norm)
            Sd3Config config = Sd3Config.AutoDetect(transformerWeights);
            _output.WriteLine($"  Detected: depth={config.Depth}, hidden={config.HiddenSize}, heads={config.NumHeads}, " +
                $"qkNorm={config.UseQkNorm}, dualAttn={(config.DualAttentionLayers?.Length ?? 0)} layers");
            if (config.DualAttentionLayers is not null)
                _output.WriteLine($"  Dual-attention layers: [{string.Join(",", config.DualAttentionLayers)}]");

            _output.WriteLine("[2/8] Tokenizing prompt...");
            using ClipTokenizer tokenizer = new(ClipVocab, ClipMerges);

            string prompt = "A photograph of an astronaut riding a horse";
            string negPrompt = "";

            int[] promptTokensL = tokenizer.Encode(prompt);
            int[] negTokensL = tokenizer.Encode(negPrompt);
            int[] promptTokensG = tokenizer.Encode(prompt);
            int[] negTokensG = tokenizer.Encode(negPrompt);

            int promptEosL = ClipTokenizer.FindEosPosition(promptTokensL);
            int negEosL = ClipTokenizer.FindEosPosition(negTokensL);
            int promptEosG = ClipTokenizer.FindEosPosition(promptTokensG);
            int negEosG = ClipTokenizer.FindEosPosition(negTokensG);

            int[]? promptTokensT5 = null, negTokensT5 = null;
            int[]? promptMaskT5 = null, negMaskT5 = null;
            T5TextEncoder? t5Encoder = null;

            if (useT5 && File.Exists(T5Spiece) && t5Weights.Count > 0)
            {
                using T5Tokenizer t5Tokenizer = new(T5Spiece);
                promptTokensT5 = t5Tokenizer.Encode(prompt);
                negTokensT5 = t5Tokenizer.Encode(negPrompt);
                promptMaskT5 = CreateAttentionMask(promptTokensT5);
                negMaskT5 = CreateAttentionMask(negTokensT5);

                _output.WriteLine($"[3/8] Loading T5-XXL ({t5Weights.Count} tensors)...");
                sw.Restart();
                t5Encoder = new T5TextEncoder(T5TextEncoderConfig.Xxl);
                t5Encoder.LoadWeights(t5Weights);
                sw.Stop();
                _output.WriteLine($"  T5-XXL loaded in {sw.ElapsedMilliseconds}ms");
            }
            else
            {
                _output.WriteLine("[3/8] Skipping T5 (disabled or weights missing)");
            }

            _output.WriteLine("[4/8] Loading CLIP-L + CLIP-G...");
            sw.Restart();
            ClipTextEncoder clipL = new(ClipTextEncoderConfig.Sd3ClipL);
            clipL.LoadWeights(clipLWeights, "text_model");
            ClipTextEncoder clipG = new(ClipTextEncoderConfig.SdxlClipG);
            clipG.LoadWeights(clipGWeights, "text_model");
            sw.Stop();
            _output.WriteLine($"  CLIP-L+G loaded in {sw.ElapsedMilliseconds}ms");

            _output.WriteLine("[5/8] Loading SD3.5 MMDiT-X transformer...");
            sw.Restart();
            Sd3Transformer transformer = new Sd3Transformer(config);
            transformer.LoadWeights(transformerWeights);
            sw.Stop();
            _output.WriteLine($"  Transformer loaded in {sw.ElapsedMilliseconds}ms");

            _output.WriteLine("[6/8] Loading VAE...");
            sw.Restart();
            VaeDecoder vae = new(VaeConfig.Sd3);
            vae.LoadWeights(vaeF32);
            sw.Stop();
            _output.WriteLine($"  VAE loaded in {sw.ElapsedMilliseconds}ms");

            _output.WriteLine("[7/8] Initializing CUDA backend + preloading weights...");
            sw.Restart();
            using CudaBackend backend = new(deviceOrdinal: 0, ptxDir: ptxDir);
            backend.PreloadWeights(transformer.EnumerateWeights());
            backend.PreloadWeights(vae.EnumerateWeights());
            sw.Stop();
            _output.WriteLine($"  Backend ready in {sw.ElapsedMilliseconds}ms (device: {backend.Capabilities.Name})");

            // SD3.5 uses a higher scheduler shift than SD3 Medium. SD3.5 Medium/Large default to ~3.0;
            // SD3.5 Large Turbo (4-step distilled) uses ~3.0 with cfg=1.0.
            float schedulerShift = 3.0f;
            using Sd3Pipeline pipeline = new(backend, clipL, clipG, t5Encoder, transformer, vae, schedulerShift);

            TextToImageRequest request = new()
            {
                Prompt = prompt,
                NegativePrompt = negPrompt,
                Width = width,
                Height = height,
                Steps = steps,
                CfgScale = cfgScale,
                Seed = 42,
            };

            _output.WriteLine($"\n[8/8] Generating {width}x{height}, {steps} steps, cfg={cfgScale}, seed=42...");
            Stopwatch genSw = Stopwatch.StartNew();

            (byte[] rgbData, int outW, int outH, int seed) = pipeline.GenerateFromTokens(
                promptTokensL, negTokensL,
                promptTokensG, negTokensG,
                promptEosL, negEosL,
                promptEosG, negEosG,
                promptTokensT5, negTokensT5,
                promptMaskT5, negMaskT5,
                request,
                progress => _output.WriteLine($"  Step {progress.Step}/{progress.TotalSteps} ({progress.ElapsedMs:F0}ms)"));

            genSw.Stop();
            _output.WriteLine($"\nGeneration complete in {genSw.Elapsed.TotalSeconds:F1}s (seed={seed})");

            Assert.Equal(width, outW);
            Assert.Equal(height, outH);
            Assert.Equal(width * height * 3, rgbData.Length);
            ValidateImageNotDegenerate(rgbData);

            Directory.CreateDirectory(OutputDir);
            string outputPath = Path.Combine(OutputDir, $"{outputName}_{DateTime.Now:yyyyMMdd_HHmmss}.bmp");
            ImagePostProcessor.SaveBmp(outputPath, rgbData, outW, outH);
            _output.WriteLine($"  Saved: {outputPath}");

            totalSw.Stop();
            _output.WriteLine($"\nTotal: {totalSw.Elapsed.TotalSeconds:F1}s");

            transformer.Dispose();
            t5Encoder?.Dispose();
        }
    }

    private void ValidateImageNotDegenerate(byte[] rgbData)
    {
        int nonZero = 0, nonFF = 0;
        foreach (byte b in rgbData)
        {
            if (b != 0) nonZero++;
            if (b != 255) nonFF++;
        }
        float nonZeroPct = nonZero / (float)rgbData.Length * 100;
        float nonFFPct = nonFF / (float)rgbData.Length * 100;
        _output.WriteLine($"  Non-zero: {nonZeroPct:F1}%, Non-255: {nonFFPct:F1}%");
        Assert.True(nonZeroPct > 10, "Image appears to be all black");
        Assert.True(nonFFPct > 10, "Image appears to be all white");
    }

    private static int[] CreateAttentionMask(int[] tokenIds)
    {
        int[] mask = new int[tokenIds.Length];
        for (int i = 0; i < tokenIds.Length; i++)
            mask[i] = tokenIds[i] != T5Tokenizer.PadTokenId ? 1 : 0;
        return mask;
    }

    private static Dictionary<string, Tensor> CastWeightsToF32(Dictionary<string, Tensor> weights)
    {
        Dictionary<string, Tensor> f32 = new(weights.Count);
        foreach (KeyValuePair<string, Tensor> kvp in weights)
        {
            f32[kvp.Key] = (kvp.Value.DType == DType.F16 || kvp.Value.DType == DType.BF16)
                ? kvp.Value.CastTo(DType.F32)
                : kvp.Value;
        }
        return f32;
    }
}
