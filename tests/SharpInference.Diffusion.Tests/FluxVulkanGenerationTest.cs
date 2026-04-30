using System.Diagnostics;
using SharpInference.Core.Tensors;
using SharpInference.Diffusion.Models.Denoisers;
using SharpInference.Diffusion.Models.TextEncoders;
using SharpInference.Diffusion.Models.Vae;
using SharpInference.Diffusion.Pipelines;
using SharpInference.Diffusion.Requests;
using SharpInference.Diffusion.Utilities;
using SharpInference.ModelHandler.CheckpointConverters;
using SharpInference.Tests.Common;
using SharpInference.Tokenizers;
using SharpInference.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace SharpInference.Diffusion.Tests;

/// <summary>
/// End-to-end Flux Schnell FP8 generation on the Vulkan backend. Mirrors the CUDA
/// path (<see cref="FluxGenerationTests.Schnell_SingleFile_GenerateImage_Gpu"/>)
/// but swaps the backend.
///
/// Skipped when the Schnell checkpoint or Vulkan loader is unavailable. Saves
/// the resulting image to <c>Output/flux_schnell_vulkan_*.bmp</c> alongside the
/// CUDA-generated reference image so they can be visually compared.
/// </summary>
public sealed class FluxVulkanGenerationTest
{
    private readonly ITestOutputHelper _output;

    public FluxVulkanGenerationTest(ITestOutputHelper output) => _output = output;

    private static string FluxSingleFilePath => TestPaths.Flux.Schnell;
    private static string TokenizerVocabPath => TestPaths.Tokenizers.ClipVocab;
    private static string TokenizerMergesPath => TestPaths.Tokenizers.ClipMerges;
    private static string T5SpieceModelPath => TestPaths.Tokenizers.T5Spiece;
    private static string OutputDir => TestPaths.OutputDir;

    private static bool VulkanAvailable()
    {
        try
        {
            using VulkanInstance instance = new();
            return instance.EnumeratePhysicalDevices().Length > 0;
        }
        catch { return false; }
    }

    private static Dictionary<string, Tensor> CastWeightsToF32(IReadOnlyDictionary<string, Tensor> weights)
    {
        Dictionary<string, Tensor> f32 = new(weights.Count);
        foreach (KeyValuePair<string, Tensor> kvp in weights)
        {
            DType dtype = kvp.Value.DType;
            f32[kvp.Key] = (dtype == DType.F16 || dtype == DType.BF16 || dtype.IsFp8)
                ? kvp.Value.CastTo(DType.F32)
                : kvp.Value;
        }
        return f32;
    }

    /// <summary>End-to-end Flux Schnell FP8, 4 steps, 512x512 on Vulkan.</summary>
    [Fact]
    public void Schnell_SingleFile_GenerateImage_Vulkan()
    {
        if (!VulkanAvailable())
        {
            _output.WriteLine("SKIPPED: no Vulkan device");
            return;
        }
        if (!File.Exists(FluxSingleFilePath))
        {
            _output.WriteLine($"SKIPPED: Schnell checkpoint not found: {FluxSingleFilePath}");
            return;
        }
        if (!File.Exists(TokenizerVocabPath) || !File.Exists(TokenizerMergesPath))
        {
            _output.WriteLine("SKIPPED: CLIP tokenizer files not found");
            return;
        }
        if (!File.Exists(T5SpieceModelPath))
        {
            _output.WriteLine("SKIPPED: T5 SentencePiece model not found");
            return;
        }

        Stopwatch totalSw = Stopwatch.StartNew();

        _output.WriteLine($"[1/7] Loading checkpoint: {Path.GetFileName(FluxSingleFilePath)}");
        Stopwatch sw = Stopwatch.StartNew();
        (FluxCheckpointConverter.ConvertedWeights converted, SharpInference.ModelHandler.SafeTensors.SafeTensorsLoader loader) =
            FluxCheckpointConverter.LoadAndConvert(FluxSingleFilePath);
        sw.Stop();
        _output.WriteLine($"  Loaded in {sw.ElapsedMilliseconds}ms");

        using (loader)
        {
            // Same FP8 strategy as CUDA: keep transformer/T5 weights in their native FP8 dtype,
            // cast VAE to F32 (small, F32-safe), let the Vulkan backend handle FP8→F16 casts via
            // CastIfNeeded inside DispatchMatmul.
            Dictionary<string, Tensor> vaeF32 = CastWeightsToF32(converted.Vae);

            // Vulkan backend — auto-transfer (no preload), 12B params would not fit in VRAM.
            using VulkanBackend backend = new();
            _output.WriteLine($"[Vulkan device] {backend.Vk}");

            (int doubleBlocks, int singleBlocks, bool hasGuidance) =
                FluxCheckpointConverter.DetectArchitecture(converted.Transformer);
            _output.WriteLine($"[2/7] Architecture: {doubleBlocks} double + {singleBlocks} single blocks, guidance={hasGuidance}");

            FluxConfig config = hasGuidance ? FluxConfig.Dev : FluxConfig.Schnell;

            _output.WriteLine("[3/7] Loading FluxTransformer (FP8 native)...");
            sw.Restart();
            FluxTransformer transformer = new(config);
            transformer.LoadWeights(converted.Transformer);
            sw.Stop();
            _output.WriteLine($"  Transformer loaded in {sw.ElapsedMilliseconds}ms");

            _output.WriteLine("[4/7] Loading CLIP-L...");
            sw.Restart();
            ClipTextEncoder clipL = new(ClipTextEncoderConfig.SdxlClipL);
            clipL.LoadWeights(converted.ClipL, "text_model");
            sw.Stop();
            _output.WriteLine($"  CLIP-L loaded in {sw.ElapsedMilliseconds}ms");

            _output.WriteLine("[5/7] Loading T5-XXL (FP8 native)...");
            sw.Restart();
            T5TextEncoder t5 = new(T5TextEncoderConfig.Xxl);
            t5.LoadWeights(converted.T5);
            sw.Stop();
            _output.WriteLine($"  T5-XXL loaded in {sw.ElapsedMilliseconds}ms");

            _output.WriteLine("[6/7] Loading VAE...");
            sw.Restart();
            VaeDecoder vaeDecoder = new(VaeConfig.Flux);
            vaeDecoder.LoadWeights(vaeF32);
            sw.Stop();
            _output.WriteLine($"  VAE loaded in {sw.ElapsedMilliseconds}ms");

            _output.WriteLine("[7/7] Tokenizing and generating (Vulkan)...");

            using ClipTokenizer clipTokenizer = new(TokenizerVocabPath, TokenizerMergesPath);
            string prompt = "A photograph of an astronaut riding a horse";
            int[] clipTokenIds = clipTokenizer.Encode(prompt);
            int eosPosition = ClipTokenizer.FindEosPosition(clipTokenIds);

            using T5Tokenizer t5Tokenizer = new(T5SpieceModelPath, maxLength: 256);
            int[] t5TokenIds = t5Tokenizer.Encode(prompt);
            int[] t5AttentionMask = T5Tokenizer.CreateAttentionMask(t5TokenIds);

            // Schnell: 4 steps, no CFG, 512×512 — direct comparison with CUDA reference image.
            // Override via env var for debugging — e.g. FLUX_VULKAN_STEPS=1 isolates first-pass issues.
            int steps = int.TryParse(Environment.GetEnvironmentVariable("FLUX_VULKAN_STEPS"), out int s) ? s : 4;
            TextToImageRequest request = new()
            {
                Prompt = prompt,
                Width = 512,
                Height = 512,
                Steps = steps,
                Seed = 42,
            };

            FluxPipeline pipeline = new(backend, clipL, t5, transformer, vaeDecoder, config);

            sw.Restart();
            (byte[] rgbData, int width, int height, int seed) = pipeline.GenerateFromTokens(
                clipTokenIds, eosPosition, t5TokenIds, t5AttentionMask,
                request, guidanceScale: 0.0f,
                onProgress: p => _output.WriteLine($"  Step {p.Step}/{p.TotalSteps} ({p.ElapsedMs:F0}ms)"));

            backend.Sync();
            sw.Stop();

            _output.WriteLine($"Generation done in {sw.ElapsedMilliseconds}ms (seed={seed})");

            Directory.CreateDirectory(OutputDir);
            string outputPath = Path.Combine(OutputDir, $"flux_schnell_vulkan_{width}x{height}_s{request.Steps}_seed{seed}.bmp");
            ImagePostProcessor.SaveBmp(outputPath, rgbData, width, height);
            _output.WriteLine($"Saved: {outputPath}");

            Assert.Equal(512, width);
            Assert.Equal(512, height);
            Assert.Equal(512 * 512 * 3, rgbData.Length);

            transformer.Dispose();
            t5.Dispose();
            foreach (Tensor t in converted.Transformer.Values) t.Dispose();
            foreach (Tensor t in converted.ClipL.Values) t.Dispose();
            foreach (Tensor t in converted.T5.Values) t.Dispose();
            foreach (Tensor t in vaeF32.Values) t.Dispose();
        }

        totalSw.Stop();
        _output.WriteLine($"Total wall-clock: {totalSw.Elapsed.TotalSeconds:F1}s");
    }
}
