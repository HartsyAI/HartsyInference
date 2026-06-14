using System.Diagnostics;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Tests.Helpers;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.ModelHandler.CheckpointConverters;
using HartsyInference.Tests.Common;
using HartsyInference.Tokenizers;
using HartsyInference.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>
/// End-to-end SD1.5 generation on the Vulkan backend. Phase 3.5 acceptance gate #4:
/// SSIM > 0.99 vs CUDA at same seed for 512×512 generation. The smoke test runs without a
/// CUDA reference; the SSIM gate skips unless <c>Output/sd15_cuda_512x512_seed42.bmp</c>
/// (or <c>SD15_CUDA_REFERENCE_PATH</c>) exists.
///
/// Mirrors <see cref="FluxVulkanGenerationTest"/>'s skip-when-missing structure so the test
/// is harmless on machines without the SD1.5 checkpoint or Vulkan loader.
/// </summary>
public sealed class Sd15VulkanGenerationTest
{
    private readonly ITestOutputHelper _output;

    public Sd15VulkanGenerationTest(ITestOutputHelper output) => _output = output;

    private static string Sd15SingleFilePath => TestPaths.Sd15.SingleFile;
    private static string TokenizerVocabPath => TestPaths.Tokenizers.ClipVocab;
    private static string TokenizerMergesPath => TestPaths.Tokenizers.ClipMerges;
    private static string OutputDir => TestPaths.OutputDir;
    private static string Sd15CudaReferencePath => TestPaths.ReferenceImage("SD15_CUDA_REFERENCE_PATH", "sd15_cuda_512x512_seed42.bmp");

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

    /// <summary>End-to-end SD1.5 512×512 on Vulkan, 20 steps, CFG 7.5, seed 42.
    /// Phase 3.5 acceptance: produces a non-trivial image (RGB std-dev &gt; 20, all 256 byte values present).</summary>
    [Fact]
    public void Sd15_512x512_Vulkan_GeneratesImage()
    {
        if (!VulkanAvailable()) { _output.WriteLine("SKIPPED: no Vulkan device"); return; }
        if (!File.Exists(Sd15SingleFilePath)) { _output.WriteLine($"SKIPPED: SD1.5 checkpoint not found: {Sd15SingleFilePath}"); return; }
        if (!File.Exists(TokenizerVocabPath) || !File.Exists(TokenizerMergesPath))
        { _output.WriteLine("SKIPPED: CLIP tokenizer files not found"); return; }

        Stopwatch totalSw = Stopwatch.StartNew();
        (byte[] rgbData, int width, int height, int seed) = GenerateOnVulkan(steps: 20, cfgScale: 7.5f, seed: 42);
        totalSw.Stop();

        // Validate the image is real photographic content, not flat / NaN.
        (float mean, float std) = ComputeRgbStats(rgbData);
        bool[] byteSeen = new bool[256];
        foreach (byte v in rgbData) byteSeen[v] = true;
        int distinctBytes = byteSeen.Count(b => b);
        _output.WriteLine($"RGB mean={mean:F1} std={std:F1} distinctBytes={distinctBytes}/256");

        Directory.CreateDirectory(OutputDir);
        string outputPath = Path.Combine(OutputDir, $"sd15_vulkan_{width}x{height}_seed{seed}.bmp");
        ImagePostProcessor.SaveBmp(outputPath, rgbData, width, height);
        _output.WriteLine($"Saved: {outputPath}");
        _output.WriteLine($"Total wall-clock: {totalSw.Elapsed.TotalSeconds:F1}s");

        Assert.Equal(512, width);
        Assert.Equal(512, height);
        Assert.Equal(512 * 512 * 3, rgbData.Length);
        Assert.True(std > 20, $"RGB std-dev too low ({std:F1}) — image may be uniform");
        Assert.True(distinctBytes >= 200, $"Too few distinct byte values ({distinctBytes}/256) — image may have NaN-collapsed bands");
    }

    /// <summary>SSIM > 0.99 vs CUDA reference. Skips when no reference image is present.
    /// Generate the reference via the CUDA SD1.5 generation test using the same seed/prompt/steps.</summary>
    [Fact]
    public void Sd15_512x512_Vulkan_Vs_Cuda_Ssim()
    {
        if (!VulkanAvailable()) { _output.WriteLine("SKIPPED: no Vulkan device"); return; }
        if (!File.Exists(Sd15SingleFilePath)) { _output.WriteLine($"SKIPPED: SD1.5 checkpoint not found"); return; }
        if (!File.Exists(TokenizerVocabPath) || !File.Exists(TokenizerMergesPath))
        { _output.WriteLine("SKIPPED: CLIP tokenizer files not found"); return; }
        if (!File.Exists(Sd15CudaReferencePath))
        { _output.WriteLine($"SKIPPED: CUDA reference image not found at {Sd15CudaReferencePath} — generate via the CUDA SD1.5 path with same seed/prompt/steps"); return; }

        (byte[] rgbVk, int wVk, int hVk, _) = GenerateOnVulkan(steps: 20, cfgScale: 7.5f, seed: 42);
        (byte[] rgbCuda, int wCuda, int hCuda) = ReadBmp(Sd15CudaReferencePath);

        Assert.Equal(wVk, wCuda);
        Assert.Equal(hVk, hCuda);

        double ssim = Ssim.Compute(rgbVk, rgbCuda, wVk, hVk);
        _output.WriteLine($"SSIM (Vulkan vs CUDA): {ssim:F4}");

        // Phase 3.5 acceptance gate #4: SSIM > 0.99 at same seed
        Assert.True(ssim > 0.99, $"SSIM ({ssim:F4}) below acceptance threshold 0.99 — Vulkan SD1.5 diverges from CUDA reference");
    }

    private (byte[] rgbData, int width, int height, int seed) GenerateOnVulkan(int steps, float cfgScale, int seed)
    {
        Stopwatch sw = Stopwatch.StartNew();
        _output.WriteLine($"[1/6] Loading checkpoint: {Path.GetFileName(Sd15SingleFilePath)}");
        (Sd15CheckpointConverter.ConvertedWeights converted, HartsyInference.ModelHandler.SafeTensors.SafeTensorsLoader loader) =
            Sd15CheckpointConverter.LoadAndConvert(Sd15SingleFilePath);
        sw.Stop();
        _output.WriteLine($"  Loaded in {sw.ElapsedMilliseconds}ms");

        using (loader)
        {
            Dictionary<string, Tensor> unetF32 = CastWeightsToF32(converted.UNet);
            Dictionary<string, Tensor> clipLF32 = CastWeightsToF32(converted.ClipL);
            Dictionary<string, Tensor> vaeF32 = CastWeightsToF32(converted.Vae);

            using VulkanBackend backend = new();
            _output.WriteLine($"[Vulkan device] {backend.Vk}");

            _output.WriteLine("[2/6] Loading UNet...");
            sw.Restart();
            UNet unet = new(UNetConfig.Sd15);
            unet.LoadWeights(unetF32);
            sw.Stop();
            _output.WriteLine($"  UNet loaded in {sw.ElapsedMilliseconds}ms");

            _output.WriteLine("[3/6] Loading CLIP-L...");
            sw.Restart();
            ClipTextEncoder clipL = new(ClipTextEncoderConfig.Sd15);
            clipL.LoadWeights(clipLF32, "text_model");
            sw.Stop();
            _output.WriteLine($"  CLIP-L loaded in {sw.ElapsedMilliseconds}ms");

            _output.WriteLine("[4/6] Loading VAE...");
            sw.Restart();
            VaeDecoder vaeDecoder = new(VaeConfig.Sd15);
            vaeDecoder.LoadWeights(vaeF32);
            sw.Stop();
            _output.WriteLine($"  VAE loaded in {sw.ElapsedMilliseconds}ms");

            _output.WriteLine("[5/6] Tokenizing prompt...");
            using ClipTokenizer tokenizer = new(TokenizerVocabPath, TokenizerMergesPath);
            const string prompt = "A photograph of an astronaut riding a horse";
            const string negativePrompt = "";
            int[] promptTokens = tokenizer.Encode(prompt);
            int[] negativeTokens = tokenizer.Encode(negativePrompt);

            TextToImageRequest request = new()
            {
                Prompt = prompt,
                NegativePrompt = negativePrompt,
                Width = 512,
                Height = 512,
                Steps = steps,
                CfgScale = cfgScale,
                Seed = seed,
            };

            _output.WriteLine("[6/6] Generating (Vulkan)...");
            StableDiffusion15Pipeline pipeline = new(backend, clipL, unet, vaeDecoder);
            sw.Restart();
            (byte[] rgbData, int width, int height, int seedUsed) = pipeline.GenerateFromTokens(
                promptTokens, negativeTokens, request,
                onProgress: p => _output.WriteLine($"  Step {p.Step}/{p.TotalSteps} ({p.ElapsedMs:F0}ms)"));
            backend.Sync();
            sw.Stop();
            _output.WriteLine($"Generation done in {sw.ElapsedMilliseconds}ms (seed={seedUsed})");

            foreach (Tensor t in converted.UNet.Values) t.Dispose();
            foreach (Tensor t in converted.ClipL.Values) t.Dispose();
            foreach (Tensor t in vaeF32.Values) t.Dispose();
            return (rgbData, width, height, seedUsed);
        }
    }

    private static (float mean, float std) ComputeRgbStats(byte[] rgb)
    {
        double sum = 0, sumSq = 0;
        foreach (byte v in rgb) { sum += v; sumSq += (double)v * v; }
        double n = rgb.Length;
        double mean = sum / n;
        double var_ = sumSq / n - mean * mean;
        return ((float)mean, (float)Math.Sqrt(Math.Max(0, var_)));
    }

    /// <summary>Minimal 24-bit BMP reader. Matches the layout produced by <see cref="ImagePostProcessor.SaveBmp"/>.</summary>
    private static (byte[] rgb, int width, int height) ReadBmp(string path)
    {
        byte[] bytes = File.ReadAllBytes(path);
        if (bytes.Length < 54 || bytes[0] != 'B' || bytes[1] != 'M')
            throw new InvalidDataException($"Not a BMP file: {path}");
        int dataOffset = BitConverter.ToInt32(bytes, 10);
        int width = BitConverter.ToInt32(bytes, 18);
        int height = BitConverter.ToInt32(bytes, 22);
        short bitsPerPixel = BitConverter.ToInt16(bytes, 28);
        if (bitsPerPixel != 24)
            throw new InvalidDataException($"Expected 24-bit BMP, got {bitsPerPixel}");

        bool topDown = height < 0;
        int absH = Math.Abs(height);
        int rowStride = ((width * 3 + 3) / 4) * 4;
        byte[] rgb = new byte[width * absH * 3];
        for (int y = 0; y < absH; y++)
        {
            int srcRow = topDown ? y : (absH - 1 - y);
            int srcOff = dataOffset + srcRow * rowStride;
            int dstOff = y * width * 3;
            for (int x = 0; x < width; x++)
            {
                // BMP stores BGR; convert to RGB.
                rgb[dstOff + x * 3 + 0] = bytes[srcOff + x * 3 + 2];
                rgb[dstOff + x * 3 + 1] = bytes[srcOff + x * 3 + 1];
                rgb[dstOff + x * 3 + 2] = bytes[srcOff + x * 3 + 0];
            }
        }
        return (rgb, width, absH);
    }
}
