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
/// End-to-end SDXL 1024×1024 generation on the Vulkan backend. Phase 3.5 acceptance gates
/// #5/#6: SDXL produces a coherent 1024² image, SSIM &gt; 0.95 vs CUDA reference. The 1024²
/// resolution is the first time the Vulkan im2col path is exercised at full SDXL scale —
/// it's the regression guard for the 64-bit element-count boundary in the spatial kernels.
///
/// Skips when the SDXL checkpoint or Vulkan loader is unavailable. Saves the result to
/// <c>Output/sdxl_vulkan_*.bmp</c>.
/// </summary>
public sealed class SdxlVulkanGenerationTest
{
    private readonly ITestOutputHelper _output;

    public SdxlVulkanGenerationTest(ITestOutputHelper output) => _output = output;

    private static string SdxlSingleFilePath => TestPaths.Sdxl.SingleFile;
    private static string TokenizerVocabPath => TestPaths.Tokenizers.ClipVocab;
    private static string TokenizerMergesPath => TestPaths.Tokenizers.ClipMerges;
    private static string OutputDir => TestPaths.OutputDir;
    private static string SdxlCudaReferencePath => TestPaths.ReferenceImage("SDXL_CUDA_REFERENCE_PATH", "sdxl_cuda_1024x1024_seed42.bmp");

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

    /// <summary>SDXL 1024×1024 on Vulkan, 20 steps. Acceptance: produces a non-trivial image.</summary>
    [Fact]
    public void Sdxl_1024x1024_Vulkan_GeneratesImage()
    {
        if (!VulkanAvailable()) { _output.WriteLine("SKIPPED: no Vulkan device"); return; }
        if (!File.Exists(SdxlSingleFilePath)) { _output.WriteLine($"SKIPPED: SDXL checkpoint not found: {SdxlSingleFilePath}"); return; }
        if (!File.Exists(TokenizerVocabPath) || !File.Exists(TokenizerMergesPath))
        { _output.WriteLine("SKIPPED: CLIP tokenizer files not found"); return; }

        Stopwatch totalSw = Stopwatch.StartNew();
        (byte[] rgbData, int width, int height, int seed) = GenerateOnVulkan(steps: 20, cfgScale: 7.0f, seed: 42);
        totalSw.Stop();

        (float mean, float std) = ComputeRgbStats(rgbData);
        bool[] byteSeen = new bool[256];
        foreach (byte v in rgbData) byteSeen[v] = true;
        int distinctBytes = byteSeen.Count(b => b);
        _output.WriteLine($"RGB mean={mean:F1} std={std:F1} distinctBytes={distinctBytes}/256");

        Directory.CreateDirectory(OutputDir);
        string outputPath = Path.Combine(OutputDir, $"sdxl_vulkan_{width}x{height}_seed{seed}.bmp");
        ImagePostProcessor.SaveBmp(outputPath, rgbData, width, height);
        _output.WriteLine($"Saved: {outputPath}");
        _output.WriteLine($"Total wall-clock: {totalSw.Elapsed.TotalSeconds:F1}s");

        Assert.Equal(1024, width);
        Assert.Equal(1024, height);
        Assert.Equal(1024 * 1024 * 3, rgbData.Length);
        Assert.True(std > 20, $"RGB std-dev too low ({std:F1}) — image may be uniform");
        Assert.True(distinctBytes >= 200, $"Too few distinct byte values ({distinctBytes}/256) — image may have NaN-collapsed bands");
    }

    /// <summary>SSIM &gt; 0.95 vs CUDA reference. Skips when no reference image is present.</summary>
    [Fact]
    public void Sdxl_1024x1024_Vulkan_Vs_Cuda_Ssim()
    {
        if (!VulkanAvailable()) { _output.WriteLine("SKIPPED: no Vulkan device"); return; }
        if (!File.Exists(SdxlSingleFilePath)) { _output.WriteLine($"SKIPPED: SDXL checkpoint not found"); return; }
        if (!File.Exists(TokenizerVocabPath) || !File.Exists(TokenizerMergesPath))
        { _output.WriteLine("SKIPPED: CLIP tokenizer files not found"); return; }
        if (!File.Exists(SdxlCudaReferencePath))
        { _output.WriteLine($"SKIPPED: CUDA reference image not found at {SdxlCudaReferencePath}"); return; }

        (byte[] rgbVk, int wVk, int hVk, _) = GenerateOnVulkan(steps: 20, cfgScale: 7.0f, seed: 42);
        (byte[] rgbCuda, int wCuda, int hCuda) = ReadBmp(SdxlCudaReferencePath);

        Assert.Equal(wVk, wCuda);
        Assert.Equal(hVk, hCuda);

        double ssim = Ssim.Compute(rgbVk, rgbCuda, wVk, hVk);
        _output.WriteLine($"SSIM (Vulkan vs CUDA): {ssim:F4}");

        // Phase 3.5 acceptance gate #5/#6: SDXL SSIM > 0.95 (looser than SD1.5 because deeper FP16
        // path accumulates more drift; checklist § 5 explicitly allows this looser threshold).
        Assert.True(ssim > 0.95, $"SSIM ({ssim:F4}) below acceptance threshold 0.95");
    }

    private (byte[] rgbData, int width, int height, int seed) GenerateOnVulkan(int steps, float cfgScale, int seed)
    {
        Stopwatch sw = Stopwatch.StartNew();
        _output.WriteLine($"[1/7] Loading checkpoint: {Path.GetFileName(SdxlSingleFilePath)}");
        (SdxlCheckpointConverter.ConvertedWeights converted, HartsyInference.ModelHandler.SafeTensors.SafeTensorsLoader loader) =
            SdxlCheckpointConverter.LoadAndConvert(SdxlSingleFilePath);
        sw.Stop();
        _output.WriteLine($"  Loaded in {sw.ElapsedMilliseconds}ms");

        using (loader)
        {
            Dictionary<string, Tensor> unetF32 = CastWeightsToF32(converted.UNet);
            Dictionary<string, Tensor> clipLF32 = CastWeightsToF32(converted.ClipL);
            Dictionary<string, Tensor> clipGF32 = CastWeightsToF32(converted.ClipG);
            Dictionary<string, Tensor> vaeF32 = CastWeightsToF32(converted.Vae);

            using VulkanBackend backend = new();
            _output.WriteLine($"[Vulkan device] {backend.Vk}");

            _output.WriteLine("[2/7] Tokenizing prompt...");
            using ClipTokenizer tokenizer = new(TokenizerVocabPath, TokenizerMergesPath);
            const string prompt = "A photograph of an astronaut riding a horse";
            const string negPrompt = "blurry, low quality, deformed";
            int[] promptTokensL = tokenizer.Encode(prompt);
            int[] negTokensL = tokenizer.Encode(negPrompt);
            int[] promptTokensG = tokenizer.Encode(prompt);
            int[] negTokensG = tokenizer.Encode(negPrompt);
            int promptEosG = ClipTokenizer.FindEosPosition(promptTokensG);
            int negEosG = ClipTokenizer.FindEosPosition(negTokensG);

            _output.WriteLine("[3/7] Loading CLIP-L...");
            sw.Restart();
            ClipTextEncoder clipL = new(ClipTextEncoderConfig.SdxlClipL);
            clipL.LoadWeights(clipLF32, "text_model");
            sw.Stop();
            _output.WriteLine($"  CLIP-L loaded in {sw.ElapsedMilliseconds}ms");

            _output.WriteLine("[4/7] Loading CLIP-G...");
            sw.Restart();
            ClipTextEncoder clipG = new(ClipTextEncoderConfig.SdxlClipG);
            clipG.LoadWeights(clipGF32, "text_model");
            sw.Stop();
            _output.WriteLine($"  CLIP-G loaded in {sw.ElapsedMilliseconds}ms");

            _output.WriteLine("[5/7] Loading UNet...");
            sw.Restart();
            UNet unet = new(UNetConfig.SdxlBase);
            unet.LoadWeights(unetF32);
            sw.Stop();
            _output.WriteLine($"  UNet loaded in {sw.ElapsedMilliseconds}ms");

            _output.WriteLine("[6/7] Loading VAE...");
            sw.Restart();
            VaeDecoder vae = new(VaeConfig.Sdxl);
            vae.LoadWeights(vaeF32);
            sw.Stop();
            _output.WriteLine($"  VAE loaded in {sw.ElapsedMilliseconds}ms");

            _output.WriteLine("[7/7] Generating (Vulkan, 1024×1024)...");
            using SdxlPipeline pipeline = new(backend, clipL, clipG, unet, vae);

            TextToImageRequest request = new()
            {
                Prompt = prompt,
                NegativePrompt = negPrompt,
                Width = 1024,
                Height = 1024,
                Steps = steps,
                CfgScale = cfgScale,
                Seed = seed,
            };

            sw.Restart();
            (byte[] rgbData, int width, int height, int seedUsed) = pipeline.GenerateFromTokens(
                promptTokensL, negTokensL,
                promptTokensG, negTokensG,
                promptEosG, negEosG,
                request,
                onProgress: p => _output.WriteLine($"  Step {p.Step}/{p.TotalSteps} ({p.ElapsedMs:F0}ms)"));
            backend.Sync();
            sw.Stop();
            _output.WriteLine($"Generation done in {sw.ElapsedMilliseconds}ms (seed={seedUsed})");

            foreach (Tensor t in converted.UNet.Values) t.Dispose();
            foreach (Tensor t in converted.ClipL.Values) t.Dispose();
            foreach (Tensor t in converted.ClipG.Values) t.Dispose();
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
                rgb[dstOff + x * 3 + 0] = bytes[srcOff + x * 3 + 2];
                rgb[dstOff + x * 3 + 1] = bytes[srcOff + x * 3 + 1];
                rgb[dstOff + x * 3 + 2] = bytes[srcOff + x * 3 + 0];
            }
        }
        return (rgb, width, absH);
    }
}
