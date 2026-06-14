using System.Diagnostics;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.ModelHandler.CheckpointConverters;
using HartsyInference.ModelHandler.SafeTensors;
using HartsyInference.Tests.Common;
using HartsyInference.Tokenizers;
using HartsyInference.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>
/// End-to-end Z-Image-Turbo / Z-Image-Base generation on the Vulkan backend. Mirrors the CUDA
/// path (<see cref="ZImageGenerationTests"/>) but swaps the backend. Phase 3.5 model-coverage
/// gate: any model that runs on CUDA should run on Vulkan unchanged via the IBackend abstraction.
///
/// Skipped when checkpoints, VAE source, Qwen3 weights, or Vulkan loader are unavailable. Saves
/// the resulting image to <c>Output/zimage_*_vulkan_*.bmp</c> alongside the CUDA reference.
/// </summary>
public sealed class ZImageVulkanGenerationTest
{
    private readonly ITestOutputHelper _output;

    public ZImageVulkanGenerationTest(ITestOutputHelper output) => _output = output;

    private static string ZImageTurboCheckpointPath => TestPaths.ZImage.Turbo;
    private static string ZImageBaseCheckpointPath => TestPaths.ZImage.Base;
    private static string FluxVaeSourcePath => TestPaths.Vae.FluxVaeSource;
    private static string Qwen3WeightsPath => TestPaths.TextEncoders.Qwen3_4B;
    private static string Qwen3VocabPath => TestPaths.Tokenizers.Qwen3Vocab;
    private static string Qwen3MergesPath => TestPaths.Tokenizers.Qwen3Merges;
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

    /// <summary>End-to-end Z-Image-Turbo on Vulkan: 512×512, 8 NFE, CFG=1.0, seed=42.
    /// Validates that an FP8Mix DiT transformer architecture other than Flux runs unchanged
    /// through the IBackend abstraction on the Vulkan path.</summary>
    [Fact]
    public void Turbo_Fp8Mix_GenerateImage_Vulkan()
    {
        if (!VulkanAvailable()) { _output.WriteLine("SKIPPED: no Vulkan device"); return; }
        if (!File.Exists(ZImageTurboCheckpointPath)) { _output.WriteLine($"SKIPPED: Z-Image-Turbo checkpoint not found: {ZImageTurboCheckpointPath}"); return; }
        if (!File.Exists(FluxVaeSourcePath)) { _output.WriteLine($"SKIPPED: Flux VAE source not found: {FluxVaeSourcePath}"); return; }
        if (!File.Exists(Qwen3WeightsPath)) { _output.WriteLine($"SKIPPED: Qwen3-4B weights not found: {Qwen3WeightsPath}"); return; }
        if (!File.Exists(Qwen3VocabPath) || !File.Exists(Qwen3MergesPath)) { _output.WriteLine("SKIPPED: Qwen3 tokenizer files not found"); return; }

        Stopwatch totalSw = Stopwatch.StartNew();
        Stopwatch sw = Stopwatch.StartNew();

        VulkanBackend backend = new();
        _output.WriteLine($"[Vulkan device] {backend.Vk}");

        const string prompt = "A photograph of an astronaut riding a horse on the moon";
        Tensor captionEmbeddings;

        // PHASE A: Qwen3 encode then dispose. Same memory pattern as the CUDA test —
        // ~16 GB Qwen3 (cast F32) + ~7 GB transformer FP8Mix exceeds available RAM if both live concurrently.
        {
            sw.Restart();
            _output.WriteLine($"[A1] Loading Qwen3-4B from: {Path.GetFileName(Qwen3WeightsPath)}");
            using SafeTensorsLoader qwenLoader = new();
            qwenLoader.Load(Qwen3WeightsPath);
            Dictionary<string, Tensor> qwenWeights = qwenLoader.GetAllTensors();
            // Vulkan backend doesn't yet support BF16 weights — pre-cast Qwen3 BF16 → F32 (matches the CUDA
            // Turbo test). Trades RAM for backend simplicity until BF16 lands as a Phase-4 work item.
            Dictionary<string, Tensor> qwenF32 = new(qwenWeights.Count);
            foreach (KeyValuePair<string, Tensor> kvp in qwenWeights)
                qwenF32[kvp.Key] = kvp.Value.DType == DType.F32 ? kvp.Value : kvp.Value.CastTo(DType.F32);
            LlamaStyleEncoder qwenEncoder = new(LlamaStyleEncoderConfig.Qwen3_4B);
            qwenEncoder.LoadWeights(qwenF32);
            _output.WriteLine($"  Qwen3-4B loaded in {sw.ElapsedMilliseconds}ms");

            using Qwen3Tokenizer tok = new(Qwen3VocabPath, Qwen3MergesPath, maxLength: 256);
            int[] tokenIds = tok.EncodeChat(prompt);
            int[] mask = Qwen3Tokenizer.CreateAttentionMask(tokenIds);
            int realLen = 0;
            for (int i = 0; i < mask.Length; i++) realLen += mask[i];
            _output.WriteLine($"[A2] Chat-templated prompt — real tokens: {realLen} of {tokenIds.Length}");

            sw.Restart();
            _output.WriteLine("[A3] Encoding prompt with Qwen3-4B (penultimate hidden state)...");
            int penultimateHfIndex = qwenEncoder.NumLayers - 1;
            Tensor encodedFull = qwenEncoder.EncodeMultiLayer(backend, [tokenIds], [penultimateHfIndex]);
            backend.Sync();
            captionEmbeddings = SliceFirstSeq(encodedFull, realLen);
            encodedFull.Dispose();
            _output.WriteLine($"  Caption embeddings: shape={captionEmbeddings.Shape}, dtype={captionEmbeddings.DType}, in {sw.ElapsedMilliseconds}ms");

            qwenEncoder.Dispose();
            foreach (Tensor t in qwenF32.Values) t.Dispose();
            qwenF32.Clear();
        }
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();

        // PHASE B: Z-Image transformer + Flux VAE
        sw.Restart();
        _output.WriteLine($"[B1] Loading Z-Image-Turbo checkpoint: {Path.GetFileName(ZImageTurboCheckpointPath)}");
        (ZImageCheckpointConverter.ConvertedWeights zConv, SafeTensorsLoader zLoader) =
            ZImageCheckpointConverter.LoadAndConvert(ZImageTurboCheckpointPath);
        _output.WriteLine($"  Loaded in {sw.ElapsedMilliseconds}ms — Transformer keys: {zConv.Transformer.Count}, FP8Mix={zConv.IsFp8Mix}");

        using (zLoader)
        {
            (int numLayers, int numRefiner, int hidden, int ffnDim, bool isFp8Mix) =
                ZImageCheckpointConverter.DetectArchitecture(zConv.Transformer);
            _output.WriteLine($"  Architecture: {numLayers} layers, {numRefiner} refiner each, hidden={hidden}, ffnDim={ffnDim}, fp8={isFp8Mix}");

            ZImageConfig zConfig = ZImageConfig.Turbo with
            {
                HiddenSize = hidden,
                NumHeads = hidden / 128,
                NumLayers = numLayers,
                NumRefinerLayers = numRefiner,
                FfnDim = ffnDim,
            };

            sw.Restart();
            _output.WriteLine("[B2] Building ZImageTransformer + loading weights (FP8 native)...");
            ZImageTransformer transformer = new(zConfig);
            transformer.LoadWeights(zConv.Transformer);
            _output.WriteLine($"  Transformer loaded in {sw.ElapsedMilliseconds}ms");

            sw.Restart();
            _output.WriteLine($"[B3] Loading Flux VAE from: {Path.GetFileName(FluxVaeSourcePath)}");
            (FluxCheckpointConverter.ConvertedWeights fluxConv, SafeTensorsLoader fluxLoader) =
                FluxCheckpointConverter.LoadAndConvert(FluxVaeSourcePath);
            using (fluxLoader)
            {
                Dictionary<string, Tensor> vaeF32 = CastWeightsToF32(fluxConv.Vae);
                VaeDecoder vaeDecoder = new(VaeConfig.ZImage);
                vaeDecoder.LoadWeights(vaeF32);
                _output.WriteLine($"  VAE loaded in {sw.ElapsedMilliseconds}ms");

                _output.WriteLine("[B4] Generating image (512×512, 8 NFE, CFG=1.0, Vulkan)...");
                ZImagePipeline pipeline = new(backend, transformer, vaeDecoder, zConfig);
                TextToImageRequest request = new()
                {
                    Prompt = prompt,
                    Width = 512,
                    Height = 512,
                    Steps = 8,
                    Seed = 42,
                };

                sw.Restart();
                (byte[] rgbData, int width, int height, int seed) = pipeline.GenerateFromEmbeddings(
                    captionEmbeddings,
                    request,
                    cfgScale: 1.0f,
                    negativeCaptionEmbeddings: null,
                    onProgress: p => _output.WriteLine($"  Step {p.Step}/{p.TotalSteps} ({p.ElapsedMs:F0}ms)"));
                backend.Sync();
                _output.WriteLine($"Generation done in {sw.ElapsedMilliseconds}ms (seed={seed})");

                Directory.CreateDirectory(OutputDir);
                string outputPath = Path.Combine(OutputDir, $"zimage_turbo_vulkan_{width}x{height}_s{request.Steps}_seed{seed}.bmp");
                ImagePostProcessor.SaveBmp(outputPath, rgbData, width, height);
                _output.WriteLine($"Saved: {outputPath}");

                Assert.Equal(512, width);
                Assert.Equal(512, height);
                Assert.Equal(512 * 512 * 3, rgbData.Length);
                ValidateImageNotDegenerate(rgbData);

                totalSw.Stop();
                _output.WriteLine($"\nTotal test time: {totalSw.ElapsedMilliseconds}ms");

                captionEmbeddings.Dispose();
                pipeline.Dispose();
                foreach (Tensor t in vaeF32.Values) t.Dispose();
            }
            foreach (Tensor t in zConv.Transformer.Values) t.Dispose();
        }
        backend.Dispose();
    }

    /// <summary>End-to-end Z-Image-Base on Vulkan: 1024×1024, 28 steps, CFG=4.0 with negative prompt.
    /// Same architecture as Turbo but un-distilled (CFG path runs cond + uncond per step).</summary>
    [Fact]
    public void Base_GenerateImage_Vulkan()
    {
        if (!VulkanAvailable()) { _output.WriteLine("SKIPPED: no Vulkan device"); return; }
        if (!File.Exists(ZImageBaseCheckpointPath)) { _output.WriteLine($"SKIPPED: Z-Image-Base checkpoint not found: {ZImageBaseCheckpointPath}"); return; }
        if (!File.Exists(FluxVaeSourcePath) || !File.Exists(Qwen3WeightsPath)
            || !File.Exists(Qwen3VocabPath) || !File.Exists(Qwen3MergesPath))
        { _output.WriteLine("SKIPPED: required dependency files missing (Flux VAE / Qwen3 weights or tokenizer)"); return; }

        Stopwatch totalSw = Stopwatch.StartNew();
        Stopwatch sw = Stopwatch.StartNew();

        VulkanBackend backend = new();
        _output.WriteLine($"[Vulkan device] {backend.Vk}");

        const string positivePrompt = "A photograph of an astronaut riding a horse on the moon";
        const string negativePrompt = "blurry, low quality, distorted, deformed, extra limbs";
        Tensor posEmb, negEmb;

        // PHASE A: encode both prompts (positive + negative for CFG), dispose Qwen3
        {
            sw.Restart();
            _output.WriteLine($"[A1] Loading Qwen3-4B from: {Path.GetFileName(Qwen3WeightsPath)}");
            using SafeTensorsLoader qwenLoader = new();
            qwenLoader.Load(Qwen3WeightsPath);
            Dictionary<string, Tensor> qwenWeights = qwenLoader.GetAllTensors();
            Dictionary<string, Tensor> qwenF32 = new(qwenWeights.Count);
            foreach (KeyValuePair<string, Tensor> kvp in qwenWeights)
                qwenF32[kvp.Key] = kvp.Value.DType == DType.F32 ? kvp.Value : kvp.Value.CastTo(DType.F32);
            LlamaStyleEncoder qwenEncoder = new(LlamaStyleEncoderConfig.Qwen3_4B);
            qwenEncoder.LoadWeights(qwenF32);
            _output.WriteLine($"  Qwen3-4B loaded in {sw.ElapsedMilliseconds}ms");

            using Qwen3Tokenizer tok = new(Qwen3VocabPath, Qwen3MergesPath, maxLength: 256);

            int[] posTokens = tok.EncodeChat(positivePrompt);
            int[] negTokens = tok.EncodeChat(negativePrompt);
            int[] posMask = Qwen3Tokenizer.CreateAttentionMask(posTokens);
            int[] negMask = Qwen3Tokenizer.CreateAttentionMask(negTokens);
            int posReal = 0; for (int i = 0; i < posMask.Length; i++) posReal += posMask[i];
            int negReal = 0; for (int i = 0; i < negMask.Length; i++) negReal += negMask[i];
            _output.WriteLine($"[A2] Tokens — pos: {posReal}/{posTokens.Length}, neg: {negReal}/{negTokens.Length}");

            sw.Restart();
            int penultimateHfIndex = qwenEncoder.NumLayers - 1;
            Tensor posEmbFull = qwenEncoder.EncodeMultiLayer(backend, [posTokens], [penultimateHfIndex]);
            Tensor negEmbFull = qwenEncoder.EncodeMultiLayer(backend, [negTokens], [penultimateHfIndex]);
            backend.Sync();
            posEmb = SliceFirstSeq(posEmbFull, posReal);
            negEmb = SliceFirstSeq(negEmbFull, negReal);
            posEmbFull.Dispose();
            negEmbFull.Dispose();
            _output.WriteLine($"  Both prompts encoded in {sw.ElapsedMilliseconds}ms");

            qwenEncoder.Dispose();
            foreach (Tensor t in qwenF32.Values) t.Dispose();
            qwenF32.Clear();
        }
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();

        sw.Restart();
        _output.WriteLine($"[B1] Loading Z-Image-Base checkpoint: {Path.GetFileName(ZImageBaseCheckpointPath)}");
        (ZImageCheckpointConverter.ConvertedWeights zConv, SafeTensorsLoader zLoader) =
            ZImageCheckpointConverter.LoadAndConvert(ZImageBaseCheckpointPath);
        _output.WriteLine($"  Loaded in {sw.ElapsedMilliseconds}ms — Transformer keys: {zConv.Transformer.Count}, FP8Mix={zConv.IsFp8Mix}");

        using (zLoader)
        {
            (int numLayers, int numRefiner, int hidden, int ffnDim, bool isFp8Mix) =
                ZImageCheckpointConverter.DetectArchitecture(zConv.Transformer);
            _output.WriteLine($"  Architecture: {numLayers} layers, {numRefiner} refiner each, hidden={hidden}, ffnDim={ffnDim}, fp8={isFp8Mix}");

            // Base uses different scheduler shift (6.0 vs Turbo 3.0). The config record carries it.
            ZImageConfig zConfig = ZImageConfig.Base with
            {
                HiddenSize = hidden,
                NumHeads = hidden / 128,
                NumLayers = numLayers,
                NumRefinerLayers = numRefiner,
                FfnDim = ffnDim,
            };

            sw.Restart();
            ZImageTransformer transformer = new(zConfig);
            transformer.LoadWeights(zConv.Transformer);
            _output.WriteLine($"[B2] Transformer loaded in {sw.ElapsedMilliseconds}ms");

            sw.Restart();
            (FluxCheckpointConverter.ConvertedWeights fluxConv, SafeTensorsLoader fluxLoader) =
                FluxCheckpointConverter.LoadAndConvert(FluxVaeSourcePath);
            using (fluxLoader)
            {
                Dictionary<string, Tensor> vaeF32 = CastWeightsToF32(fluxConv.Vae);
                VaeDecoder vaeDecoder = new(VaeConfig.ZImage);
                vaeDecoder.LoadWeights(vaeF32);
                _output.WriteLine($"[B3] VAE loaded in {sw.ElapsedMilliseconds}ms");

                _output.WriteLine("[B4] Generating image (1024×1024, 28 steps, CFG=4.0, Vulkan)...");
                ZImagePipeline pipeline = new(backend, transformer, vaeDecoder, zConfig);
                TextToImageRequest request = new()
                {
                    Prompt = positivePrompt,
                    NegativePrompt = negativePrompt,
                    Width = 1024,
                    Height = 1024,
                    Steps = 28,
                    CfgScale = 4.0f,
                    Seed = 42,
                };

                sw.Restart();
                (byte[] rgbData, int width, int height, int seed) = pipeline.GenerateFromEmbeddings(
                    posEmb,
                    request,
                    cfgScale: 4.0f,
                    negativeCaptionEmbeddings: negEmb,
                    onProgress: p => _output.WriteLine($"  Step {p.Step}/{p.TotalSteps} ({p.ElapsedMs:F0}ms)"));
                backend.Sync();
                _output.WriteLine($"Generation done in {sw.ElapsedMilliseconds}ms (seed={seed})");

                Directory.CreateDirectory(OutputDir);
                string outputPath = Path.Combine(OutputDir, $"zimage_base_vulkan_{width}x{height}_s{request.Steps}_cfg{request.CfgScale:F0}_seed{seed}.bmp");
                ImagePostProcessor.SaveBmp(outputPath, rgbData, width, height);
                _output.WriteLine($"Saved: {outputPath}");

                Assert.Equal(1024, width);
                Assert.Equal(1024, height);
                Assert.Equal(1024 * 1024 * 3, rgbData.Length);
                ValidateImageNotDegenerate(rgbData);

                totalSw.Stop();
                _output.WriteLine($"\nTotal test time: {totalSw.ElapsedMilliseconds}ms");

                posEmb.Dispose();
                negEmb.Dispose();
                pipeline.Dispose();
                foreach (Tensor t in vaeF32.Values) t.Dispose();
            }
            foreach (Tensor t in zConv.Transformer.Values) t.Dispose();
        }
        backend.Dispose();
    }

    private static unsafe Tensor SliceFirstSeq(Tensor t, int realLen)
    {
        long batch = t.Shape[0];
        long seqLen = t.Shape[1];
        long dim = t.Shape[2];
        if (realLen > seqLen)
            throw new ArgumentOutOfRangeException(nameof(realLen), $"realLen={realLen} > seqLen={seqLen}");

        TensorShape outShape = new(batch, realLen, dim);
        Tensor result = new(outShape, t.DType);
        long bytesPerToken = dim * sizeof(float);
        float* src = (float*)t.DataPointer;
        float* dst = (float*)result.DataPointer;
        for (long b = 0; b < batch; b++)
        {
            for (long s = 0; s < realLen; s++)
            {
                long srcOff = (b * seqLen + s) * dim;
                long dstOff = (b * realLen + s) * dim;
                Buffer.MemoryCopy(src + srcOff, dst + dstOff, bytesPerToken, bytesPerToken);
            }
        }
        return result;
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

    private static void ValidateImageNotDegenerate(byte[] rgb)
    {
        long sum = 0;
        long sumSq = 0;
        bool[] byteSeen = new bool[256];
        foreach (byte v in rgb) { sum += v; sumSq += (long)v * v; byteSeen[v] = true; }
        double n = rgb.Length;
        double mean = sum / n;
        double std = Math.Sqrt(Math.Max(0, sumSq / n - mean * mean));
        int distinctBytes = 0;
        for (int i = 0; i < 256; i++) if (byteSeen[i]) distinctBytes++;

        Assert.True(std > 20, $"RGB std-dev too low ({std:F1}) — image may be uniform / NaN-collapsed");
        Assert.True(distinctBytes >= 200, $"Too few distinct byte values ({distinctBytes}/256) — image may have NaN-collapsed bands");
    }
}
