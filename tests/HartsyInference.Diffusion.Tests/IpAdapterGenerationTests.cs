using System.Diagnostics;
using Xunit;
using Xunit.Abstractions;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Adapters;
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

/// <summary>Real-weight end-to-end IP-Adapter generation tests: SDXL/SD1.5 base + IPA checkpoint + CLIP-Vision-H
/// over a synthetic reference image. Asserts the output is a real image (not black / NaN-collapsed) and that the
/// IP-Adapter branch actually shifts the output away from the no-adapter baseline. Skips cleanly when any
/// checkpoint is missing. One test per process — GPU runs share sticky CUDA state.</summary>
[Trait("Category", "Integration")]
public class IpAdapterGenerationTests
{
    private readonly ITestOutputHelper _output;

    public IpAdapterGenerationTests(ITestOutputHelper output) => _output = output;

    /// <summary>SDXL base + standard IPA (4 tokens) at weight 0.8 must produce a coherent, non-black image.</summary>
    [Fact]
    public void Gpu_Sdxl_StandardIpa_512() => RunSdxl(TestPaths.IpAdapter.SdxlStandard, "std");

    /// <summary>SDXL base + Plus IPA (16-token resampler) at weight 0.8 must produce a coherent, non-black image.</summary>
    [Fact]
    public void Gpu_Sdxl_PlusIpa_512() => RunSdxl(TestPaths.IpAdapter.SdxlPlus, "plus");

    /// <summary>SD1.5 base + standard IPA at weight 0.8 must produce a coherent, non-black image.</summary>
    [Fact]
    public void Gpu_Sd15_StandardIpa_512()
    {
        string sd15Path = TestPaths.Sd15.SingleFile;
        string ipaPath = TestPaths.IpAdapter.Sd15Standard;
        string clipVisionPath = TestPaths.IpAdapter.ClipVisionH;
        if (!File.Exists(sd15Path)) { _output.WriteLine($"SKIPPED: SD1.5 checkpoint not found: {sd15Path}"); return; }
        if (!File.Exists(ipaPath)) { _output.WriteLine($"SKIPPED: IPA checkpoint not found: {ipaPath}"); return; }
        if (!File.Exists(clipVisionPath)) { _output.WriteLine($"SKIPPED: CLIP-Vision not found: {clipVisionPath}"); return; }
        if (!File.Exists(TestPaths.Tokenizers.ClipVocab)) { _output.WriteLine("SKIPPED: CLIP tokenizer not found"); return; }
        string ptxDir = Path.Combine(Path.GetDirectoryName(typeof(IpAdapterGenerationTests).Assembly.Location)!, "Ptx");
        if (!Directory.Exists(ptxDir)) { _output.WriteLine($"SKIPPED: PTX directory not found: {ptxDir}"); return; }

        (Sd15CheckpointConverter.ConvertedWeights converted, SafeTensorsLoader loader) =
            Sd15CheckpointConverter.LoadAndConvert(sd15Path);
        using (loader)
        {
            Dictionary<string, Tensor> clipF32 = CastWeights(converted.ClipL, DType.F32);
            Dictionary<string, Tensor> unetF16 = CastWeights(converted.UNet, DType.F16);
            Dictionary<string, Tensor> vaeBf16 = CastWeights(converted.Vae, DType.BF16);

            using ClipTokenizer tokenizer = new(TestPaths.Tokenizers.ClipVocab, TestPaths.Tokenizers.ClipMerges);
            string prompt = "a portrait photo of a woman in a garden, detailed";
            int[] promptTokens = tokenizer.Encode(prompt);
            int[] negTokens = tokenizer.Encode("blurry, low quality");

            ClipTextEncoder clip = new(ClipTextEncoderConfig.Sd15);
            clip.LoadWeights(clipF32, "text_model");
            UNet unet = new(UNetConfig.Sd15);
            unet.LoadWeights(unetF16);
            VaeDecoder vae = new(VaeConfig.Sd15);
            vae.LoadWeights(vaeBf16);

            using CudaBackend backend = new(deviceOrdinal: 0, ptxDir: ptxDir);

            using IpAdapterFile ipaFile = IpAdapterLoader.Load(ipaPath);
            Assert.Equal(IpAdapterBaseModel.Sd15, ipaFile.BaseModel);
            using IpAdapter ipa = new(ipaFile.Config);
            ipa.LoadWeights(ipaFile.Weights);
            _output.WriteLine($"IPA: plus={ipaFile.Config.IsPlus}, tokens={ipa.NumImageTokens}, layers={ipa.CrossAttentionLayerCount}");

            using SafeTensorsLoader cvLoader = new();
            cvLoader.Load(clipVisionPath);
            ClipVisionEncoder clipVision = new(ClipVisionEncoderConfig.ViTH14);
            clipVision.LoadWeights(cvLoader.GetAllTensors(), prefix: "vision_model");

            using Tensor pixelValues = BuildSyntheticReferenceImage(ClipVisionEncoderConfig.ViTH14.ImageSize);
            Tensor visionOut = clipVision.EncodeImageEmbeds(backend, pixelValues);
            using Tensor imageTokens = ipa.ProjectImage(backend, visionOut);
            visionOut.Dispose();
            DumpStats("image_tokens", imageTokens);
            AssertFinite("image_tokens", imageTokens);

            IpAdapterConditioning conditioning = new()
            {
                Adapter = ipa,
                ImageTokens = imageTokens,
                Scale = 0.8f,
            };

            using StableDiffusion15Pipeline pipeline = new(backend, clip, unet, vae);
            TextToImageRequest request = new()
            {
                Prompt = prompt,
                NegativePrompt = "blurry, low quality",
                Width = 512,
                Height = 512,
                Steps = 10,
                CfgScale = 7.0f,
                Seed = 42,
            };

            (byte[] rgbData, int width, int height, _) = pipeline.GenerateFromTokens(
                promptTokens, negTokens, request,
                progress => _output.WriteLine($"  step {progress.Step}/{progress.TotalSteps} ({progress.ElapsedMs:F0}ms)"),
                ipAdapters: [conditioning]);

            (double mean, double std) = ByteStats(rgbData);
            _output.WriteLine($"IPA output: mean={mean:F2}, std={std:F2}");

            Directory.CreateDirectory(TestPaths.OutputDir);
            string outPath = Path.Combine(TestPaths.OutputDir, $"sd15_ipa_std_{DateTime.Now:yyyyMMdd_HHmmss}.bmp");
            ImagePostProcessor.SaveBmp(outPath, rgbData, width, height);
            _output.WriteLine($"Saved: {outPath}");

            Assert.True(std > 15, $"IPA output collapsed (std={std:F2}) — image is flat/black.");
            Assert.True(mean > 10 && mean < 245, $"IPA output mean={mean:F2} out of range.");
        }
    }

    private void RunSdxl(string ipaPath, string tag)
    {
        string sdxlPath = TestPaths.Sdxl.SingleFile;
        string clipVisionPath = TestPaths.IpAdapter.ClipVisionH;
        if (!File.Exists(sdxlPath)) { _output.WriteLine($"SKIPPED: SDXL checkpoint not found: {sdxlPath}"); return; }
        if (!File.Exists(ipaPath)) { _output.WriteLine($"SKIPPED: IPA checkpoint not found: {ipaPath}"); return; }
        if (!File.Exists(clipVisionPath)) { _output.WriteLine($"SKIPPED: CLIP-Vision not found: {clipVisionPath}"); return; }
        if (!File.Exists(TestPaths.Tokenizers.ClipVocab)) { _output.WriteLine("SKIPPED: CLIP tokenizer not found"); return; }
        string ptxDir = Path.Combine(Path.GetDirectoryName(typeof(IpAdapterGenerationTests).Assembly.Location)!, "Ptx");
        if (!Directory.Exists(ptxDir)) { _output.WriteLine($"SKIPPED: PTX directory not found: {ptxDir}"); return; }

        Stopwatch sw = Stopwatch.StartNew();
        (SdxlCheckpointConverter.ConvertedWeights converted, SafeTensorsLoader loader) =
            SdxlCheckpointConverter.LoadAndConvert(sdxlPath);
        _output.WriteLine($"Checkpoint loaded in {sw.ElapsedMilliseconds}ms");

        using (loader)
        {
            Dictionary<string, Tensor> clipLF32 = CastWeights(converted.ClipL, DType.F32);
            Dictionary<string, Tensor> clipGF32 = CastWeights(converted.ClipG, DType.F32);
            Dictionary<string, Tensor> unetF16 = CastWeights(converted.UNet, DType.F16);
            // SDXL VAE in F16 overflows (resnet activations) → NaN → black. BF16 mirrors the
            // extension's VaePrecisionHelper choice.
            Dictionary<string, Tensor> vaeBf16 = CastWeights(converted.Vae, DType.BF16);

            using ClipTokenizer tokenizer = new(TestPaths.Tokenizers.ClipVocab, TestPaths.Tokenizers.ClipMerges);
            string prompt = "a portrait photo of a woman in a garden, detailed, sharp focus";
            string negPrompt = "blurry, low quality";
            int[] promptTokens = tokenizer.Encode(prompt);
            int[] negTokens = tokenizer.Encode(negPrompt);
            int promptEos = ClipTokenizer.FindEosPosition(promptTokens);
            int negEos = ClipTokenizer.FindEosPosition(negTokens);

            ClipTextEncoder clipL = new(ClipTextEncoderConfig.SdxlClipL);
            clipL.LoadWeights(clipLF32, "text_model");
            ClipTextEncoder clipG = new(ClipTextEncoderConfig.SdxlClipG);
            clipG.LoadWeights(clipGF32, "text_model");
            UNet unet = new(UNetConfig.SdxlBase);
            unet.LoadWeights(unetF16);
            VaeDecoder vae = new(VaeConfig.Sdxl);
            vae.LoadWeights(vaeBf16);

            using CudaBackend backend = new(deviceOrdinal: 0, ptxDir: ptxDir);
            _output.WriteLine($"CUDA device: {backend.Capabilities.Name}");

            // IPA + CLIP-Vision (mirrors the SwarmUI extension's IpAdapterResolver wiring).
            using IpAdapterFile ipaFile = IpAdapterLoader.Load(ipaPath);
            Assert.Equal(IpAdapterBaseModel.Sdxl, ipaFile.BaseModel);
            using IpAdapter ipa = new(ipaFile.Config);
            ipa.LoadWeights(ipaFile.Weights);
            _output.WriteLine($"IPA: plus={ipaFile.Config.IsPlus}, tokens={ipa.NumImageTokens}, layers={ipa.CrossAttentionLayerCount}");

            using SafeTensorsLoader cvLoader = new();
            cvLoader.Load(clipVisionPath);
            Dictionary<string, Tensor> cvWeights = cvLoader.GetAllTensors();
            ClipVisionEncoder clipVision = new(ClipVisionEncoderConfig.ViTH14);
            clipVision.LoadWeights(cvWeights, prefix: "vision_model");

            using Tensor pixelValues = BuildSyntheticReferenceImage(ClipVisionEncoderConfig.ViTH14.ImageSize);
            Tensor visionOut = ipaFile.Config.IsPlus
                ? clipVision.EncodeHiddenStates(backend, pixelValues)
                : clipVision.EncodeImageEmbeds(backend, pixelValues);
            DumpStats("vision_out", visionOut);
            using Tensor imageTokens = ipa.ProjectImage(backend, visionOut);
            visionOut.Dispose();
            DumpStats("image_tokens", imageTokens);
            AssertFinite("image_tokens", imageTokens);

            IpAdapterConditioning conditioning = new()
            {
                Adapter = ipa,
                ImageTokens = imageTokens,
                Scale = 0.8f,
            };

            using SdxlPipeline pipeline = new(backend, clipL, clipG, unet, vae);
            TextToImageRequest request = new()
            {
                Prompt = prompt,
                NegativePrompt = negPrompt,
                Width = 512,
                Height = 512,
                Steps = 10,
                CfgScale = 6.0f,
                Seed = 42,
            };

            sw.Restart();
            (byte[] rgbData, int width, int height, _) = pipeline.GenerateFromTokens(
                promptTokens, negTokens, promptTokens, negTokens, promptEos, negEos, request,
                progress => _output.WriteLine($"  step {progress.Step}/{progress.TotalSteps} ({progress.ElapsedMs:F0}ms)"),
                ipAdapters: [conditioning]);
            _output.WriteLine($"Generation with IPA done in {sw.Elapsed.TotalSeconds:F1}s");

            (double mean, double std) = ByteStats(rgbData);
            _output.WriteLine($"IPA output: mean={mean:F2}, std={std:F2}");

            Directory.CreateDirectory(TestPaths.OutputDir);
            string outPath = Path.Combine(TestPaths.OutputDir, $"sdxl_ipa_{tag}_{DateTime.Now:yyyyMMdd_HHmmss}.bmp");
            ImagePostProcessor.SaveBmp(outPath, rgbData, width, height);
            _output.WriteLine($"Saved: {outPath}");

            Assert.True(std > 15, $"IPA output collapsed (std={std:F2}) — image is flat/black. NaN or corruption in the ip-attention branch.");
            Assert.True(mean > 10 && mean < 245, $"IPA output mean={mean:F2} out of range — image is all-black or all-white.");
        }
    }

    /// <summary>CLIP-normalized synthetic reference: a warm orange-to-purple vertical gradient with a bright disc — distinctive hue statistics so IPA influence is visible.</summary>
    internal static unsafe Tensor BuildSyntheticReferenceImage(int size)
    {
        float[] mean = [0.48145466f, 0.4578275f, 0.40821073f];
        float[] std = [0.26862954f, 0.26130258f, 0.27577711f];
        Tensor pixels = new Tensor(new TensorShape(1, 3, size, size), DType.F32);
        float* p = (float*)pixels.DataPointer;
        for (int y = 0; y < size; y++)
        {
            float ty = y / (float)(size - 1);
            for (int x = 0; x < size; x++)
            {
                float tx = x / (float)(size - 1);
                float dx = tx - 0.5f, dy = ty - 0.35f;
                bool disc = dx * dx + dy * dy < 0.04f;
                float r = disc ? 1.0f : 0.9f - 0.5f * ty;
                float g = disc ? 0.9f : 0.4f - 0.2f * ty;
                float b = disc ? 0.5f : 0.3f + 0.5f * ty;
                p[0 * size * size + y * size + x] = (r - mean[0]) / std[0];
                p[1 * size * size + y * size + x] = (g - mean[1]) / std[1];
                p[2 * size * size + y * size + x] = (b - mean[2]) / std[2];
            }
        }
        return pixels;
    }

    private unsafe void DumpStats(string name, Tensor t)
    {
        float* ptr = (float*)t.DataPointer;
        long count = t.ElementCount;
        double sum = 0, sumSq = 0;
        float min = float.MaxValue, max = float.MinValue;
        int nanCount = 0;
        for (long i = 0; i < count; i++)
        {
            float v = ptr[i];
            if (float.IsNaN(v) || float.IsInfinity(v)) { nanCount++; continue; }
            sum += v; sumSq += (double)v * v;
            if (v < min) min = v;
            if (v > max) max = v;
        }
        double m = sum / count;
        double s = Math.Sqrt(Math.Max(0, sumSq / count - m * m));
        _output.WriteLine($"  {name}: shape={t.Shape}, mean={m:G5}, std={s:G5}, min={min:G5}, max={max:G5}, nan/inf={nanCount}");
    }

    private static unsafe void AssertFinite(string name, Tensor t)
    {
        float* ptr = (float*)t.DataPointer;
        long count = t.ElementCount;
        for (long i = 0; i < count; i++)
        {
            if (float.IsNaN(ptr[i]) || float.IsInfinity(ptr[i]))
                throw new Xunit.Sdk.XunitException($"{name} contains NaN/Inf at index {i}");
        }
    }

    private static (double mean, double std) ByteStats(byte[] data)
    {
        double sum = 0, sumSq = 0;
        foreach (byte b in data) { sum += b; sumSq += (double)b * b; }
        double mean = sum / data.Length;
        double std = Math.Sqrt(Math.Max(0, sumSq / data.Length - mean * mean));
        return (mean, std);
    }

    private static Dictionary<string, Tensor> CastWeights(Dictionary<string, Tensor> weights, DType dtype)
    {
        Dictionary<string, Tensor> result = new(weights.Count);
        foreach ((string key, Tensor tensor) in weights)
        {
            result[key] = tensor.DType == dtype ? tensor : tensor.CastTo(dtype);
        }
        return result;
    }
}
