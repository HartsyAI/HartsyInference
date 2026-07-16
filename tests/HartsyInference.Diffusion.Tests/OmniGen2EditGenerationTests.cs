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
using HartsyInference.ModelHandler.CheckpointConverters;
using HartsyInference.ModelHandler.CheckpointConverters.Utils;
using HartsyInference.ModelHandler.SafeTensors;
using HartsyInference.Tests.Common;
using HartsyInference.Tokenizers;
using HartsyInference.Vision.Codec;

namespace HartsyInference.Diffusion.Tests;

/// <summary>End-to-end OmniGen 2 reference-image edit (text + image → image): live Qwen2.5-VL-3B text
/// conditioning (the upstream edit pipeline is text-only through the mllm — no vision tokens), the reference
/// image VAE-encoded into the transformer's token stream, and the reference triple-pass dual guidance
/// <c>pred = uncond + ig·(refPred − uncond) + tg·(cond − refPred)</c>. Skips cleanly when the transformer, VAE,
/// text encoder, or reference PNG (env <c>OMNIGEN2_EDIT_REF_PNG</c>) is missing. Sized for a 12 GB GPU: the
/// 7 GB fp16 text encoder is staged in/out before the 8 GB BF16 DiT preloads, and the edit runs at 512².</summary>
public sealed class OmniGen2EditGenerationTests
{
    private static string OutputDir => TestPaths.OutputDir;
    private readonly ITestOutputHelper _output;

    public OmniGen2EditGenerationTests(ITestOutputHelper output) => _output = output;

    /// <summary>ComfyUI's OmniGen2 system prompt, verbatim (comfy/text_encoders/omnigen2.py llama_template).</summary>
    private const string SystemPrompt =
        "You are a helpful assistant that generates high-quality images based on user instructions.";

    [Fact]
    [Trait("Category", "GpuIntegration")]
    [Trait("Category", "Slow")]
    public void OmniGen2_V1_Gpu_512_Edit()
    {
        string refPngPath = Environment.GetEnvironmentVariable("OMNIGEN2_EDIT_REF_PNG") ?? "";
        string prompt = Environment.GetEnvironmentVariable("OMNIGEN2_EDIT_PROMPT")
            ?? "change the apple to a green apple";
        if (string.IsNullOrEmpty(refPngPath) || !File.Exists(refPngPath))
        {
            _output.WriteLine("SKIPPED: set OMNIGEN2_EDIT_REF_PNG to an RGB PNG reference image.");
            return;
        }
        if (!File.Exists(TestPaths.OmniGen2.Transformer))
        {
            _output.WriteLine($"SKIPPED: OmniGen 2 transformer not found: {TestPaths.OmniGen2.Transformer}");
            return;
        }
        if (!File.Exists(TestPaths.OmniGen2.Vae))
        {
            _output.WriteLine($"SKIPPED: VAE not found: {TestPaths.OmniGen2.Vae}");
            return;
        }
        if (!File.Exists(TestPaths.OmniGen2.TextEncoder))
        {
            _output.WriteLine($"SKIPPED: Qwen2.5-VL-3B text encoder not found: {TestPaths.OmniGen2.TextEncoder}");
            return;
        }

        string assemblyDir = Path.GetDirectoryName(typeof(OmniGen2EditGenerationTests).Assembly.Location)!;
        string ptxDir = Path.Combine(assemblyDir, "Ptx");
        if (!Directory.Exists(ptxDir))
        {
            _output.WriteLine($"SKIPPED: PTX directory not found: {ptxDir}");
            return;
        }

        Stopwatch totalSw = Stopwatch.StartNew();
        Stopwatch sw = Stopwatch.StartNew();

        // Reference image → the OmniGen2ImageProcessor resize (ratio = min(sqrt(maxPixels/cur), maxSide/long, 1),
        // dims floored to multiples of 16). maxPixels = 512² keeps the joint sequence 3060-sized; the upstream
        // pipeline exposes exactly this knob (`max_pixels`).
        byte[] pngBytes = File.ReadAllBytes(refPngPath);
        (byte[] refRgb, int refW, int refH) = PngDecoder.Decode(pngBytes);
        (int outH, int outW) = ComputeReferenceSize(refH, refW, maxPixels: 512 * 512, maxSideLength: 1024);
        byte[] resized = ResizeBilinear(refRgb, refW, refH, outW, outH);
        using Tensor refImage = ImagePostProcessor.RgbBytesToTensor(resized, outW, outH);
        _output.WriteLine($"Reference: {refW}x{refH} → {outW}x{outH} (align_res output size)");

        _output.WriteLine($"[1/6] Loading + converting transformer: {Path.GetFileName(TestPaths.OmniGen2.Transformer)}");
        (OmniGen2CheckpointConverter.ConvertedWeights converted, SafeTensorsLoader loader) =
            OmniGen2CheckpointConverter.LoadAndConvert(TestPaths.OmniGen2.Transformer);
        _output.WriteLine($"  Loaded {converted.Transformer.Count} keys in {sw.ElapsedMilliseconds}ms");

        using (loader)
        {
            Dictionary<string, Tensor> transformerWeights = CastWeightsToBf16(converted.Transformer);

            sw.Restart();
            OmniGen2Config config = OmniGen2Config.V1;
            using OmniGen2Transformer transformer = new(config);
            transformer.LoadWeights(transformerWeights);
            if (!transformer.HasRefStack)
            {
                _output.WriteLine("SKIPPED: checkpoint has no ref_image_patch_embedder / ref_image_refiner weights.");
                return;
            }
            _output.WriteLine($"[2/6] Transformer ready in {sw.ElapsedMilliseconds}ms (ref stack loaded)");

            sw.Restart();
            using SafeTensorsLoader vaeLoader = new();
            vaeLoader.Load(TestPaths.OmniGen2.Vae);
            Dictionary<string, Tensor> vaeWeights = ConvertVaeWeights(vaeLoader.GetAllTensors());
            VaeDecoder vaeDecoder = new(VaeConfig.Flux);
            vaeDecoder.LoadWeights(vaeWeights);
            VaeEncoder vaeEncoder = new(VaeConfig.Flux);
            vaeEncoder.LoadWeights(vaeWeights);
            _output.WriteLine($"[3/6] VAE (encoder + decoder) ready in {sw.ElapsedMilliseconds}ms");

            sw.Restart();
            using CudaBackend backend = new(deviceOrdinal: 0, ptxDir: ptxDir);
            (nuint freeBytes, nuint totalBytes) = backend.Context.GetMemoryInfo();
            double freeGb = freeBytes / (1024.0 * 1024.0 * 1024.0);
            const double MinRequiredGb = 9.5;
            if (freeGb < MinRequiredGb)
            {
                _output.WriteLine($"SKIPPED: only {freeGb:F1} GB free VRAM (total {totalBytes / (1024.0 * 1024.0 * 1024.0):F1} GB); need ≥{MinRequiredGb} GB.");
                return;
            }

            // Live Qwen2.5-VL-3B text conditioning (staged BEFORE the DiT preload — the 7 GB fp16 encoder and
            // the 8 GB BF16 DiT cannot coexist on a 12 GB card).
            sw.Restart();
            using SafeTensorsLoader teLoader = new();
            teLoader.Load(TestPaths.OmniGen2.TextEncoder);
            using LlamaStyleEncoder textEncoder = new(LlamaStyleEncoderConfig.Qwen2_5_VL_3B);
            textEncoder.LoadWeights(teLoader.GetAllTensors());
            using Qwen3Tokenizer tokenizer = new(maxLength: 512);
            Tensor condEmbeds = textEncoder.Encode(backend, [EncodeWithTemplate(tokenizer, prompt)]);
            Tensor negEmbeds = textEncoder.Encode(backend, [EncodeWithTemplate(tokenizer, "")]);
            unsafe
            {
                _ = condEmbeds.DataPointer;   // host-materialize: survives the FreeActivations below
                _ = negEmbeds.DataPointer;
            }
            backend.Sync();
            backend.FreeWeights(textEncoder.EnumerateWeights());
            backend.FreeActivations();
            _output.WriteLine($"[4/6] Qwen2.5-VL-3B conditioning ({condEmbeds.Shape[1]} tokens) in {sw.ElapsedMilliseconds}ms");

            using OmniGen2Pipeline pipeline = new(backend, transformer, vaeDecoder, vaeEncoder, config);

            TextToImageRequest request = new()
            {
                Prompt = prompt,
                NegativePrompt = "",
                Width = outW,
                Height = outH,
                Steps = 20,
                CfgScale = 5.0f,
                Seed = 42,
            };

            _output.WriteLine($"\n[5/6] Edit {outW}x{outH}, 20 steps, tg=5.0, ig=2.0, seed=42...");
            Stopwatch genSw = Stopwatch.StartNew();
            (byte[] rgb, int genW, int genH, int seed) = pipeline.EditFromEmbeddings(
                condEmbeds, negEmbeds, [refImage], request,
                textGuidanceScale: 5.0f, imageGuidanceScale: 2.0f,
                onProgress: progress => _output.WriteLine($"  Step {progress.Step}/{progress.TotalSteps} ({progress.ElapsedMs:F0}ms)"));
            genSw.Stop();
            condEmbeds.Dispose();
            negEmbeds.Dispose();
            _output.WriteLine($"\n[6/6] Edit complete in {genSw.Elapsed.TotalSeconds:F1}s (seed={seed})");

            Assert.Equal(outW, genW);
            Assert.Equal(outH, genH);
            Assert.Equal(genW * genH * 3, rgb.Length);
            ValidateImageNotDegenerate(rgb);
            LogChannelShift(resized, rgb);

            Directory.CreateDirectory(OutputDir);
            string outputPath = Path.Combine(OutputDir, $"omnigen2_v1_512_edit_{DateTime.Now:yyyyMMdd_HHmmss}.bmp");
            ImagePostProcessor.SaveBmp(outputPath, rgb, genW, genH);
            _output.WriteLine($"  Saved: {outputPath}");

            totalSw.Stop();
            _output.WriteLine($"\nTotal: {totalSw.Elapsed.TotalSeconds:F1}s");
        }
    }

    /// <summary>ComfyUI's OmniGen2 chat template (system block kept, no assistant prefix) — the exact
    /// tokenization <c>OmniGen2Loader.EncodeWithTemplate</c> uses in the SwarmUI extension.</summary>
    private static int[] EncodeWithTemplate(Qwen3Tokenizer tokenizer, string prompt)
    {
        List<int> ids = new(96);
        ids.Add(Qwen3Tokenizer.ImStartId);
        ids.AddRange(tokenizer.EncodeRaw("system\n" + SystemPrompt));
        ids.Add(Qwen3Tokenizer.ImEndId);
        ids.AddRange(tokenizer.EncodeRaw("\n"));
        ids.Add(Qwen3Tokenizer.ImStartId);
        ids.AddRange(tokenizer.EncodeRaw("user\n" + prompt));
        ids.Add(Qwen3Tokenizer.ImEndId);
        ids.AddRange(tokenizer.EncodeRaw("\n"));
        return [.. ids];
    }

    /// <summary>OmniGen2ImageProcessor.get_new_height_width: <c>ratio = min(sqrt(maxPixels/cur), maxSide/long, 1)</c>,
    /// dims floored to multiples of 16 (vae_scale_factor 8 × patch 2).</summary>
    private static (int h, int w) ComputeReferenceSize(int height, int width, int maxPixels, int maxSideLength)
    {
        double maxSideRatio = height > width ? (double)maxSideLength / height : (double)maxSideLength / width;
        double maxPixelsRatio = Math.Sqrt((double)maxPixels / ((double)height * width));
        double ratio = Math.Min(Math.Min(maxPixelsRatio, maxSideRatio), 1.0);
        int newH = (int)(height * ratio) / 16 * 16;
        int newW = (int)(width * ratio) / 16 * 16;
        return (Math.Max(16, newH), Math.Max(16, newW));
    }

    private static byte[] ResizeBilinear(byte[] src, int srcW, int srcH, int dstW, int dstH)
    {
        if (srcW == dstW && srcH == dstH) return src;
        byte[] dst = new byte[dstW * dstH * 3];
        for (int y = 0; y < dstH; y++)
        {
            float fy = (y + 0.5f) * srcH / dstH - 0.5f;
            int y0 = Math.Clamp((int)MathF.Floor(fy), 0, srcH - 1);
            int y1 = Math.Min(y0 + 1, srcH - 1);
            float wy = fy - y0;
            for (int x = 0; x < dstW; x++)
            {
                float fx = (x + 0.5f) * srcW / dstW - 0.5f;
                int x0 = Math.Clamp((int)MathF.Floor(fx), 0, srcW - 1);
                int x1 = Math.Min(x0 + 1, srcW - 1);
                float wx = fx - x0;
                for (int c = 0; c < 3; c++)
                {
                    float top = src[(y0 * srcW + x0) * 3 + c] * (1 - wx) + src[(y0 * srcW + x1) * 3 + c] * wx;
                    float bot = src[(y1 * srcW + x0) * 3 + c] * (1 - wx) + src[(y1 * srcW + x1) * 3 + c] * wx;
                    dst[(y * dstW + x) * 3 + c] = (byte)Math.Clamp(top * (1 - wy) + bot * wy + 0.5f, 0, 255);
                }
            }
        }
        return dst;
    }

    /// <summary>Logs the red→green channel shift between the reference and the edit output — evidence for
    /// prompts like "change the apple to a green apple" (the output's G−R mean should rise vs the input's).</summary>
    private void LogChannelShift(byte[] refRgb, byte[] outRgb)
    {
        (double refR, double refG) = MeanRg(refRgb);
        (double outR, double outG) = MeanRg(outRgb);
        _output.WriteLine($"  Channel means — ref: R={refR:F1} G={refG:F1} (G−R={refG - refR:F1}); " +
            $"out: R={outR:F1} G={outG:F1} (G−R={outG - outR:F1})");
    }

    private static (double r, double g) MeanRg(byte[] rgb)
    {
        double r = 0, g = 0;
        long pixels = rgb.Length / 3;
        for (long i = 0; i < pixels; i++)
        {
            r += rgb[i * 3];
            g += rgb[i * 3 + 1];
        }
        return (r / pixels, g / pixels);
    }

    private static Dictionary<string, Tensor> ConvertVaeWeights(Dictionary<string, Tensor> weights)
    {
        Dictionary<string, Tensor> result = new(weights.Count);
        foreach (KeyValuePair<string, Tensor> kvp in weights)
        {
            string? diffusersKey = CheckpointConvertUtils.ConvertVaeKey(kvp.Key);
            if (diffusersKey is null)
                continue;
            DType dt = kvp.Value.DType;
            result[diffusersKey] = (dt == DType.F16 || dt == DType.BF16) ? kvp.Value.CastTo(DType.F32) : kvp.Value;
        }
        return result;
    }

    private static Dictionary<string, Tensor> CastWeightsToBf16(Dictionary<string, Tensor> weights)
    {
        Dictionary<string, Tensor> bf16 = new(weights.Count);
        foreach (KeyValuePair<string, Tensor> kvp in weights)
        {
            DType dt = kvp.Value.DType;
            bf16[kvp.Key] = (dt == DType.F16 || dt == DType.F32) ? kvp.Value.CastTo(DType.BF16) : kvp.Value;
        }
        return bf16;
    }

    private void ValidateImageNotDegenerate(byte[] rgb)
    {
        int nonZero = 0, nonFF = 0;
        foreach (byte b in rgb)
        {
            if (b != 0) nonZero++;
            if (b != 255) nonFF++;
        }
        float nzPct = nonZero / (float)rgb.Length * 100;
        float nffPct = nonFF / (float)rgb.Length * 100;
        _output.WriteLine($"  Non-zero: {nzPct:F1}%, Non-255: {nffPct:F1}%");
        Assert.True(nzPct > 10, "Image appears all-black");
        Assert.True(nffPct > 10, "Image appears all-white");
    }
}
