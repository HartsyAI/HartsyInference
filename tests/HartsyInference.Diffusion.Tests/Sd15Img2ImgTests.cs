using System.Diagnostics;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.ModelHandler.CheckpointConverters;
using HartsyInference.Tests.Common;
using HartsyInference.Tokenizers;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>
/// Integration tests for SD1.5 image-to-image (encoder + AddNoise + partial denoise + decoder).
/// Covers: pipeline wiring (img2img without VaeEncoder throws); strength=0 short-circuits to source pass-through;
/// end-to-end CPU run with strength=0.6 produces a valid output image.
///
/// All checkpoint-dependent tests skip cleanly when <c>SD15_SINGLE_FILE_PATH</c> / CLIP tokenizer files are absent.
/// </summary>
public sealed class Sd15Img2ImgTests
{
    private readonly ITestOutputHelper _output;

    public Sd15Img2ImgTests(ITestOutputHelper output) => _output = output;

    private static string Sd15SingleFilePath => TestPaths.Sd15.SingleFile;
    private static string TokenizerVocabPath => TestPaths.Tokenizers.ClipVocab;
    private static string TokenizerMergesPath => TestPaths.Tokenizers.ClipMerges;
    private static string OutputDir => TestPaths.OutputDir;

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

    // ── Wiring tests (no checkpoint required) ───────────────────────────

    [Fact]
    public void GenerateImg2Img_WithoutVaeEncoder_Throws()
    {
        // Without a VaeEncoder, img2img must fail loudly rather than silently doing the wrong thing.
        // We can verify this without loading any checkpoint by pre-checking the constructor path.
        // Construct a minimal pipeline through the legacy ctor that does NOT take a VaeEncoder.
        using CpuBackend backend = new();
        ClipTextEncoder textEncoder = new(ClipTextEncoderConfig.Sd15);
        UNet unet = new(UNetConfig.Sd15);
        VaeDecoder vaeDecoder = new(VaeConfig.Sd15);

        using StableDiffusion15Pipeline pipeline = new(backend, textEncoder, unet, vaeDecoder);

        Tensor source = new Tensor(new TensorShape(1, 3, 64, 64), DType.F32);
        ImageToImageRequest request = new()
        {
            Prompt = "test",
            Width = 64,
            Height = 64,
            SourceImage = source,
        };

        Assert.Throws<InvalidOperationException>(() =>
            pipeline.GenerateFromTokens(promptTokenIds: [], negativePromptTokenIds: [], request));

        source.Dispose();
    }

    [Fact]
    public void GenerateImg2Img_WrongSourceShape_Throws()
    {
        using CpuBackend backend = new();
        ClipTextEncoder textEncoder = new(ClipTextEncoderConfig.Sd15);
        UNet unet = new(UNetConfig.Sd15);
        VaeDecoder vaeDecoder = new(VaeConfig.Sd15);
        VaeEncoder vaeEncoder = new(VaeConfig.Sd15);

        using StableDiffusion15Pipeline pipeline = new(backend, textEncoder, unet, vaeDecoder, vaeEncoder);

        // Source shape doesn't match the request resolution → ArgumentException.
        Tensor source = new Tensor(new TensorShape(1, 3, 32, 32), DType.F32);
        ImageToImageRequest request = new()
        {
            Prompt = "test",
            Width = 64,
            Height = 64,
            SourceImage = source,
        };

        Assert.Throws<ArgumentException>(() =>
            pipeline.GenerateFromTokens(promptTokenIds: [], negativePromptTokenIds: [], request));

        source.Dispose();
    }

    // ── Strength=0 short-circuit (no checkpoint required) ──────────────

    [Fact]
    public void GenerateImg2Img_Strength0_PassesSourceThrough()
    {
        // Strength=0 short-circuits before hitting any model code, so no real weights are needed.
        using CpuBackend backend = new();
        ClipTextEncoder textEncoder = new(ClipTextEncoderConfig.Sd15);
        UNet unet = new(UNetConfig.Sd15);
        VaeDecoder vaeDecoder = new(VaeConfig.Sd15);
        VaeEncoder vaeEncoder = new(VaeConfig.Sd15);

        using StableDiffusion15Pipeline pipeline = new(backend, textEncoder, unet, vaeDecoder, vaeEncoder);

        // Build a recognizable RGB pattern (gradient) and feed it as a pre-normalized tensor.
        const int w = 64, h = 64;
        byte[] sourceBytes = new byte[w * h * 3];
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                int o = (y * w + x) * 3;
                sourceBytes[o + 0] = (byte)(x * 4);   // R: horizontal gradient
                sourceBytes[o + 1] = (byte)(y * 4);   // G: vertical gradient
                sourceBytes[o + 2] = (byte)((x + y) * 2); // B: diagonal
            }

        Tensor source = ImagePostProcessor.RgbBytesToTensor(sourceBytes, w, h);

        ImageToImageRequest request = new()
        {
            Prompt = "ignored",
            Width = w,
            Height = h,
            Steps = 10,
            CfgScale = 7.5f,
            Seed = 42,
            SourceImage = source,
            Strength = 0.0f,
        };

        (byte[] outBytes, int outW, int outH, int seed) = pipeline.GenerateFromTokens(
            promptTokenIds: [], negativePromptTokenIds: [], request);

        Assert.Equal(w, outW);
        Assert.Equal(h, outH);
        Assert.Equal(42, seed);
        // Strength=0 must round-trip the bytes exactly (no encode/decode loss).
        Assert.Equal(sourceBytes, outBytes);
        source.Dispose();
    }

    // ── End-to-end CPU img2img with real SD1.5 weights ────────────────

    [Fact]
    public void GenerateImg2Img_EndToEnd_Sd15Cpu_ProducesValidImage()
    {
        if (!File.Exists(Sd15SingleFilePath)) { _output.WriteLine($"SKIPPED: SD1.5 checkpoint not found: {Sd15SingleFilePath}"); return; }
        if (!File.Exists(TokenizerVocabPath) || !File.Exists(TokenizerMergesPath))
        { _output.WriteLine("SKIPPED: CLIP tokenizer files not found"); return; }

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

            using CpuBackend backend = new();

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

            _output.WriteLine("[4/6] Loading VAE (encoder + decoder)...");
            sw.Restart();
            VaeDecoder vaeDecoder = new(VaeConfig.Sd15);
            vaeDecoder.LoadWeights(vaeF32);
            VaeEncoder vaeEncoder = new(VaeConfig.Sd15);
            vaeEncoder.LoadWeights(vaeF32);
            sw.Stop();
            _output.WriteLine($"  VAE loaded in {sw.ElapsedMilliseconds}ms");

            _output.WriteLine("[5/6] Building synthetic source image (64x64 gradient)...");
            // Use a small resolution to keep CPU end-to-end runtime tractable. SD1.5's UNet doesn't strictly require 512×512;
            // 64×64 is divisible by 8 (latent 8×8) and is the minimum that exercises every layer of the encoder/decoder.
            const int W = 64, H = 64;
            byte[] sourceBytes = new byte[W * H * 3];
            for (int y = 0; y < H; y++)
                for (int x = 0; x < W; x++)
                {
                    int o = (y * W + x) * 3;
                    sourceBytes[o + 0] = (byte)((x * 255) / W);
                    sourceBytes[o + 1] = (byte)((y * 255) / H);
                    sourceBytes[o + 2] = (byte)(((x + y) * 255) / (W + H));
                }
            Tensor source = ImagePostProcessor.RgbBytesToTensor(sourceBytes, W, H);

            _output.WriteLine("[6/6] Tokenizing & generating (CPU, 4 steps, strength=0.6)...");
            using ClipTokenizer tokenizer = new(TokenizerVocabPath, TokenizerMergesPath);
            const string prompt = "a colorful abstract painting";
            const string negativePrompt = "";
            int[] promptTokens = tokenizer.Encode(prompt);
            int[] negativeTokens = tokenizer.Encode(negativePrompt);

            ImageToImageRequest request = new()
            {
                Prompt = prompt,
                NegativePrompt = negativePrompt,
                Width = W,
                Height = H,
                // Few steps to keep CPU runtime reasonable; the test verifies wiring + non-degenerate output, not visual quality.
                Steps = 4,
                CfgScale = 7.5f,
                Seed = 42,
                SourceImage = source,
                Strength = 0.6f,
            };

            using StableDiffusion15Pipeline pipeline = new(backend, clipL, unet, vaeDecoder, vaeEncoder);

            sw.Restart();
            (byte[] outBytes, int outW, int outH, int seed) = pipeline.GenerateFromTokens(
                promptTokens, negativeTokens, request,
                onProgress: p => _output.WriteLine($"  Step {p.Step}/{p.TotalSteps} ({p.ElapsedMs:F0}ms)"));
            sw.Stop();
            _output.WriteLine($"Img2img complete in {sw.ElapsedMilliseconds}ms (seed={seed})");

            (float mean, float std) = ComputeRgbStats(outBytes);
            _output.WriteLine($"Output RGB mean={mean:F1} std={std:F1}");

            Directory.CreateDirectory(OutputDir);
            string srcPath = Path.Combine(OutputDir, $"sd15_img2img_source_{W}x{H}.bmp");
            string outPath = Path.Combine(OutputDir, $"sd15_img2img_out_{W}x{H}_seed{seed}_str060.bmp");
            ImagePostProcessor.SaveBmp(srcPath, sourceBytes, W, H);
            ImagePostProcessor.SaveBmp(outPath, outBytes, W, H);
            _output.WriteLine($"Saved: {srcPath}\n       {outPath}");

            // Acceptance: output has correct shape, isn't NaN-collapsed (low std), and isn't byte-identical to source
            // (proves the denoising loop actually ran).
            Assert.Equal(W, outW);
            Assert.Equal(H, outH);
            Assert.Equal(W * H * 3, outBytes.Length);
            Assert.True(std > 5.0f, $"Output looks degenerate (std={std:F1} too low)");
            Assert.NotEqual(sourceBytes, outBytes);

            source.Dispose();
            foreach (Tensor t in unetF32.Values) t.Dispose();
            foreach (Tensor t in clipLF32.Values) t.Dispose();
            foreach (Tensor t in vaeF32.Values) t.Dispose();
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
}
