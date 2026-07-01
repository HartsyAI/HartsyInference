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
using HartsyInference.ModelHandler.SafeTensors;
using HartsyInference.Tests.Common;
using HartsyInference.Tokenizers;

namespace HartsyInference.Diffusion.Tests;

/// <summary>End-to-end Flux.1 Kontext instruction-edit generation. Kontext reuses the base Flux transformer
/// (x_embedder is 64-channel — verified) and conditions on a reference image by VAE-encoding it, 2×2-packing
/// it to 64-ch tokens, and APPENDING those tokens to the noise tokens along the sequence each denoise step
/// (RoPE temporal-axis = 1 distinguishes them). The transformer output is sliced back to the noise tokens.
/// The Kontext checkpoint is transformer-only; encoders + VAE come from the Dev FP8 file. Skips cleanly when
/// artifacts or the reference RGB are missing.</summary>
public sealed class FluxKontextGenerationTests
{
    private static string KontextPath => TestPaths.Flux.Kontext;
    private static string DevFp8Path => TestPaths.Flux.Dev;
    private static string RefRgbPath => Path.Combine(TestPaths.OutputDir, "References", "kontext_ref_astronaut_512.rgb");
    private readonly ITestOutputHelper _output;

    public FluxKontextGenerationTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void Kontext_Edit_Gpu_1024()
    {
        if (!File.Exists(KontextPath)) { _output.WriteLine($"SKIPPED: Kontext checkpoint not found: {KontextPath}"); return; }
        if (!File.Exists(DevFp8Path)) { _output.WriteLine($"SKIPPED: Dev FP8 (shared encoders+VAE) not found: {DevFp8Path}"); return; }
        if (!File.Exists(TestPaths.Tokenizers.ClipVocab) || !File.Exists(TestPaths.Tokenizers.ClipMerges) || !File.Exists(TestPaths.Tokenizers.T5Spiece))
        { _output.WriteLine("SKIPPED: tokenizer assets missing"); return; }
        if (!File.Exists(RefRgbPath)) { _output.WriteLine($"SKIPPED: reference RGB not found: {RefRgbPath} (1024x1024x3 raw bytes)"); return; }

        string assemblyDir = Path.GetDirectoryName(typeof(FluxKontextGenerationTests).Assembly.Location)!;
        string ptxDir = Path.Combine(assemblyDir, "Ptx");
        if (!Directory.Exists(ptxDir)) { _output.WriteLine($"SKIPPED: PTX dir not found: {ptxDir}"); return; }
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: no CUDA"); return; }

        const int W = 512, H = 512;
        Stopwatch total = Stopwatch.StartNew();

        _output.WriteLine($"[1/6] Loading Kontext transformer: {Path.GetFileName(KontextPath)}");
        (FluxCheckpointConverter.ConvertedWeights kontext, SafeTensorsLoader kLoader) = FluxCheckpointConverter.LoadAndConvert(KontextPath);
        _output.WriteLine($"  {kontext.Transformer.Count} transformer keys");
        (FluxCheckpointConverter.ConvertedWeights dev, SafeTensorsLoader dLoader) = FluxCheckpointConverter.LoadAndConvert(DevFp8Path);

        using (kLoader)
        using (dLoader)
        {
            Dictionary<string, Tensor> vaeF32 = CastWeightsToF32(dev.Vae);
            using CudaBackend backend = new(deviceOrdinal: 0, ptxDir: ptxDir);
            // fp8 transformer (~11GB) + doubled Kontext sequence. Caching the dequantized BF16 weights
            // (CacheWeightCasts default) would ~double the transformer to 22GB and OOM. Transient per-GEMM
            // dequant keeps the fp8 footprint.
            backend.CacheWeightCasts = false;

            FluxConfig config = FluxConfig.Dev; // Kontext is a Dev-architecture fine-tune (guidance embedder)
            using FluxTransformer transformer = new(config);
            transformer.LoadWeights(kontext.Transformer);

            ClipTextEncoder clipL = new(ClipTextEncoderConfig.SdxlClipL);
            clipL.LoadWeights(dev.ClipL, "text_model");
            T5TextEncoder t5 = new(T5TextEncoderConfig.Xxl);
            t5.LoadWeights(dev.T5);
            VaeDecoder vaeDecoder = new(VaeConfig.Flux);
            vaeDecoder.LoadWeights(vaeF32);
            VaeEncoder vaeEncoder = new(VaeConfig.Flux);
            vaeEncoder.LoadWeights(vaeF32);
            _output.WriteLine("[2/6] Transformer + encoders + VAE (encode/decode) loaded");

            // Reference image: raw 1024x1024x3 RGB bytes → [1, 3, H, W]
            byte[] refBytes = File.ReadAllBytes(RefRgbPath);
            Assert.Equal(W * H * 3, refBytes.Length);
            using Tensor refImage = ImagePostProcessor.RgbBytesToTensor(refBytes, W, H);
            _output.WriteLine("[3/6] Reference image loaded");

            using ClipTokenizer clipTok = new(TestPaths.Tokenizers.ClipVocab, TestPaths.Tokenizers.ClipMerges);
            using T5Tokenizer t5Tok = new(TestPaths.Tokenizers.T5Spiece, maxLength: 512);
            string editPrompt = "Convert this photo to black and white grayscale";
            int[] cl = clipTok.Encode(editPrompt);
            int eos = ClipTokenizer.FindEosPosition(cl);
            int[] t5t = t5Tok.Encode(editPrompt);
            int[] t5m = T5Tokenizer.CreateAttentionMask(t5t);

            TextToImageRequest request = new() { Prompt = editPrompt, Width = W, Height = H, Steps = 28, Seed = 42 };

            using FluxPipeline pipeline = new(backend, clipL, t5, transformer, vaeDecoder, vaeEncoder, config);

            _output.WriteLine($"[4/6] Editing (Kontext) {W}x{H}, 28 steps, seed=42, prompt='{editPrompt}'...");
            Stopwatch gen = Stopwatch.StartNew();
            (byte[] rgb, int outW, int outH, int seed) = pipeline.GenerateFromTokens(
                cl, eos, t5t, t5m, request, guidanceScale: 4.0f,
                onProgress: p => _output.WriteLine($"  Step {p.Step}/{p.TotalSteps} ({p.ElapsedMs:F0}ms)"),
                kontextRefImage: refImage);
            backend.Sync();
            _output.WriteLine($"[5/6] Edit complete in {gen.Elapsed.TotalSeconds:F1}s (seed={seed})");

            Assert.Equal(W, outW);
            Assert.Equal(H, outH);
            Assert.Equal(W * H * 3, rgb.Length);
            ValidateNotDegenerate(rgb);

            Directory.CreateDirectory(TestPaths.OutputDir);
            string outPath = Path.Combine(TestPaths.OutputDir, $"flux_kontext_edit_{DateTime.Now:yyyyMMdd_HHmmss}.bmp");
            ImagePostProcessor.SaveBmp(outPath, rgb, outW, outH);
            _output.WriteLine($"[6/6] Saved: {outPath}");

            total.Stop();
            _output.WriteLine($"Total: {total.Elapsed.TotalSeconds:F1}s");
            t5.Dispose();
        }
    }

    private void ValidateNotDegenerate(byte[] rgb)
    {
        int nz = 0, nff = 0;
        foreach (byte b in rgb) { if (b != 0) nz++; if (b != 255) nff++; }
        _output.WriteLine($"  Non-zero: {nz / (float)rgb.Length * 100:F1}%, Non-255: {nff / (float)rgb.Length * 100:F1}%");
        Assert.True(nz / (float)rgb.Length * 100 > 10, "all black");
        Assert.True(nff / (float)rgb.Length * 100 > 10, "all white");
    }

    private static Dictionary<string, Tensor> CastWeightsToF32(Dictionary<string, Tensor> weights)
    {
        Dictionary<string, Tensor> f32 = new(weights.Count);
        foreach (KeyValuePair<string, Tensor> kvp in weights)
        {
            DType dt = kvp.Value.DType;
            f32[kvp.Key] = (dt == DType.F16 || dt == DType.BF16 || dt.IsFp8) ? kvp.Value.CastTo(DType.F32) : kvp.Value;
        }
        return f32;
    }
}
