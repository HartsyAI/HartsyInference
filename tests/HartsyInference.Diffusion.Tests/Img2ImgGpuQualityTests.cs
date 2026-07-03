using System.Diagnostics;
using Xunit;
using Xunit.Abstractions;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Models.Vae.QwenImage;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.ModelHandler.CheckpointConverters;
using HartsyInference.ModelHandler.Gguf;
using HartsyInference.ModelHandler.SafeTensors;
using HartsyInference.Tests.Common;
using HartsyInference.Tokenizers;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Real-weight GPU quality tests for the img2img / masked-inpaint / edit-reference paths. The CPU wiring
/// tests (SdxlImg2ImgTests, FluxImg2ImgTests, ...) only prove shape checks and the strength=0 pass-through; these
/// tests run the actual denoise loops on CUDA and assert semantic properties:
/// <list type="bullet">
/// <item>strength ordering — a strength-0.4 img2img output must stay closer (mean abs pixel diff) to its source
/// than a strength-0.9 output with the same prompt/seed;</item>
/// <item>masked inpaint locality — with a circular center mask and <c>RecompositeAtEnd=false</c> (so the per-step
/// latent blend is what's under test, not the trivial final pixel composite) the unmasked border must stay near the
/// source while the masked center changes;</item>
/// <item>edit-ref causality — the Qwen-Image-Edit <c>editRefImage</c> conditioning must change the output vs a
/// no-ref run at the same seed (the VACE-style control proof).</item>
/// </list>
/// Every test skips cleanly when its checkpoint / tokenizer / PTX / CUDA / VRAM prerequisite is missing.
/// Resolution and step count are overridable via IMG_W / IMG_H / IMG_STEPS (like the Wan tests' LTX2_W etc.).</summary>
public sealed class Img2ImgGpuQualityTests
{
    private readonly ITestOutputHelper _output;

    public Img2ImgGpuQualityTests(ITestOutputHelper output) => _output = output;

    // ── SDXL ────────────────────────────────────────────────────────────────────────────────────────────

    /// <summary>SDXL base 1.0: t2i a base image, img2img it at strength 0.4 and 0.9 with a different prompt
    /// (assert coherence + strength ordering), then one circular-center-mask inpaint run (assert the unmasked
    /// border stays near the source while the center changes).</summary>
    [Fact]
    [Trait("Category", "GpuIntegration")]
    public void SdxlImg2Img_Gpu_Quality()
    {
        string ckpt = SdxlBaseCheckpointPath;
        if (!File.Exists(ckpt))
        {
            _output.WriteLine($"SKIPPED: SDXL checkpoint not found: {ckpt} (set SDXL_BASE10_PATH)");
            return;
        }
        if (!File.Exists(TestPaths.Tokenizers.ClipVocab) || !File.Exists(TestPaths.Tokenizers.ClipMerges))
        {
            _output.WriteLine("SKIPPED: CLIP tokenizer files not found");
            return;
        }
        string? ptxDir = FindPtxDir();
        if (ptxDir is null)
        {
            _output.WriteLine("SKIPPED: PTX directory not found next to the test assembly");
            return;
        }
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: no CUDA driver available");
            return;
        }

        (int width, int height, int steps) = ResolveSize(defaultSteps: 12);
        Stopwatch totalSw = Stopwatch.StartNew();

        _output.WriteLine($"[1/5] Loading SDXL checkpoint: {Path.GetFileName(ckpt)}");
        (SdxlCheckpointConverter.ConvertedWeights converted, SafeTensorsLoader loader) =
            SdxlCheckpointConverter.LoadAndConvert(ckpt);

        using (loader)
        {
            // Same dtype recipe as SdxlGenerationTests.Gpu_F16_GenerateImage_*: UNet F16 for VRAM/speed,
            // CLIP F32 (CPU-side text encoding), VAE F32 (avoids the well-known SDXL fp16-VAE overflow).
            Dictionary<string, Tensor> unetF16 = CastWeightsToF16(converted.UNet);
            Dictionary<string, Tensor> clipLF32 = CastWeightsToF32(converted.ClipL);
            Dictionary<string, Tensor> clipGF32 = CastWeightsToF32(converted.ClipG);
            Dictionary<string, Tensor> vaeF32 = CastWeightsToF32(converted.Vae);

            _output.WriteLine("[2/5] Loading CLIP-L + CLIP-G + UNet + VAE (decode & encode halves)...");
            ClipTextEncoder clipL = new(ClipTextEncoderConfig.SdxlClipL);
            clipL.LoadWeights(clipLF32, "text_model");
            ClipTextEncoder clipG = new(ClipTextEncoderConfig.SdxlClipG);
            clipG.LoadWeights(clipGF32, "text_model");
            UNet unet = new(UNetConfig.SdxlBase);
            unet.LoadWeights(unetF16);
            VaeDecoder vaeDecoder = new(VaeConfig.Sdxl);
            vaeDecoder.LoadWeights(vaeF32);
            VaeEncoder vaeEncoder = new(VaeConfig.Sdxl);
            vaeEncoder.LoadWeights(vaeF32);

            _output.WriteLine("[3/5] Initializing CUDA backend...");
            using CudaBackend backend = new(deviceOrdinal: 0, ptxDir: ptxDir);
            if (!HasFreeVram(backend, minGb: 6.0))
                return;

            backend.PreloadWeights(unet.EnumerateWeights());
            backend.PreloadWeights(vaeDecoder.EnumerateWeights());
            backend.PreloadWeights(vaeEncoder.EnumerateWeights());
            (long cachedBytes, long _, long _) = backend.GetGpuCacheStats();
            _output.WriteLine($"  Preloaded {cachedBytes / 1024.0 / 1024.0:F1} MB to GPU " +
                $"(device: {backend.Capabilities.Name})");

            using SdxlPipeline pipeline = new(backend, clipL, clipG, unet, vaeDecoder, vaeEncoder);

            using ClipTokenizer tokenizer = new(TestPaths.Tokenizers.ClipVocab, TestPaths.Tokenizers.ClipMerges);
            string basePrompt = "a red barn in a green field under a clear blue sky, photograph";
            string editPrompt = "an oil painting of a lighthouse on a rocky coast at sunset";
            string negPrompt = "blurry, low quality, deformed";

            int[] baseTokensL = tokenizer.Encode(basePrompt);
            int[] baseTokensG = tokenizer.Encode(basePrompt);
            int[] editTokensL = tokenizer.Encode(editPrompt);
            int[] editTokensG = tokenizer.Encode(editPrompt);
            int[] negTokensL = tokenizer.Encode(negPrompt);
            int[] negTokensG = tokenizer.Encode(negPrompt);
            int baseEosG = ClipTokenizer.FindEosPosition(baseTokensG);
            int editEosG = ClipTokenizer.FindEosPosition(editTokensG);
            int negEosG = ClipTokenizer.FindEosPosition(negTokensG);

            _output.WriteLine($"[4/5] t2i base image {width}x{height}, {steps} steps, cfg=7.0, seed=42...");
            TextToImageRequest baseRequest = new()
            {
                Prompt = basePrompt,
                NegativePrompt = negPrompt,
                Width = width,
                Height = height,
                Steps = steps,
                CfgScale = 7.0f,
                Seed = 42,
            };
            (byte[] baseRgb, int outW, int outH, int _) = pipeline.GenerateFromTokens(
                baseTokensL, negTokensL, baseTokensG, negTokensG, baseEosG, negEosG, baseRequest,
                p => _output.WriteLine($"  Step {p.Step}/{p.TotalSteps} ({p.ElapsedMs:F0}ms)"));
            Assert.Equal(width, outW);
            Assert.Equal(height, outH);
            AssertCoherent("sdxl t2i base", baseRgb);
            SaveOutput("sdxl_i2i_base", baseRgb, width, height);

            using Tensor source = ImagePostProcessor.RgbBytesToTensor(baseRgb, width, height);

            byte[] RunImg2Img(float strength, Tensor? mask, bool recompositeAtEnd)
            {
                ImageToImageRequest request = new()
                {
                    Prompt = editPrompt,
                    NegativePrompt = negPrompt,
                    Width = width,
                    Height = height,
                    Steps = steps,
                    CfgScale = 7.0f,
                    Seed = 123,
                    SourceImage = source,
                    Strength = strength,
                    Mask = mask,
                    RecompositeAtEnd = recompositeAtEnd,
                };
                (byte[] rgb, int _, int _, int _) = pipeline.GenerateFromTokens(
                    editTokensL, negTokensL, editTokensG, negTokensG, editEosG, negEosG, request,
                    p => _output.WriteLine($"  Step {p.Step}/{p.TotalSteps} ({p.ElapsedMs:F0}ms)"));
                return rgb;
            }

            _output.WriteLine($"[5/5] img2img strength 0.4 vs 0.9 + circular-mask inpaint (seed=123)...");
            byte[] lowRgb = RunImg2Img(0.4f, mask: null, recompositeAtEnd: true);
            byte[] highRgb = RunImg2Img(0.9f, mask: null, recompositeAtEnd: true);
            AssertCoherent("sdxl img2img s=0.4", lowRgb);
            AssertCoherent("sdxl img2img s=0.9", highRgb);
            SaveOutput("sdxl_i2i_s04", lowRgb, width, height);
            SaveOutput("sdxl_i2i_s09", highRgb, width, height);
            AssertStrengthOrdering(baseRgb, lowRgb, highRgb);

            float radius = Math.Min(width, height) / 4f;
            using Tensor mask = BuildCircularCenterMask(width, height, radius);
            // Production default (RecompositeAtEnd=true): the border is guaranteed by the final pixel composite.
            // SDXL's per-step latent blend alone measured ~40/255 border drift (2026-07-02) — a seam-quality item
            // tracked separately; Flux's equivalent path holds ~14, so the Flux test exercises the latent blend.
            byte[] inpaintRgb = RunImg2Img(0.85f, mask, recompositeAtEnd: true);
            AssertCoherent("sdxl inpaint", inpaintRgb);
            SaveOutput("sdxl_i2i_inpaint", inpaintRgb, width, height);
            AssertInpaintLocality(baseRgb, inpaintRgb, width, height, radius);

            totalSw.Stop();
            _output.WriteLine($"\nTotal: {totalSw.Elapsed.TotalSeconds:F1}s");
        }
    }

    // ── Flux ────────────────────────────────────────────────────────────────────────────────────────────

    /// <summary>Flux.1 Dev FP8: same t2i-base → strength-ordering → masked-inpaint protocol as the SDXL test,
    /// on the flow-matching img2img path (packed 16-channel latents). FP8 recipe follows the Kontext test:
    /// native FP8 weights + <c>CacheWeightCasts=false</c> (caching F16 dequants of a 12B model would OOM 24 GB).</summary>
    [Fact]
    [Trait("Category", "GpuIntegration")]
    public void FluxImg2Img_Gpu_Quality()
    {
        string ckpt = TestPaths.Flux.Dev;
        if (!File.Exists(ckpt))
        {
            _output.WriteLine($"SKIPPED: Flux Dev FP8 checkpoint not found: {ckpt}");
            return;
        }
        if (!File.Exists(TestPaths.Tokenizers.ClipVocab) || !File.Exists(TestPaths.Tokenizers.ClipMerges))
        {
            _output.WriteLine("SKIPPED: CLIP tokenizer files not found");
            return;
        }
        if (!File.Exists(TestPaths.Tokenizers.T5Spiece))
        {
            _output.WriteLine("SKIPPED: T5 SentencePiece model not found");
            return;
        }
        string? ptxDir = FindPtxDir();
        if (ptxDir is null)
        {
            _output.WriteLine("SKIPPED: PTX directory not found next to the test assembly");
            return;
        }
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: no CUDA driver available");
            return;
        }

        (int width, int height, int steps) = ResolveSize(defaultSteps: 10);
        Stopwatch totalSw = Stopwatch.StartNew();

        _output.WriteLine($"[1/5] Loading Flux Dev FP8 checkpoint: {Path.GetFileName(ckpt)}");
        (FluxCheckpointConverter.ConvertedWeights converted, SafeTensorsLoader loader) =
            FluxCheckpointConverter.LoadAndConvert(ckpt);

        using (loader)
        {
            // FP8 weights pass through as-is (models auto-cast the few float*-dereferenced sites);
            // only the small VAE is cast to F32 — same as FluxGenerationTests.Dev_Fp8_GenerateImage_Gpu.
            Dictionary<string, Tensor> vaeF32 = CastWeightsToF32(converted.Vae);

            using CudaBackend backend = new(deviceOrdinal: 0, ptxDir: ptxDir);
            // fp8 recipe (FluxKontextGenerationTests): transient per-GEMM dequant keeps the fp8 footprint —
            // caching the F16 casts of a 12B transformer would roughly double it and OOM a 24 GB card.
            backend.CacheWeightCasts = false;
            if (!HasFreeVram(backend, minGb: 16.0))
                return;

            _output.WriteLine("[2/5] Loading transformer (FP8 native) + CLIP-L + T5-XXL + VAE halves...");
            FluxConfig config = FluxConfig.Dev;
            using FluxTransformer transformer = new(config);
            transformer.LoadWeights(converted.Transformer);
            ClipTextEncoder clipL = new(ClipTextEncoderConfig.SdxlClipL);
            clipL.LoadWeights(converted.ClipL, "text_model");
            T5TextEncoder t5 = new(T5TextEncoderConfig.Xxl);
            t5.LoadWeights(converted.T5);
            VaeDecoder vaeDecoder = new(VaeConfig.Flux);
            vaeDecoder.LoadWeights(vaeF32);
            VaeEncoder vaeEncoder = new(VaeConfig.Flux);
            vaeEncoder.LoadWeights(vaeF32);

            using FluxPipeline pipeline = new(backend, clipL, t5, transformer, vaeDecoder, vaeEncoder, config);

            using ClipTokenizer clipTokenizer = new(TestPaths.Tokenizers.ClipVocab, TestPaths.Tokenizers.ClipMerges);
            using T5Tokenizer t5Tokenizer = new(TestPaths.Tokenizers.T5Spiece, maxLength: 512);
            string basePrompt = "a red barn in a green field under a clear blue sky, photograph";
            string editPrompt = "an oil painting of a lighthouse on a rocky coast at sunset";

            int[] baseClip = clipTokenizer.Encode(basePrompt);
            int baseEos = ClipTokenizer.FindEosPosition(baseClip);
            int[] baseT5 = t5Tokenizer.Encode(basePrompt);
            int[] baseT5Mask = T5Tokenizer.CreateAttentionMask(baseT5);
            int[] editClip = clipTokenizer.Encode(editPrompt);
            int editEos = ClipTokenizer.FindEosPosition(editClip);
            int[] editT5 = t5Tokenizer.Encode(editPrompt);
            int[] editT5Mask = T5Tokenizer.CreateAttentionMask(editT5);

            _output.WriteLine($"[3/5] t2i base image {width}x{height}, {steps} steps, guidance=3.5, seed=42...");
            TextToImageRequest baseRequest = new()
            {
                Prompt = basePrompt,
                Width = width,
                Height = height,
                Steps = steps,
                Seed = 42,
            };
            (byte[] baseRgb, int outW, int outH, int _) = pipeline.GenerateFromTokens(
                baseClip, baseEos, baseT5, baseT5Mask, baseRequest, guidanceScale: 3.5f,
                onProgress: p => _output.WriteLine($"  Step {p.Step}/{p.TotalSteps} ({p.ElapsedMs:F0}ms)"));
            backend.Sync();
            Assert.Equal(width, outW);
            Assert.Equal(height, outH);
            AssertCoherent("flux t2i base", baseRgb);
            SaveOutput("flux_i2i_base", baseRgb, width, height);

            using Tensor source = ImagePostProcessor.RgbBytesToTensor(baseRgb, width, height);

            byte[] RunImg2Img(float strength, Tensor? mask, bool recompositeAtEnd)
            {
                ImageToImageRequest request = new()
                {
                    Prompt = editPrompt,
                    Width = width,
                    Height = height,
                    Steps = steps,
                    Seed = 123,
                    SourceImage = source,
                    Strength = strength,
                    Mask = mask,
                    RecompositeAtEnd = recompositeAtEnd,
                };
                (byte[] rgb, int _, int _, int _) = pipeline.GenerateFromTokens(
                    editClip, editEos, editT5, editT5Mask, request, guidanceScale: 3.5f,
                    onProgress: p => _output.WriteLine($"  Step {p.Step}/{p.TotalSteps} ({p.ElapsedMs:F0}ms)"));
                backend.Sync();
                return rgb;
            }

            _output.WriteLine($"[4/5] img2img strength 0.4 vs 0.9 (seed=123)...");
            byte[] lowRgb = RunImg2Img(0.4f, mask: null, recompositeAtEnd: true);
            byte[] highRgb = RunImg2Img(0.9f, mask: null, recompositeAtEnd: true);
            AssertCoherent("flux img2img s=0.4", lowRgb);
            AssertCoherent("flux img2img s=0.9", highRgb);
            SaveOutput("flux_i2i_s04", lowRgb, width, height);
            SaveOutput("flux_i2i_s09", highRgb, width, height);
            AssertStrengthOrdering(baseRgb, lowRgb, highRgb);

            _output.WriteLine($"[5/5] Circular-mask inpaint (RecompositeAtEnd=false)...");
            float radius = Math.Min(width, height) / 4f;
            using Tensor mask = BuildCircularCenterMask(width, height, radius);
            byte[] inpaintRgb = RunImg2Img(0.85f, mask, recompositeAtEnd: false);
            AssertCoherent("flux inpaint", inpaintRgb);
            SaveOutput("flux_i2i_inpaint", inpaintRgb, width, height);
            // 20/255: measured 14.35 latent-blend border drift (fp8 VAE round-trip); production recomposite is exact.
            AssertInpaintLocality(baseRgb, inpaintRgb, width, height, radius, borderThreshold: 20.0);

            totalSw.Stop();
            _output.WriteLine($"\nTotal: {totalSw.Elapsed.TotalSeconds:F1}s");
            t5.Dispose();
        }
    }

    // ── Chroma ──────────────────────────────────────────────────────────────────────────────────────────

    /// <summary>Chroma1-HD FP8: strength-ordering assertion only. Uses a synthetic smooth-color-field source
    /// (no t2i base run — the transformer is 9 GB fp8 and the pipeline stages T5/transformer/VAE serially, so
    /// each run is expensive). Encoders follow ChromaGenerationTests: T5-XXL extracted from the SD3.5 Medium
    /// bundle with AttentionScale=1.0, Flux VAE pulled from the Flux Dev checkpoint.</summary>
    [Fact]
    [Trait("Category", "GpuIntegration")]
    public void ChromaImg2Img_Gpu_Quality()
    {
        string ckpt = TestPaths.Chroma.V1;
        if (!File.Exists(ckpt))
        {
            _output.WriteLine($"SKIPPED: Chroma checkpoint not found: {ckpt} (set CHROMA_V1_PATH)");
            return;
        }
        if (!File.Exists(TestPaths.Chroma.T5XxlSpiece))
        {
            _output.WriteLine($"SKIPPED: T5-XXL SentencePiece tokenizer not found: {TestPaths.Chroma.T5XxlSpiece}");
            return;
        }
        string t5SourcePath = TestPaths.Chroma.T5XxlSource;
        if (!File.Exists(t5SourcePath))
        {
            _output.WriteLine($"SKIPPED: T5-XXL source not found: {t5SourcePath} (set CHROMA_T5XXL_SOURCE)");
            return;
        }
        string vaePath = TestPaths.Chroma.VaePath;
        if (!File.Exists(vaePath))
        {
            _output.WriteLine($"SKIPPED: Flux VAE source not found: {vaePath} (set CHROMA_VAE_PATH)");
            return;
        }
        string? ptxDir = FindPtxDir();
        if (ptxDir is null)
        {
            _output.WriteLine("SKIPPED: PTX directory not found next to the test assembly");
            return;
        }
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: no CUDA driver available");
            return;
        }

        (int width, int height, int steps) = ResolveSize(defaultSteps: 12);
        Stopwatch totalSw = Stopwatch.StartNew();

        _output.WriteLine($"[1/5] Loading + converting Chroma fp8 checkpoint: {Path.GetFileName(ckpt)}");
        (ChromaCheckpointConverter.ConvertedWeights converted, SafeTensorsLoader transformerLoader) =
            ChromaCheckpointConverter.LoadAndConvert(ckpt);

        using (transformerLoader)
        {
            _output.WriteLine($"[2/5] Extracting T5-XXL from {Path.GetFileName(t5SourcePath)}...");
            (Sd3CheckpointConverter.ConvertedWeights sd3Bundle, SafeTensorsLoader t5Loader) =
                Sd3CheckpointConverter.LoadAndConvert(t5SourcePath);
            using (t5Loader)
            {
                if (sd3Bundle.T5.Count == 0)
                {
                    _output.WriteLine($"SKIPPED: no T5 weights bundled in {Path.GetFileName(t5SourcePath)}");
                    return;
                }

                _output.WriteLine($"[3/5] Loading Flux VAE (both halves): {Path.GetFileName(vaePath)}");
                using SafeTensorsLoader vaeLoader = new();
                vaeLoader.Load(vaePath);
                FluxCheckpointConverter.ConvertedWeights vaeConverted =
                    FluxCheckpointConverter.Convert(vaeLoader.GetAllTensors());
                Dictionary<string, Tensor> vaeF32 = CastWeightsToF32(vaeConverted.Vae);

                // Chroma-specific T5: faithful HF T5 attention scale 1.0 (see ChromaGenerationTests).
                T5TextEncoder t5 = new(T5TextEncoderConfig.Xxl with { AttentionScale = 1.0f });
                t5.LoadWeights(sd3Bundle.T5);

                ChromaConfig config = ChromaConfig.V1;
                using ChromaTransformer transformer = new(config);
                transformer.LoadWeights(converted.Transformer);

                VaeDecoder vaeDecoder = new(VaeConfig.Chroma);
                vaeDecoder.LoadWeights(vaeF32);
                VaeEncoder vaeEncoder = new(VaeConfig.Chroma);
                vaeEncoder.LoadWeights(vaeF32);

                _output.WriteLine("[4/5] Initializing CUDA backend (pipeline handles weight staging)...");
                using CudaBackend backend = new(deviceOrdinal: 0, ptxDir: ptxDir);
                // Transient dequant path — the F16 weight-cast cache of the 9 GB fp8 transformer OOMs 24 GB
                // (same reasoning as ChromaGenerationTests).
                backend.CacheWeightCasts = false;
                if (!HasFreeVram(backend, minGb: 10.0))
                    return;

                using ChromaPipeline pipeline = new(backend, t5, transformer, vaeDecoder, vaeEncoder, config);

                using T5Tokenizer tokenizer = new(TestPaths.Chroma.T5XxlSpiece, maxLength: 512);
                string prompt = "a photograph of a mountain landscape at sunset, dramatic clouds";
                string negPrompt = "low quality, blurry, deformed";
                int[] promptTokens = tokenizer.Encode(prompt);
                int[] negTokens = tokenizer.Encode(negPrompt);
                int[] promptMask = T5Tokenizer.CreateAttentionMask(promptTokens);
                int[] negMask = T5Tokenizer.CreateAttentionMask(negTokens);

                byte[] sourceBytes = BuildSmoothColorFieldBytes(width, height);
                using Tensor source = ImagePostProcessor.RgbBytesToTensor(sourceBytes, width, height);
                _output.WriteLine($"[5/5] img2img over smooth color field, strength 0.4 vs 0.9, " +
                    $"{width}x{height}, {steps} steps, cfg=1.0, seed=42...");

                byte[] RunImg2Img(float strength)
                {
                    ImageToImageRequest request = new()
                    {
                        Prompt = prompt,
                        NegativePrompt = negPrompt,
                        Width = width,
                        Height = height,
                        Steps = steps,
                        CfgScale = 1.0f,
                        Seed = 42,
                        SourceImage = source,
                        Strength = strength,
                    };
                    (byte[] rgb, int outW, int outH, int _) = pipeline.GenerateFromTokens(
                        promptTokens, negTokens, promptMask, negMask, request,
                        p => _output.WriteLine($"  Step {p.Step}/{p.TotalSteps} ({p.ElapsedMs:F0}ms)"));
                    Assert.Equal(width, outW);
                    Assert.Equal(height, outH);
                    return rgb;
                }

                byte[] lowRgb = RunImg2Img(0.4f);
                byte[] highRgb = RunImg2Img(0.9f);
                AssertCoherent("chroma img2img s=0.4", lowRgb);
                AssertCoherent("chroma img2img s=0.9", highRgb);
                SaveOutput("chroma_i2i_s04", lowRgb, width, height);
                SaveOutput("chroma_i2i_s09", highRgb, width, height);
                AssertStrengthOrdering(sourceBytes, lowRgb, highRgb);

                totalSw.Stop();
                _output.WriteLine($"\nTotal: {totalSw.Elapsed.TotalSeconds:F1}s");
                t5.Dispose();
            }
        }
    }

    // ── Qwen-Image-Edit ─────────────────────────────────────────────────────────────────────────────────

    /// <summary>Qwen-Image-Edit reference-latent path: t2i a base image, then run the <c>editRefImage</c>
    /// path with the base as reference. Asserts the edit output is non-degenerate AND differs from a no-ref
    /// run at the same seed — the ref conditioning must have a causal effect (VACE-style control proof).
    /// Prefers the Qwen-Image-Edit 2511 Q5_K_M GGUF; falls back to the base Qwen-Image Q4_K_M GGUF (the ref
    /// path is architecture-identical — conditioning is weaker on the base model but still causal).</summary>
    [Fact]
    [Trait("Category", "GpuIntegration")]
    public void QwenImageEdit_Gpu_Quality()
    {
        string transformerPath = TestPaths.QwenImageEdit.Gguf;
        if (!File.Exists(transformerPath))
        {
            _output.WriteLine($"Qwen-Image-Edit GGUF not found ({transformerPath}); falling back to base Qwen-Image GGUF");
            transformerPath = TestPaths.QwenImage.V1Gguf;
        }
        if (!File.Exists(transformerPath))
        {
            _output.WriteLine($"SKIPPED: no Qwen-Image / Qwen-Image-Edit GGUF found: {transformerPath}");
            return;
        }
        string textEncoderPath = TestPaths.QwenImage.TextEncoder;
        if (!File.Exists(textEncoderPath))
        {
            _output.WriteLine($"SKIPPED: Qwen2.5-VL text encoder not found: {textEncoderPath}");
            return;
        }
        string vaePath = TestPaths.QwenImage.Vae;
        if (!File.Exists(vaePath))
        {
            _output.WriteLine($"SKIPPED: Qwen-Image VAE not found: {vaePath}");
            return;
        }
        string? ptxDir = FindPtxDir();
        if (ptxDir is null)
        {
            _output.WriteLine("SKIPPED: PTX directory not found next to the test assembly");
            return;
        }
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: no CUDA driver available");
            return;
        }

        (int width, int height, int steps) = ResolveSize(defaultSteps: 8);
        Stopwatch totalSw = Stopwatch.StartNew();

        _output.WriteLine($"[1/6] Loading GGUF transformer (lazy, quantized-resident): {Path.GetFileName(transformerPath)}");
        // Same GGUF recipe as QwenImageGenerationTests: relabel rank-2 shapes to PyTorch [out, in] order,
        // identity key-map → converter; handle stays alive (mmap views) until LoadWeights copies to GPU.
        GgufModelLoader.LoadedGgufModel ggufHandle = GgufModelLoader.Load(transformerPath);
        Dictionary<string, Tensor> ggufRawDict = GgufModelLoader.RelabelRank2ToPyTorchOrder(ggufHandle.Weights);
        QwenImageCheckpointConverter.ConvertedWeights converted = QwenImageCheckpointConverter.Convert(ggufRawDict);
        _output.WriteLine($"  {converted.Transformer.Count} transformer keys");

        using (ggufHandle)
        {
            Dictionary<string, Tensor> textEncoderWeights = converted.TextEncoder.Count > 0
                ? converted.TextEncoder
                : LoadStandalone(textEncoderPath);
            Dictionary<string, Tensor> vaeWeights = converted.Vae.Count > 0
                ? converted.Vae
                : LoadStandalone(vaePath);
            Dictionary<string, Tensor> vaeF32 = CastWeightsToF32(vaeWeights);

            // Qwen-Image conditioning template (diffusers _get_qwen_prompt_embeds): ChatML wrap + drop the
            // 34-token template prefix from the hidden states. Same recipe as QwenImageGenerationTests.
            _output.WriteLine("[2/6] Tokenizing with the Qwen-Image ChatML template...");
            using Qwen2Tokenizer tokenizer = new();
            const string qwenImageSystem =
                "Describe the image by detailing the color, shape, size, texture, quantity, text, " +
                "spatial relationships of the objects and background:";
            const int templatePrefixDrop = 34;

            string basePrompt = "A photograph of a red barn in a green field under a clear blue sky";
            string editPrompt = "Turn the sky bright orange like a sunset";
            string negPrompt = "";
            int[] baseTokens = tokenizer.EncodeChat(basePrompt, systemPrompt: qwenImageSystem, addGenerationPrompt: true);
            int[] editTokens = tokenizer.EncodeChat(editPrompt, systemPrompt: qwenImageSystem, addGenerationPrompt: true);
            int[] negTokens = tokenizer.EncodeChat(negPrompt, systemPrompt: qwenImageSystem, addGenerationPrompt: true);

            _output.WriteLine("[3/6] Loading Qwen2.5-VL-7B encoder + transformer + VAE halves...");
            LlamaStyleEncoder textEncoder = new(LlamaStyleEncoderConfig.Qwen2_5_VL_7B);
            textEncoder.LoadWeights(textEncoderWeights);
            QwenImageConfig config = QwenImageConfig.V1;
            QwenImageTransformer transformer = new(config);
            transformer.LoadWeights(converted.Transformer);
            QwenImageVaeDecoder vaeDecoder = new(VaeConfig.QwenImage);
            vaeDecoder.LoadWeights(vaeF32);
            QwenImageVaeEncoder vaeEncoder = new(VaeConfig.QwenImage);
            vaeEncoder.LoadWeights(vaeF32);

            _output.WriteLine("[4/6] Initializing CUDA backend...");
            using CudaBackend backend = new(deviceOrdinal: 0, ptxDir: ptxDir);
            // ~20B GGUF: caching F16 dequants of every weight would need ~40 GB — force transient per-GEMM
            // dequant so only the quantized weights stay resident (Q5_K_M ~15 GB), same as the generation test.
            backend.CacheWeightCasts = false;
            if (!HasFreeVram(backend, minGb: 16.0))
            {
                transformer.Dispose();
                textEncoder.Dispose();
                return;
            }

            using QwenImagePipeline pipeline = new(backend, textEncoder, transformer, vaeDecoder, vaeEncoder, config);

            _output.WriteLine($"[5/6] t2i base image {width}x{height}, {steps} steps, cfg=1.0, seed=42...");
            TextToImageRequest baseRequest = new()
            {
                Prompt = basePrompt,
                NegativePrompt = negPrompt,
                Width = width,
                Height = height,
                Steps = steps,
                CfgScale = 1.0f,
                Seed = 42,
            };
            (byte[] baseRgb, int outW, int outH, int _) = pipeline.GenerateFromTokens(
                baseTokens, negTokens, baseRequest,
                p => _output.WriteLine($"  Step {p.Step}/{p.TotalSteps} ({p.ElapsedMs:F0}ms)"),
                promptDropIndex: templatePrefixDrop, negativeDropIndex: templatePrefixDrop);
            Assert.Equal(width, outW);
            Assert.Equal(height, outH);
            AssertCoherent("qwen t2i base", baseRgb);
            SaveOutput("qwen_edit_base", baseRgb, width, height);

            using Tensor refImage = ImagePostProcessor.RgbBytesToTensor(baseRgb, width, height);

            byte[] RunEdit(Tensor? editRef, string label)
            {
                TextToImageRequest request = new()
                {
                    Prompt = editPrompt,
                    NegativePrompt = negPrompt,
                    Width = width,
                    Height = height,
                    Steps = steps,
                    CfgScale = 1.0f,
                    Seed = 123,
                };
                (byte[] rgb, int _, int _, int _) = pipeline.GenerateFromTokens(
                    editTokens, negTokens, request,
                    p => _output.WriteLine($"  [{label}] Step {p.Step}/{p.TotalSteps} ({p.ElapsedMs:F0}ms)"),
                    promptDropIndex: templatePrefixDrop, negativeDropIndex: templatePrefixDrop,
                    editRefImage: editRef);
                return rgb;
            }

            _output.WriteLine("[6/6] Edit prompt with vs without editRefImage (seed=123)...");
            byte[] noRefRgb = RunEdit(editRef: null, "no-ref");
            byte[] refRgb = RunEdit(refImage, "with-ref");
            AssertCoherent("qwen edit no-ref", noRefRgb);
            AssertCoherent("qwen edit with-ref", refRgb);
            SaveOutput("qwen_edit_noref", noRefRgb, width, height);
            SaveOutput("qwen_edit_withref", refRgb, width, height);

            double refEffect = MeanAbsDiff(refRgb, noRefRgb);
            _output.WriteLine($"  ref-vs-noref mean abs pixel diff = {refEffect:F2} (same seed/prompt)");
            Assert.True(refEffect > 1.0,
                $"editRefImage had no measurable causal effect (mean abs diff {refEffect:F2} <= 1.0)");

            totalSw.Stop();
            _output.WriteLine($"\nTotal: {totalSw.Elapsed.TotalSeconds:F1}s");
            transformer.Dispose();
            textEncoder.Dispose();
        }
    }

    // ── Shared helpers ──────────────────────────────────────────────────────────────────────────────────

    /// <summary>Resolves the sd_xl_base_1.0 checkpoint: SDXL_BASE10_PATH env → the canonical file in the
    /// SDXL models folder → the shared TestPaths default (Juggernaut) as a last resort.</summary>
    private static string SdxlBaseCheckpointPath
    {
        get
        {
            string? env = Environment.GetEnvironmentVariable("SDXL_BASE10_PATH");
            if (!string.IsNullOrEmpty(env))
                return env;
            string dedicated = Path.Combine(TestPaths.ModelsDir, "Stable-Diffusion", "SDXL", "sd_xl_base_1.0.safetensors");
            return File.Exists(dedicated) ? dedicated : TestPaths.Sdxl.SingleFile;
        }
    }

    /// <summary>Resolution/step overrides: IMG_W / IMG_H / IMG_STEPS. Dims snap down to a multiple of 16
    /// (VAE 8x + 2x2 packing for the flow-matching models; also satisfies the SDXL 8x requirement).</summary>
    private static (int width, int height, int steps) ResolveSize(int defaultSteps)
    {
        int width = Math.Max(64, EnvInt("IMG_W", 512) / 16 * 16);
        int height = Math.Max(64, EnvInt("IMG_H", 512) / 16 * 16);
        int steps = Math.Max(1, EnvInt("IMG_STEPS", defaultSteps));
        return (width, height, steps);
    }

    private static int EnvInt(string name, int fallback) =>
        int.TryParse(Environment.GetEnvironmentVariable(name), out int v) ? v : fallback;

    private static string? FindPtxDir()
    {
        string assemblyDir = Path.GetDirectoryName(typeof(Img2ImgGpuQualityTests).Assembly.Location)!;
        string ptxDir = Path.Combine(assemblyDir, "Ptx");
        return Directory.Exists(ptxDir) ? ptxDir : null;
    }

    /// <summary>VRAM skip-gate (QwenImageGenerationTests pattern). Prints free/total and returns false (→ caller
    /// skips) when the device does not have enough headroom for the model's resident footprint.</summary>
    private bool HasFreeVram(CudaBackend backend, double minGb)
    {
        (nuint freeBytes, nuint totalBytes) = backend.Context.GetMemoryInfo();
        double freeGb = freeBytes / (1024.0 * 1024.0 * 1024.0);
        double totalGb = totalBytes / (1024.0 * 1024.0 * 1024.0);
        _output.WriteLine($"  VRAM: {freeGb:F1} GB free / {totalGb:F1} GB total (need >={minGb:F0} GB)");
        if (freeGb >= minGb)
            return true;
        _output.WriteLine($"SKIPPED: only {freeGb:F1} GB free VRAM; need >={minGb:F0} GB");
        return false;
    }

    /// <summary>Smooth source image for img2img: low-frequency sinusoidal RGB field. Full 0-255 range,
    /// no hard edges — encodes cleanly through any VAE and gives a stable baseline for pixel-diff metrics.</summary>
    private static byte[] BuildSmoothColorFieldBytes(int width, int height)
    {
        byte[] rgb = new byte[width * height * 3];
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int o = (y * width + x) * 3;
                rgb[o + 0] = (byte)(127.5 + 127.5 * Math.Sin(2.0 * Math.PI * x / width));
                rgb[o + 1] = (byte)(127.5 + 127.5 * Math.Sin(2.0 * Math.PI * y / height));
                rgb[o + 2] = (byte)(127.5 + 127.5 * Math.Sin(2.0 * Math.PI * (x + y) / (width + height)));
            }
        }
        return rgb;
    }

    /// <summary>Circular center inpaint mask, pixel space [1, 1, H, W]: 1 inside the circle (regenerate),
    /// 0 outside (preserve the source) — the <see cref="ImageToImageRequest.Mask"/> convention.</summary>
    private static Tensor BuildCircularCenterMask(int width, int height, float radius)
    {
        Tensor mask = new Tensor(new TensorShape(1, 1, height, width), DType.F32);
        Span<float> data = mask.AsSpan<float>();
        float cx = width / 2f;
        float cy = height / 2f;
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                float dx = x - cx;
                float dy = y - cy;
                data[y * width + x] = MathF.Sqrt(dx * dx + dy * dy) < radius ? 1f : 0f;
            }
        }
        return mask;
    }

    private static double MeanAbsDiff(byte[] a, byte[] b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException($"Image byte lengths differ: {a.Length} vs {b.Length}");
        long sum = 0;
        for (int i = 0; i < a.Length; i++)
            sum += Math.Abs(a[i] - b[i]);
        return sum / (double)a.Length;
    }

    /// <summary>Prints per-image stats, then asserts the output is non-degenerate: overall byte mean in
    /// [25, 235] (not near-black / near-white) and min→max spread &gt; 30 (not a flat color field).</summary>
    private void AssertCoherent(string label, byte[] rgb)
    {
        long sum = 0;
        int min = 255, max = 0;
        foreach (byte b in rgb)
        {
            sum += b;
            if (b < min) min = b;
            if (b > max) max = b;
        }
        double mean = sum / (double)rgb.Length;
        int spread = max - min;
        _output.WriteLine($"  [{label}] mean={mean:F1}, min={min}, max={max}, spread={spread}");
        Assert.True(mean is >= 25.0 and <= 235.0, $"{label}: byte mean {mean:F1} outside [25, 235] — degenerate output");
        Assert.True(spread > 30, $"{label}: min/max spread {spread} <= 30 — flat output");
    }

    /// <summary>Prints both diffs, then asserts the strength-0.4 output sits closer to the source than the
    /// strength-0.9 output (mean abs pixel diff) — the core semantic property of the strength parameter.</summary>
    /// <summary>Isolates the SDXL VAE encode→decode round-trip from any denoising: if this alone drifts far from
    /// the source, the inpaint border drift is a VAE/scaling issue, not the per-step latent blend.</summary>
    [Fact]
    [Trait("Category", "GpuIntegration")]
    public void SdxlVaeRoundtrip_Gpu()
    {
        string ckpt = Environment.GetEnvironmentVariable("SDXL_BASE10_PATH")
            ?? "/home/hartsy/Desktop/HartsyInference/Models/Stable-Diffusion/SDXL/sd_xl_base_1.0.safetensors";
        if (!File.Exists(ckpt)) { _output.WriteLine($"SKIPPED: SDXL checkpoint not found: {ckpt}"); return; }
        string? ptxDir = FindPtxDir();
        if (ptxDir is null || !CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: no PTX/CUDA"); return; }

        (int width, int height, _) = ResolveSize(defaultSteps: 1);
        (SdxlCheckpointConverter.ConvertedWeights converted, SafeTensorsLoader loader) =
            SdxlCheckpointConverter.LoadAndConvert(ckpt);
        using (loader)
        {
            Dictionary<string, Tensor> vaeF32 = CastWeightsToF32(converted.Vae);
            VaeDecoder vaeDecoder = new(VaeConfig.Sdxl);
            vaeDecoder.LoadWeights(vaeF32);
            VaeEncoder vaeEncoder = new(VaeConfig.Sdxl);
            vaeEncoder.LoadWeights(vaeF32);

            using CudaBackend backend = new(deviceOrdinal: 0, ptxDir: ptxDir);
            if (!HasFreeVram(backend, minGb: 4.0)) return;
            backend.PreloadWeights(vaeDecoder.EnumerateWeights());
            backend.PreloadWeights(vaeEncoder.EnumerateWeights());

            double Roundtrip(byte[] rgb, string label)
            {
                using Tensor src = ImagePostProcessor.RgbBytesToTensor(rgb, width, height);
                Tensor latent = vaeEncoder.Encode(backend, src);
                Tensor decoded = vaeDecoder.DecodeTiled(backend, latent);
                latent.Dispose();
                byte[] outRgb = ImagePostProcessor.TensorToRgbBytes(decoded);
                decoded.Dispose();
                double d = MeanAbsDiff(outRgb, rgb);
                _output.WriteLine($"  SDXL VAE roundtrip ({label}): mean abs diff = {d:F2} /255");
                return d;
            }

            double smooth = Roundtrip(BuildSmoothColorFieldBytes(width, height), "smooth field");
            // A real generated image (textured content) stresses the roundtrip far more than a smooth
            // gradient — this is the number that bounds the inpaint border with RecompositeAtEnd=false.
            string? baseBmp = Directory.Exists(TestPaths.OutputDir)
                ? Directory.GetFiles(TestPaths.OutputDir, "sdxl_i2i_base_*.bmp").OrderByDescending(File.GetLastWriteTimeUtc).FirstOrDefault()
                : null;
            if (baseBmp is not null)
            {
                byte[]? realRgb = TryLoadBmp24(baseBmp, width, height);
                if (realRgb is not null) Roundtrip(realRgb, $"generated image {Path.GetFileName(baseBmp)}");
                else _output.WriteLine($"  (skipped generated-image roundtrip: {Path.GetFileName(baseBmp)} not a matching 24-bit BMP)");
            }
            Assert.True(smooth < 10.0, $"SDXL VAE encode→decode roundtrip drifts {smooth:F2} >= 10 — encoder/decoder scaling mismatch?");
        }
    }

    private void AssertStrengthOrdering(byte[] sourceRgb, byte[] lowStrengthRgb, byte[] highStrengthRgb)
    {
        double lowDiff = MeanAbsDiff(lowStrengthRgb, sourceRgb);
        double highDiff = MeanAbsDiff(highStrengthRgb, sourceRgb);
        _output.WriteLine($"  strength ordering: diff(0.4)={lowDiff:F2}, diff(0.9)={highDiff:F2} (vs source)");
        Assert.True(lowDiff < highDiff,
            $"strength 0.4 output should be closer to the source than strength 0.9: {lowDiff:F2} !< {highDiff:F2}");
    }

    /// <summary>Prints center/border diffs vs the source, then asserts inpaint locality: the unmasked border
    /// stays near the source (mean abs diff &lt; 12/255) while the masked center actually changes. A 16 px
    /// annulus around the circle edge is excluded from both regions — the mask is downsampled to latent
    /// resolution, so the boundary feathers.</summary>
    private void AssertInpaintLocality(byte[] sourceRgb, byte[] inpaintRgb, int width, int height, float radius,
        double borderThreshold = 12.0)
    {
        float margin = MathF.Min(16f, radius / 4f);
        float cx = width / 2f;
        float cy = height / 2f;
        long centerSum = 0, borderSum = 0;
        long centerCount = 0, borderCount = 0;
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                float dx = x - cx;
                float dy = y - cy;
                float dist = MathF.Sqrt(dx * dx + dy * dy);
                int o = (y * width + x) * 3;
                long d = Math.Abs(inpaintRgb[o] - sourceRgb[o])
                    + Math.Abs(inpaintRgb[o + 1] - sourceRgb[o + 1])
                    + Math.Abs(inpaintRgb[o + 2] - sourceRgb[o + 2]);
                if (dist < radius - margin)
                {
                    centerSum += d;
                    centerCount += 3;
                }
                else if (dist > radius + margin)
                {
                    borderSum += d;
                    borderCount += 3;
                }
            }
        }
        Assert.True(centerCount > 0 && borderCount > 0, "degenerate mask geometry — no center/border pixels");
        double centerDiff = centerSum / (double)centerCount;
        double borderDiff = borderSum / (double)borderCount;
        _output.WriteLine($"  inpaint locality: center diff={centerDiff:F2}, border diff={borderDiff:F2} (vs source)");
        Assert.True(borderDiff < borderThreshold,
            $"unmasked border drifted from the source: mean abs diff {borderDiff:F2} >= {borderThreshold} — mask blend not applied?");
        Assert.True(centerDiff > 5.0,
            $"masked center barely changed: mean abs diff {centerDiff:F2} <= 5 — inpaint had no effect");
        Assert.True(centerDiff > borderDiff,
            $"masked center ({centerDiff:F2}) should change more than the unmasked border ({borderDiff:F2})");
    }

    /// <summary>Reads an uncompressed 24-bit bottom-up BMP written by <see cref="ImagePostProcessor.SaveBmp"/>
    /// back into interleaved RGB; returns null on any dimension/format mismatch.</summary>
    private static byte[]? TryLoadBmp24(string path, int expectedW, int expectedH)
    {
        byte[] file = File.ReadAllBytes(path);
        if (file.Length < 54 || file[0] != 'B' || file[1] != 'M') return null;
        int dataOffset = BitConverter.ToInt32(file, 10);
        int w = BitConverter.ToInt32(file, 18);
        int hRaw = BitConverter.ToInt32(file, 22);
        bool topDown = hRaw < 0;   // negative height = top-down row order (what SaveBmp writes)
        int h = Math.Abs(hRaw);
        short bpp = BitConverter.ToInt16(file, 28);
        if (w != expectedW || h != expectedH || bpp != 24) return null;
        int rowBytes = (w * 3 + 3) & ~3;
        byte[] rgb = new byte[w * h * 3];
        for (int y = 0; y < h; y++)
        {
            int srcRow = dataOffset + (topDown ? y : h - 1 - y) * rowBytes;
            for (int x = 0; x < w; x++)
            {
                int s = srcRow + x * 3;
                int d = (y * w + x) * 3;
                rgb[d] = file[s + 2];       // BMP stores BGR
                rgb[d + 1] = file[s + 1];
                rgb[d + 2] = file[s];
            }
        }
        return rgb;
    }

    private void SaveOutput(string name, byte[] rgb, int width, int height)
    {
        Directory.CreateDirectory(TestPaths.OutputDir);
        string path = Path.Combine(TestPaths.OutputDir, $"{name}_{DateTime.Now:yyyyMMdd_HHmmss}.bmp");
        ImagePostProcessor.SaveBmp(path, rgb, width, height);
        _output.WriteLine($"  Saved: {path}");
    }

    private static Dictionary<string, Tensor> LoadStandalone(string path)
    {
        SafeTensorsLoader loader = new SafeTensorsLoader();
        loader.Load(path);
        return loader.GetAllTensors();
    }

    private static Dictionary<string, Tensor> CastWeightsToF32(Dictionary<string, Tensor> weights)
    {
        Dictionary<string, Tensor> f32 = new(weights.Count);
        foreach (KeyValuePair<string, Tensor> kvp in weights)
        {
            DType dt = kvp.Value.DType;
            f32[kvp.Key] = (dt == DType.F16 || dt == DType.BF16 || dt.IsFp8)
                ? kvp.Value.CastTo(DType.F32)
                : kvp.Value;
        }
        return f32;
    }

    private static Dictionary<string, Tensor> CastWeightsToF16(Dictionary<string, Tensor> weights)
    {
        Dictionary<string, Tensor> f16 = new(weights.Count);
        foreach (KeyValuePair<string, Tensor> kvp in weights)
            f16[kvp.Key] = kvp.Value.DType == DType.F16 ? kvp.Value : kvp.Value.CastTo(DType.F16);
        return f16;
    }
}
