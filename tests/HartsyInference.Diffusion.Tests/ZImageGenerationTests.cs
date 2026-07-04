using System.Diagnostics;
using Xunit;
using Xunit.Abstractions;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
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

/// <summary>
/// End-to-end Z-Image-Turbo generation. Pipeline: prompt → Qwen3-4B encode → ZImageTransformer denoise → Flux VAE decode → BMP.
/// Z-Image-Turbo is 8 NFE distilled, CFG=1.0 (single forward per step), no negative prompt.
/// VAE is sourced from a Flux.1 dev checkpoint (Z-Image's vae/config.json says <c>_name_or_path: "flux-dev"</c>).
/// </summary>
public sealed class ZImageGenerationTests
{
    private static string ZImageCheckpointPath => TestPaths.ZImage.Turbo;

    /// <summary>Path to a Z-Image-Base single-file checkpoint (e.g. <c>SwarmUI_Z-Image-Base-FP8Mix.safetensors</c> or the diffusers shards combined). Set Z_IMAGE_BASE_PATH to override; the default looks under Models/.</summary>
    private static string ZImageBaseCheckpointPath => TestPaths.ZImage.Base;

    private static string FluxVaeSourcePath => TestPaths.Vae.FluxVaeSource;
    private static string Qwen3WeightsPath => TestPaths.TextEncoders.Qwen3_4B;
    private static string Qwen3VocabPath => TestPaths.Tokenizers.Qwen3Vocab;
    private static string Qwen3MergesPath => TestPaths.Tokenizers.Qwen3Merges;
    private static string OutputDir => TestPaths.OutputDir;

    private readonly ITestOutputHelper _output;

    public ZImageGenerationTests(ITestOutputHelper output) => _output = output;

    /// <summary>
    /// GPU end-to-end Z-Image-Turbo generation at 512×512, 8 NFE, CFG=1.0. Loads the SwarmUI FP8Mix transformer
    /// (fused QKV, ComfyUI fp8_scaled), the Flux.1 VAE (extracted from a Flux dev checkpoint), and Qwen3-4B
    /// as the text encoder. Single forward per step (Turbo is distilled — no CFG).
    /// </summary>
    [Fact]
    public void Turbo_Fp8Mix_GenerateImage_Gpu()
    {
        if (!File.Exists(ZImageCheckpointPath))
        {
            _output.WriteLine($"SKIPPED: Z-Image checkpoint not found: {ZImageCheckpointPath}");
            return;
        }
        if (!File.Exists(FluxVaeSourcePath))
        {
            _output.WriteLine($"SKIPPED: Flux VAE source checkpoint not found: {FluxVaeSourcePath}");
            return;
        }
        if (!File.Exists(Qwen3WeightsPath))
        {
            _output.WriteLine($"SKIPPED: Qwen3 weights not found: {Qwen3WeightsPath}");
            return;
        }
        if (!File.Exists(Qwen3VocabPath) || !File.Exists(Qwen3MergesPath))
        {
            _output.WriteLine("SKIPPED: Qwen3 tokenizer files not found");
            return;
        }

        string assemblyDir = Path.GetDirectoryName(typeof(ZImageGenerationTests).Assembly.Location)!;
        string ptxDir = Path.Combine(assemblyDir, "Ptx");
        if (!Directory.Exists(ptxDir))
        {
            _output.WriteLine($"SKIPPED: PTX directory not found: {ptxDir}");
            return;
        }
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: no CUDA driver available");
            return;
        }

        Stopwatch totalSw = Stopwatch.StartNew();
        Stopwatch sw = Stopwatch.StartNew();

        int gpuOrdinal = int.TryParse(Environment.GetEnvironmentVariable("HARTSY_TEST_GPU_ORDINAL"), out int ord) ? ord : 0;
        CudaBackend backend = new CudaBackend(deviceOrdinal: gpuOrdinal, ptxDir: ptxDir);
        // On a 12 GB card, caching fp8→fp16 weight casts (~13 GB) leaves no room for the VAE decode. Opt out
        // (HARTSY_TEST_CACHE_CASTS=off) to keep the transformer fp8-resident and fit the whole pipeline.
        if (string.Equals(Environment.GetEnvironmentVariable("HARTSY_TEST_CACHE_CASTS"), "off", StringComparison.OrdinalIgnoreCase))
            backend.CacheWeightCasts = false;
        _output.WriteLine($"[GPU] CudaBackend on device ordinal {gpuOrdinal}, CacheWeightCasts={backend.CacheWeightCasts}");

        // ── PHASE A: encode prompt with Qwen3-4B, then dispose it before loading Z-Image (memory pressure: 16 GB Qwen3 F32 + 7 GB Z-Image FP8 + activations exceeds 32 GB easily) ──
        string prompt = "A photograph of an astronaut riding a horse on the moon";
        Tensor captionEmbeddings;
        {
            sw.Restart();
            _output.WriteLine($"[A1] Loading Qwen3-4B from: {Path.GetFileName(Qwen3WeightsPath)}");
            using SafeTensorsLoader qwenLoader = new();
            qwenLoader.Load(Qwen3WeightsPath);
            // Mirror the production ZImageLoader: pass the raw tensors straight to the encoder (it handles
            // fp8/bf16/comfy-quant natively). A blanket CastTo(F32) both breaks on the U8 comfy_quant markers
            // and inflates the encoder to ~16 GB — too big for a 12 GB card.
            Dictionary<string, Tensor> qwenWeights = qwenLoader.GetAllTensors();
            LlamaStyleEncoder qwenEncoder = new(LlamaStyleEncoderConfig.Qwen3_4B);
            qwenEncoder.LoadWeights(qwenWeights);
            _output.WriteLine($"  Qwen3-4B loaded in {sw.ElapsedMilliseconds}ms");

            // Z-Image's diffusers pipeline applies a chat template and uses penultimate hidden state.
            // Without these the model gets the wrong text distribution and outputs glitched images.
            // See pipeline_z_image.py:213-240 — `apply_chat_template(...)` + `hidden_states[-2]`.
            using Qwen3Tokenizer tok = new(Qwen3VocabPath, Qwen3MergesPath, maxLength: 256);
            int[] tokenIds = tok.EncodeChat(prompt);
            int[] mask = Qwen3Tokenizer.CreateAttentionMask(tokenIds);
            int realLen = 0;
            for (int i = 0; i < mask.Length; i++) realLen += mask[i];
            _output.WriteLine($"[A2] Chat-templated prompt — real tokens: {realLen} of {tokenIds.Length}");

            sw.Restart();
            _output.WriteLine("[A3] Encoding prompt with Qwen3-4B (penultimate hidden state) ...");
            // hidden_states[-2] in HF terms = output of layer NumLayers-2 = hfLayerIndex (NumLayers-1).
            int penultimateHfIndex = qwenEncoder.NumLayers - 1;
            Tensor encodedFull = qwenEncoder.EncodeMultiLayer(backend, [tokenIds], [penultimateHfIndex]);
            backend.Sync();
            // encodedFull is [1, maxLength, 2560]. Filter to real-token-only [1, realLen, 2560]
            // — diffusers does `prompt_embeds[i][prompt_masks[i]]` which drops pad slots.
            captionEmbeddings = SliceFirstSeq(encodedFull, realLen);
            encodedFull.Dispose();
            _output.WriteLine($"  Caption embeddings: shape={captionEmbeddings.Shape}, dtype={captionEmbeddings.DType}, in {sw.ElapsedMilliseconds}ms");
            LogTensorStats("captionEmbeddings", captionEmbeddings);

            // Force Qwen3 weights to drop before loading the transformer.
            qwenEncoder.Dispose();
            // qwenWeights holds the original mmap-borrowed tensors; the loader's Dispose will release them.
        }
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();
        _output.WriteLine($"  After Qwen3 dispose: free ≈ {GC.GetGCMemoryInfo().TotalAvailableMemoryBytes / (1024 * 1024 * 1024)} GB");

        // ── PHASE B: load Z-Image transformer + VAE ──
        sw.Restart();
        _output.WriteLine($"[B1] Loading Z-Image checkpoint: {Path.GetFileName(ZImageCheckpointPath)}");
        (ZImageCheckpointConverter.ConvertedWeights zConv, SafeTensorsLoader zLoader) =
            ZImageCheckpointConverter.LoadAndConvert(ZImageCheckpointPath);
        _output.WriteLine($"  Loaded in {sw.ElapsedMilliseconds}ms — Transformer keys: {zConv.Transformer.Count}, FP8Mix={zConv.IsFp8Mix}");
        LogDTypeDistribution("Z-Image Transformer", zConv.Transformer);

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
            _output.WriteLine("[B2] Building ZImageTransformer + loading weights...");
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

                int res = int.TryParse(Environment.GetEnvironmentVariable("HARTSY_TEST_ZIMAGE_RES"), out int r) ? r : 512;
                _output.WriteLine($"[B4] Generating image ({res}×{res}, 8 NFE, CFG=1.0)...");
                ZImagePipeline pipeline = new(backend, transformer, vaeDecoder, zConfig);
                TextToImageRequest request = new()
                {
                    Prompt = prompt,
                    Width = res,
                    Height = res,
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
                string outputPath = Path.Combine(OutputDir, $"zimage_turbo_{width}x{height}_s{request.Steps}_seed{seed}.bmp");
                ImagePostProcessor.SaveBmp(outputPath, rgbData, width, height);
                _output.WriteLine($"Saved: {outputPath}");

                Assert.Equal(res, width);
                Assert.Equal(res, height);
                Assert.Equal(res * res * 3, rgbData.Length);
                ValidateImageNotDegenerate(rgbData);

                totalSw.Stop();
                _output.WriteLine($"\nTotal test time: {totalSw.ElapsedMilliseconds}ms");

                captionEmbeddings.Dispose();
                pipeline.Dispose();
                transformer.Dispose();
                backend.Dispose();
            }
        }
    }

    /// <summary>
    /// GPU end-to-end Z-Image-Base generation at 1024×1024, 28 steps, CFG=4.0 with a negative prompt. Architecturally
    /// identical to Turbo but un-distilled — uses standard CFG (two forwards per step: cond + uncond) and a static
    /// scheduler shift of 6.0 (Turbo uses 3.0). Same checkpoint converter, transformer, VAE, and Qwen3 encoder paths.
    /// Set Z_IMAGE_BASE_PATH to point at a Base single-file checkpoint.
    /// </summary>
    [Fact]
    public void Base_GenerateImage_Gpu()
    {
        if (!File.Exists(ZImageBaseCheckpointPath))
        {
            _output.WriteLine($"SKIPPED: Z-Image-Base checkpoint not found: {ZImageBaseCheckpointPath}");
            return;
        }
        if (!File.Exists(FluxVaeSourcePath) || !File.Exists(Qwen3WeightsPath) ||
            !File.Exists(Qwen3VocabPath) || !File.Exists(Qwen3MergesPath))
        {
            _output.WriteLine("SKIPPED: required dependency files missing (Flux VAE / Qwen3 weights or tokenizer)");
            return;
        }

        string assemblyDir = Path.GetDirectoryName(typeof(ZImageGenerationTests).Assembly.Location)!;
        string ptxDir = Path.Combine(assemblyDir, "Ptx");
        if (!Directory.Exists(ptxDir))
        {
            _output.WriteLine($"SKIPPED: PTX directory not found: {ptxDir}");
            return;
        }
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: no CUDA driver available");
            return;
        }

        Stopwatch totalSw = Stopwatch.StartNew();
        Stopwatch sw = Stopwatch.StartNew();

        CudaBackend backend = new CudaBackend(deviceOrdinal: 0, ptxDir: ptxDir);

        string positivePrompt = "A photograph of an astronaut riding a horse on the moon";
        string negativePrompt = "blurry, low quality, distorted, deformed, extra limbs";

        // ── PHASE A: Qwen3 encode both prompts, then dispose Qwen3 weights before loading the
        // ~12 GB BF16 transformer. Same memory pattern as the Turbo test — without it, peak RAM
        // (Qwen3 F32 ~16 GB + transformer mmap + activations) overflows 32 GB.
        Tensor posEmb, negEmb;
        {
            sw.Restart();
            _output.WriteLine($"[A1] Loading Qwen3-4B from: {Path.GetFileName(Qwen3WeightsPath)}");
            using SafeTensorsLoader qwenLoader = new();
            qwenLoader.Load(Qwen3WeightsPath);
            Dictionary<string, Tensor> qwenWeights = qwenLoader.GetAllTensors();
            // Don't pre-cast Qwen3 weights to F32 — they're BF16 (~7.5 GB) and casting doubles to ~15 GB,
            // which overflows ~18 GB of available RAM. CudaBackend now handles BF16 weight + F32 input via
            // the cast_bf16_f32 PTX kernel, so the encoder runs natively with BF16 weights.
            LlamaStyleEncoder qwenEncoder = new(LlamaStyleEncoderConfig.Qwen3_4B);
            qwenEncoder.LoadWeights(qwenWeights);
            _output.WriteLine($"  Qwen3-4B loaded in {sw.ElapsedMilliseconds}ms");

            using Qwen3Tokenizer tok = new(Qwen3VocabPath, Qwen3MergesPath, maxLength: 256);
            int[] posTokens = tok.EncodeChat(positivePrompt);
            int[] negTokens = tok.EncodeChat(negativePrompt);
            int posReal = 0, negReal = 0;
            foreach (int m in Qwen3Tokenizer.CreateAttentionMask(posTokens)) posReal += m;
            foreach (int m in Qwen3Tokenizer.CreateAttentionMask(negTokens)) negReal += m;
            _output.WriteLine($"[A2] Chat-templated — pos: {posReal} real / {posTokens.Length} total; neg: {negReal} real / {negTokens.Length} total");

            sw.Restart();
            _output.WriteLine("[A3] Encoding both prompts (penultimate hidden state)...");
            int penultimateHfIndex = qwenEncoder.NumLayers - 1;  // diffusers `hidden_states[-2]`
            Tensor posEmbFull = qwenEncoder.EncodeMultiLayer(backend, [posTokens], [penultimateHfIndex]);
            Tensor negEmbFull = qwenEncoder.EncodeMultiLayer(backend, [negTokens], [penultimateHfIndex]);
            backend.Sync();
            posEmb = SliceFirstSeq(posEmbFull, posReal);
            negEmb = SliceFirstSeq(negEmbFull, negReal);
            posEmbFull.Dispose();
            negEmbFull.Dispose();
            _output.WriteLine($"  Caption embeddings done in {sw.ElapsedMilliseconds}ms");
            LogTensorStats("posEmb", posEmb);
            LogTensorStats("negEmb", negEmb);

            qwenEncoder.Dispose();
        }
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();
        _output.WriteLine($"  After Qwen3 dispose: free ≈ {GC.GetGCMemoryInfo().TotalAvailableMemoryBytes / (1024 * 1024 * 1024)} GB");

        // ── PHASE B: Load the BF16 Base transformer + Flux VAE, run the CFG generation loop ──
        sw.Restart();
        _output.WriteLine($"[B1] Loading Z-Image-Base checkpoint: {Path.GetFileName(ZImageBaseCheckpointPath)}");
        (ZImageCheckpointConverter.ConvertedWeights zConv, SafeTensorsLoader zLoader) =
            ZImageCheckpointConverter.LoadAndConvert(ZImageBaseCheckpointPath);
        _output.WriteLine($"  Loaded in {sw.ElapsedMilliseconds}ms — Transformer keys: {zConv.Transformer.Count}, FP8Mix={zConv.IsFp8Mix}");
        LogDTypeDistribution("Z-Image-Base Transformer", zConv.Transformer);

        using (zLoader)
        {
            (int numLayers, int numRefiner, int hidden, int ffnDim, bool isFp8Mix) =
                ZImageCheckpointConverter.DetectArchitecture(zConv.Transformer);
            _output.WriteLine($"  Architecture: {numLayers} layers, {numRefiner} refiner each, hidden={hidden}, ffnDim={ffnDim}, fp8={isFp8Mix}");

            // Z-Image-Base: architecturally identical to Turbo. Only SchedulerShift differs (6.0 vs 3.0)
            // and the sampling regime (CFG vs no-CFG distilled).
            ZImageConfig zConfig = ZImageConfig.Base with
            {
                HiddenSize = hidden,
                NumHeads = hidden / 128,
                NumLayers = numLayers,
                NumRefinerLayers = numRefiner,
                FfnDim = ffnDim,
            };

            // Pass BF16 weights through directly — CudaBackend.Linear casts BF16 weight + F32 input
            // via the cast_bf16_f32 PTX kernel, so we don't need to materialize an F32 copy on CPU
            // (which would cost ~22 GB and tighten Phase B memory unnecessarily).
            sw.Restart();
            ZImageTransformer transformer = new(zConfig);
            transformer.LoadWeights(zConv.Transformer);
            _output.WriteLine($"[B2] Transformer built in {sw.ElapsedMilliseconds}ms");

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

                _output.WriteLine($"[B4] Generating 512×512 image, 28 steps, CFG=4.0 (Base regime)...");
                ZImagePipeline pipeline = new(backend, transformer, vaeDecoder, zConfig);
                TextToImageRequest request = new()
                {
                    Prompt = positivePrompt,
                    Width = 512,
                    Height = 512,
                    Steps = 28,
                    Seed = 42,
                };

                sw.Restart();
                (byte[] rgbData, int width, int height, int seed) = pipeline.GenerateFromEmbeddings(
                    posEmb, request, cfgScale: 4.0f, negativeCaptionEmbeddings: negEmb,
                    onProgress: p => _output.WriteLine($"  Step {p.Step}/{p.TotalSteps} ({p.ElapsedMs:F0}ms)"));
                backend.Sync();
                _output.WriteLine($"Generation done in {sw.ElapsedMilliseconds}ms (seed={seed})");

                Directory.CreateDirectory(OutputDir);
                string outputPath = Path.Combine(OutputDir, $"zimage_base_{width}x{height}_s{request.Steps}_cfg4_seed{seed}.bmp");
                ImagePostProcessor.SaveBmp(outputPath, rgbData, width, height);
                _output.WriteLine($"Saved: {outputPath}");

                Assert.Equal(request.Width, width);
                Assert.Equal(request.Height, height);
                Assert.Equal(width * height * 3, rgbData.Length);
                ValidateImageNotDegenerate(rgbData);

                totalSw.Stop();
                _output.WriteLine($"\nTotal test time: {totalSw.ElapsedMilliseconds}ms");

                posEmb.Dispose();
                negEmb.Dispose();
                pipeline.Dispose();
                transformer.Dispose();
                backend.Dispose();
            }
        }
    }

    private void LogDTypeDistribution(string name, Dictionary<string, Tensor> weights)
    {
        Dictionary<string, int> dtypeCounts = new();
        foreach (KeyValuePair<string, Tensor> kvp in weights)
        {
            string dtype = kvp.Value.DType.ToString();
            dtypeCounts[dtype] = dtypeCounts.GetValueOrDefault(dtype, 0) + 1;
        }
        string distribution = string.Join(", ", dtypeCounts.Select(x => $"{x.Key}={x.Value}"));
        _output.WriteLine($"  {name} dtypes: {distribution}");
    }

    private void LogTensorStats(string name, Tensor tensor)
    {
        ReadOnlySpan<float> data = tensor.AsReadOnlySpan<float>();
        float min = float.MaxValue, max = float.MinValue;
        double sumAbs = 0;
        int nan = 0, inf = 0;
        for (int i = 0; i < data.Length; i++)
        {
            float v = data[i];
            if (float.IsNaN(v)) { nan++; continue; }
            if (float.IsInfinity(v)) { inf++; continue; }
            if (v < min) min = v;
            if (v > max) max = v;
            sumAbs += Math.Abs(v);
        }
        _output.WriteLine($"  [{name}] shape={tensor.Shape} dtype={tensor.DType} min={min:E3} max={max:E3} abs_mean={sumAbs / data.Length:E3} nan={nan} inf={inf}");
    }

    private void ValidateImageNotDegenerate(byte[] rgbData)
    {
        int nonZero = 0;
        int nonFF = 0;
        foreach (byte b in rgbData)
        {
            if (b != 0) nonZero++;
            if (b != 255) nonFF++;
        }
        float nonZeroPct = nonZero / (float)rgbData.Length * 100;
        float nonFFPct = nonFF / (float)rgbData.Length * 100;
        _output.WriteLine($"  Non-zero bytes: {nonZeroPct:F1}%, Non-255 bytes: {nonFFPct:F1}%");
        Assert.True(nonZeroPct > 10, "Image appears to be all black");
        Assert.True(nonFFPct > 10, "Image appears to be all white");
    }

    /// <summary>Slices a [B, S, D] F32 tensor down to [B, realLen, D] by keeping only the first <paramref name="realLen"/> sequence positions. Used to drop pad-position hidden states from the Qwen3 output before feeding the Z-Image transformer (mirrors diffusers' <c>prompt_embeds[i][prompt_masks[i]]</c>).</summary>
    private static unsafe Tensor SliceFirstSeq(Tensor t, int realLen)
    {
        long batch = t.Shape[0];
        long seqLen = t.Shape[1];
        long dim = t.Shape[2];
        if (realLen > seqLen)
            throw new ArgumentOutOfRangeException(nameof(realLen), $"realLen={realLen} > seqLen={seqLen}");

        TensorShape outShape = new TensorShape(batch, realLen, dim);
        Tensor result = new Tensor(outShape, t.DType);
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

    private static Dictionary<string, Tensor> CastWeightsToF32(Dictionary<string, Tensor> weights)
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
}
