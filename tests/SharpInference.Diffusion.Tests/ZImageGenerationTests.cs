using System.Diagnostics;
using Xunit;
using Xunit.Abstractions;
using SharpInference.Core.Tensors;
using SharpInference.Cpu;
using SharpInference.Cuda;
using SharpInference.Diffusion.Models.Denoisers;
using SharpInference.Diffusion.Models.TextEncoders;
using SharpInference.Diffusion.Models.Vae;
using SharpInference.Diffusion.Pipelines;
using SharpInference.Diffusion.Requests;
using SharpInference.Diffusion.Utilities;
using SharpInference.ModelHandler.CheckpointConverters;
using SharpInference.ModelHandler.SafeTensors;
using SharpInference.Tokenizers;

namespace SharpInference.Diffusion.Tests;

/// <summary>
/// End-to-end Z-Image-Turbo generation. Pipeline: prompt → Qwen3-4B encode → ZImageTransformer denoise → Flux VAE decode → BMP.
/// Z-Image-Turbo is 8 NFE distilled, CFG=1.0 (single forward per step), no negative prompt.
/// VAE is sourced from a Flux.1 dev checkpoint (Z-Image's vae/config.json says <c>_name_or_path: "flux-dev"</c>).
/// </summary>
public sealed class ZImageGenerationTests
{
    private static string PickPath(string envVar, string winPath, string linuxPath)
    {
        string? v = Environment.GetEnvironmentVariable(envVar);
        if (!string.IsNullOrEmpty(v)) return v;
        return OperatingSystem.IsWindows() ? winPath : linuxPath;
    }

    private static readonly string LinuxRepoRoot =
        Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", ".."));

    private static readonly string ZImageCheckpointPath = PickPath(
        "Z_IMAGE_TURBO_PATH",
        @"C:\Users\kaleb\Desktop\Projects\SwarmUI\Models\Stable-Diffusion\Z-Image\SwarmUI_Z-Image-Turbo-FP8Mix.safetensors",
        Path.Combine(LinuxRepoRoot, "Models", "SwarmUI_Z-Image-Turbo-FP8Mix.safetensors"));

    /// <summary>Path to a Z-Image-Base single-file checkpoint (e.g. <c>SwarmUI_Z-Image-Base-FP8Mix.safetensors</c> or the diffusers shards combined). Set Z_IMAGE_BASE_PATH to override; the default looks under Models/.</summary>
    private static readonly string ZImageBaseCheckpointPath = PickPath(
        "Z_IMAGE_BASE_PATH",
        @"C:\Users\kaleb\Desktop\Projects\SwarmUI\Models\Stable-Diffusion\Z-Image\SwarmUI_Z-Image-Base-FP8Mix.safetensors",
        Path.Combine(LinuxRepoRoot, "Models", "SwarmUI_Z-Image-Base-FP8Mix.safetensors"));

    private static readonly string FluxVaeSourcePath = PickPath(
        "FLUX_VAE_SOURCE_PATH",
        @"C:\Users\kaleb\Desktop\Projects\SwarmUI\Models\Stable-Diffusion\Flux\flux1-dev-fp8.safetensors",
        Path.Combine(LinuxRepoRoot, "Models", "flux1-dev-fp8.safetensors"));

    private static readonly string Qwen3WeightsPath = PickPath(
        "QWEN3_4B_PATH",
        @"C:\Users\kaleb\Desktop\Projects\SwarmUI\Models\Stable-Diffusion\Flux\qwen_3_4b.safetensors",
        Path.Combine(LinuxRepoRoot, "Models", "qwen_3_4b.safetensors"));

    private static readonly string Qwen3VocabPath = PickPath(
        "QWEN3_VOCAB_PATH",
        @"C:\Users\kaleb\Desktop\projects\SharpInference\tests\test-models\qwen3-4b\vocab.json",
        Path.Combine(LinuxRepoRoot, "tests", "test-models", "qwen3-4b", "vocab.json"));

    private static readonly string Qwen3MergesPath = PickPath(
        "QWEN3_MERGES_PATH",
        @"C:\Users\kaleb\Desktop\projects\SharpInference\tests\test-models\qwen3-4b\merges.txt",
        Path.Combine(LinuxRepoRoot, "tests", "test-models", "qwen3-4b", "merges.txt"));

    private static readonly string OutputDir = PickPath(
        "Z_IMAGE_OUTPUT_DIR",
        @"C:\Users\kaleb\Desktop\projects\SharpInference\Output",
        Path.Combine(LinuxRepoRoot, "Output"));

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

        Stopwatch totalSw = Stopwatch.StartNew();
        Stopwatch sw = Stopwatch.StartNew();

        CudaBackend backend = new CudaBackend(deviceOrdinal: 0, ptxDir: ptxDir);

        // ── PHASE A: encode prompt with Qwen3-4B, then dispose it before loading Z-Image (memory pressure: 16 GB Qwen3 F32 + 7 GB Z-Image FP8 + activations exceeds 32 GB easily) ──
        string prompt = "A photograph of an astronaut riding a horse on the moon";
        Tensor captionEmbeddings;
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

            using Qwen3Tokenizer tok = new(Qwen3VocabPath, Qwen3MergesPath, maxLength: 64);
            int[] tokenIds = tok.Encode(prompt, appendEos: true);
            _output.WriteLine($"[A2] Tokenized prompt: \"{prompt}\" — raw tokens: {tok.EncodeRaw(prompt).Count}");

            sw.Restart();
            _output.WriteLine("[A3] Encoding prompt with Qwen3-4B...");
            Tensor encoded = qwenEncoder.Encode(backend, [tokenIds]);
            backend.Sync();
            // Materialize to a CPU-owned tensor that doesn't reference any Qwen3 weight buffers.
            captionEmbeddings = encoded;
            _output.WriteLine($"  Caption embeddings: shape={captionEmbeddings.Shape}, dtype={captionEmbeddings.DType}, in {sw.ElapsedMilliseconds}ms");
            LogTensorStats("captionEmbeddings", captionEmbeddings);

            // Force Qwen3 weights to drop. Cast tensors hold heap-allocated F32 buffers — disposing the
            // F32 dict and the encoder lets ~16 GB go before we load the transformer.
            qwenEncoder.Dispose();
            foreach (Tensor t in qwenF32.Values)
                t.Dispose();
            qwenF32.Clear();
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

                _output.WriteLine("[B4] Generating image (512×512, 8 NFE, CFG=1.0)...");
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
                string outputPath = Path.Combine(OutputDir, $"zimage_turbo_{width}x{height}_s{request.Steps}_seed{seed}.bmp");
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

        Stopwatch totalSw = Stopwatch.StartNew();
        Stopwatch sw = Stopwatch.StartNew();

        _output.WriteLine($"[1/7] Loading Z-Image-Base checkpoint: {Path.GetFileName(ZImageBaseCheckpointPath)}");
        (ZImageCheckpointConverter.ConvertedWeights zConv, SafeTensorsLoader zLoader) =
            ZImageCheckpointConverter.LoadAndConvert(ZImageBaseCheckpointPath);
        _output.WriteLine($"  Loaded in {sw.ElapsedMilliseconds}ms — Transformer keys: {zConv.Transformer.Count}, FP8Mix={zConv.IsFp8Mix}");
        LogDTypeDistribution("Z-Image-Base Transformer", zConv.Transformer);

        using (zLoader)
        {
            (int numLayers, int numRefiner, int hidden, int ffnDim, bool isFp8Mix) =
                ZImageCheckpointConverter.DetectArchitecture(zConv.Transformer);
            _output.WriteLine($"  Architecture: {numLayers} layers, {numRefiner} refiner each, hidden={hidden}, ffnDim={ffnDim}, fp8={isFp8Mix}");

            // Z-Image-Base is architecturally identical to Turbo per Tongyi-MAI/Z-Image transformer/config.json.
            // Only the scheduler shift differs (6.0 vs 3.0) and the sampling regime (CFG vs no-CFG distilled).
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
            _output.WriteLine($"[2/7] Transformer loaded in {sw.ElapsedMilliseconds}ms");

            sw.Restart();
            _output.WriteLine($"[3/7] Loading Flux VAE from: {Path.GetFileName(FluxVaeSourcePath)}");
            (FluxCheckpointConverter.ConvertedWeights fluxConv, SafeTensorsLoader fluxLoader) =
                FluxCheckpointConverter.LoadAndConvert(FluxVaeSourcePath);
            using (fluxLoader)
            {
                Dictionary<string, Tensor> vaeF32 = CastWeightsToF32(fluxConv.Vae);
                VaeDecoder vaeDecoder = new(VaeConfig.ZImage);
                vaeDecoder.LoadWeights(vaeF32);
                _output.WriteLine($"  VAE loaded in {sw.ElapsedMilliseconds}ms");

                sw.Restart();
                _output.WriteLine("[4/7] Loading Qwen3-4B...");
                using SafeTensorsLoader qwenLoader = new();
                qwenLoader.Load(Qwen3WeightsPath);
                Dictionary<string, Tensor> qwenWeights = qwenLoader.GetAllTensors();
                Dictionary<string, Tensor> qwenF32 = new(qwenWeights.Count);
                foreach (KeyValuePair<string, Tensor> kvp in qwenWeights)
                    qwenF32[kvp.Key] = kvp.Value.DType == DType.F32 ? kvp.Value : kvp.Value.CastTo(DType.F32);
                LlamaStyleEncoder qwenEncoder = new(LlamaStyleEncoderConfig.Qwen3_4B);
                qwenEncoder.LoadWeights(qwenF32);
                _output.WriteLine($"  Qwen3-4B loaded in {sw.ElapsedMilliseconds}ms");

                string positivePrompt = "A photograph of an astronaut riding a horse on the moon";
                string negativePrompt = "blurry, low quality, distorted, deformed, extra limbs";
                _output.WriteLine($"[5/7] Tokenizing positive + negative prompts");
                using Qwen3Tokenizer tok = new(Qwen3VocabPath, Qwen3MergesPath, maxLength: 64);
                int[] posTokens = tok.Encode(positivePrompt, appendEos: true);
                int[] negTokens = tok.Encode(negativePrompt, appendEos: true);

                sw.Restart();
                _output.WriteLine("[6/7] Building backend + encoding both prompts with Qwen3-4B...");
                CudaBackend backend = new CudaBackend(deviceOrdinal: 0, ptxDir: ptxDir);
                Tensor posEmb = qwenEncoder.Encode(backend, [posTokens]);
                Tensor negEmb = qwenEncoder.Encode(backend, [negTokens]);
                backend.Sync();
                _output.WriteLine($"  Caption embeddings done in {sw.ElapsedMilliseconds}ms");
                LogTensorStats("posEmb", posEmb);
                LogTensorStats("negEmb", negEmb);

                _output.WriteLine($"[7/7] Generating 1024×1024 image, 28 steps, CFG=4.0 (Base regime)...");
                ZImagePipeline pipeline = new(backend, transformer, vaeDecoder, zConfig);
                TextToImageRequest request = new()
                {
                    Prompt = positivePrompt,
                    Width = 1024,
                    Height = 1024,
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

                Assert.Equal(1024, width);
                Assert.Equal(1024, height);
                Assert.Equal(1024 * 1024 * 3, rgbData.Length);
                ValidateImageNotDegenerate(rgbData);

                totalSw.Stop();
                _output.WriteLine($"\nTotal test time: {totalSw.ElapsedMilliseconds}ms");

                posEmb.Dispose();
                negEmb.Dispose();
                pipeline.Dispose();
                qwenEncoder.Dispose();
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
