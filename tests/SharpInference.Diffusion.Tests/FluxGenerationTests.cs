using System.Diagnostics;
using Xunit;
using Xunit.Abstractions;
using SharpInference.Core.Backends;
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
/// End-to-end Flux image generation tests using single-file checkpoints.
/// Loads a Flux checkpoint, runs the full pipeline (CLIP-L pooled + T5-XXL → FluxTransformer → VAE decode),
/// and saves the output as a BMP file.
///
/// WARNING: These tests are VERY SLOW on CPU (12B param model). Use GPU backend for reasonable run times.
/// Set FLUX_SINGLE_FILE_PATH, CLIP_VOCAB_PATH, CLIP_MERGES_PATH, and T5_SPIECE_MODEL_PATH
/// environment variables or use defaults.
/// </summary>
public sealed class FluxGenerationTests
{
    private static string PickPath(string envVar, string winPath, string linuxPath)
    {
        string? v = Environment.GetEnvironmentVariable(envVar);
        if (!string.IsNullOrEmpty(v)) return v;
        return OperatingSystem.IsWindows() ? winPath : linuxPath;
    }

    private static readonly string LinuxRepoRoot =
        Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", ".."));

    private static readonly string FluxSingleFilePath = PickPath(
        "FLUX_SINGLE_FILE_PATH",
        @"C:\Users\kaleb\Desktop\Projects\SwarmUI\Models\Stable-Diffusion\Flux\flux1-schnell-fp8.safetensors",
        Path.Combine(LinuxRepoRoot, "Models", "flux1-schnell-fp8.safetensors"));

    private static readonly string FluxDevFp8Path = PickPath(
        "FLUX_DEV_FP8_PATH",
        @"C:\Users\kaleb\Desktop\Projects\SwarmUI\Models\Stable-Diffusion\Flux\flux1-dev-fp8.safetensors",
        Path.Combine(LinuxRepoRoot, "Models", "flux1-dev-fp8.safetensors"));

    private static readonly string FluxKreaFp8ScaledPath = PickPath(
        "FLUX_KREA_FP8_PATH",
        @"C:\Users\kaleb\Desktop\Projects\SwarmUI\Models\Stable-Diffusion\Flux\flux1-krea-dev_fp8_scaled.safetensors",
        Path.Combine(LinuxRepoRoot, "Models", "flux1-krea-dev_fp8_scaled.safetensors"));

    private static readonly string TokenizerVocabPath = PickPath(
        "CLIP_VOCAB_PATH",
        @"C:\Users\kaleb\Desktop\projects\SharpInference\tests\test-models\clip_vocab.json",
        Path.Combine(LinuxRepoRoot, "tests", "test-models", "clip_vocab.json"));

    private static readonly string TokenizerMergesPath = PickPath(
        "CLIP_MERGES_PATH",
        @"C:\Users\kaleb\Desktop\projects\SharpInference\tests\test-models\clip_merges.txt",
        Path.Combine(LinuxRepoRoot, "tests", "test-models", "clip_merges.txt"));

    private static readonly string T5SpieceModelPath = PickPath(
        "T5_SPIECE_MODEL_PATH",
        @"C:\Users\kaleb\Desktop\projects\SharpInference\tests\test-models\t5_spiece.model",
        Path.Combine(LinuxRepoRoot, "tests", "test-models", "t5_spiece.model"));

    private static readonly string OutputDir = PickPath(
        "FLUX_OUTPUT_DIR",
        @"C:\Users\kaleb\Desktop\projects\SharpInference\Output",
        Path.Combine(LinuxRepoRoot, "Output"));

    private readonly ITestOutputHelper _output;

    public FluxGenerationTests(ITestOutputHelper output) => _output = output;

    /// <summary>
    /// Full end-to-end Flux Schnell image generation from a single-file checkpoint.
    /// Small resolution (256x256) and 4 steps to keep test time manageable.
    /// </summary>
    [Fact]
    public void Schnell_SingleFile_GenerateImage_Small()
    {
        if (!File.Exists(FluxSingleFilePath))
        {
            _output.WriteLine($"SKIPPED: Checkpoint not found: {FluxSingleFilePath}");
            return;
        }
        if (!File.Exists(TokenizerVocabPath) || !File.Exists(TokenizerMergesPath))
        {
            _output.WriteLine("SKIPPED: CLIP tokenizer files not found");
            return;
        }
        if (!File.Exists(T5SpieceModelPath))
        {
            _output.WriteLine("SKIPPED: T5 SentencePiece model not found");
            return;
        }

        Stopwatch totalSw = Stopwatch.StartNew();

        // 1. Load and convert checkpoint
        _output.WriteLine($"[1/7] Loading checkpoint: {Path.GetFileName(FluxSingleFilePath)}");
        Stopwatch sw = Stopwatch.StartNew();
        (FluxCheckpointConverter.ConvertedWeights converted, SafeTensorsLoader loader) =
            FluxCheckpointConverter.LoadAndConvert(FluxSingleFilePath);
        sw.Stop();
        _output.WriteLine($"  Loaded in {sw.ElapsedMilliseconds}ms");
        _output.WriteLine($"  Transformer: {converted.Transformer.Count} keys");
        _output.WriteLine($"  CLIP-L: {converted.ClipL.Count} keys");
        _output.WriteLine($"  T5: {converted.T5.Count} keys");
        _output.WriteLine($"  VAE: {converted.Vae.Count} keys");

        using (loader)
        {
            Dictionary<string, Tensor> transformerF32 = CastWeightsToF32(converted.Transformer);
            Dictionary<string, Tensor> clipLF32 = CastWeightsToF32(converted.ClipL);
            Dictionary<string, Tensor> t5F32 = CastWeightsToF32(converted.T5);
            Dictionary<string, Tensor> vaeF32 = CastWeightsToF32(converted.Vae);

            CpuBackend backend = new CpuBackend();

            // 2. Detect architecture and create config
            (int doubleBlocks, int singleBlocks, bool hasGuidance) =
                FluxCheckpointConverter.DetectArchitecture(converted.Transformer);
            _output.WriteLine($"[2/7] Architecture: {doubleBlocks} double + {singleBlocks} single blocks, guidance={hasGuidance}");

            FluxConfig config = hasGuidance ? FluxConfig.Dev : FluxConfig.Schnell;

            // 3. Load transformer
            _output.WriteLine("[3/7] Loading FluxTransformer...");
            sw.Restart();
            FluxTransformer transformer = new FluxTransformer(config);
            transformer.LoadWeights(transformerF32);
            sw.Stop();
            _output.WriteLine($"  Transformer loaded in {sw.ElapsedMilliseconds}ms");

            // 4. Load CLIP-L
            _output.WriteLine("[4/7] Loading CLIP-L...");
            sw.Restart();
            ClipTextEncoder clipL = new ClipTextEncoder(ClipTextEncoderConfig.SdxlClipL);
            clipL.LoadWeights(clipLF32, "text_model");
            sw.Stop();
            _output.WriteLine($"  CLIP-L loaded in {sw.ElapsedMilliseconds}ms");

            // 5. Load T5-XXL
            _output.WriteLine("[5/7] Loading T5-XXL...");
            sw.Restart();
            T5TextEncoder t5 = new T5TextEncoder(T5TextEncoderConfig.Xxl);
            t5.LoadWeights(t5F32);
            sw.Stop();
            _output.WriteLine($"  T5-XXL loaded in {sw.ElapsedMilliseconds}ms");

            // 6. Load VAE
            _output.WriteLine("[6/7] Loading VAE...");
            sw.Restart();
            VaeDecoder vaeDecoder = new VaeDecoder(VaeConfig.Flux);
            vaeDecoder.LoadWeights(vaeF32);
            sw.Stop();
            _output.WriteLine($"  VAE loaded in {sw.ElapsedMilliseconds}ms");

            // 7. Tokenize and generate
            _output.WriteLine("[7/7] Tokenizing and generating...");

            // CLIP-L tokenizer
            using ClipTokenizer clipTokenizer = new ClipTokenizer(TokenizerVocabPath, TokenizerMergesPath);
            string prompt = "A photograph of an astronaut riding a horse";
            int[] clipTokenIds = clipTokenizer.Encode(prompt);
            int eosPosition = ClipTokenizer.FindEosPosition(clipTokenIds);

            // T5 tokenizer (256 max for Schnell)
            using T5Tokenizer t5Tokenizer = new T5Tokenizer(T5SpieceModelPath, maxLength: 256);
            int[] t5TokenIds = t5Tokenizer.Encode(prompt);
            int[] t5AttentionMask = T5Tokenizer.CreateAttentionMask(t5TokenIds);

            TextToImageRequest request = new TextToImageRequest
            {
                Prompt = prompt,
                Width = 256,
                Height = 256,
                Steps = 4,
                Seed = 42,
            };

            FluxPipeline pipeline = new FluxPipeline(backend, clipL, t5, transformer, vaeDecoder, config);

            sw.Restart();
            (byte[] rgbData, int width, int height, int seed) = pipeline.GenerateFromTokens(
                clipTokenIds, eosPosition, t5TokenIds, t5AttentionMask,
                request, guidanceScale: 0.0f,
                onProgress: p => _output.WriteLine($"  Step {p.Step}/{p.TotalSteps} ({p.ElapsedMs:F0}ms)"));
            sw.Stop();

            _output.WriteLine($"Generation done in {sw.ElapsedMilliseconds}ms (seed={seed})");

            // Save output
            Directory.CreateDirectory(OutputDir);
            string outputPath = Path.Combine(OutputDir, $"flux_schnell_{width}x{height}_s{request.Steps}_seed{seed}.bmp");
            ImagePostProcessor.SaveBmp(outputPath, rgbData, width, height);
            _output.WriteLine($"Saved: {outputPath}");

            Assert.Equal(256, width);
            Assert.Equal(256, height);
            Assert.Equal(256 * 256 * 3, rgbData.Length);

            // Check not all black or all white
            ValidateImageNotDegenerate(rgbData);

            totalSw.Stop();
            _output.WriteLine($"\nTotal test time: {totalSw.ElapsedMilliseconds}ms");

            pipeline.Dispose();
            transformer.Dispose();
            t5.Dispose();
        }
    }

    /// <summary>
    /// GPU end-to-end Flux Schnell image generation. Uses CudaBackend with auto-transfer
    /// pattern (no weight preloading) — weights stay in system RAM, streamed to GPU per-op.
    /// F32 for now since many Flux operations still use CPU pointer code.
    /// </summary>
    [Fact]
    public void Schnell_SingleFile_GenerateImage_Gpu()
    {
        if (!File.Exists(FluxSingleFilePath))
        {
            _output.WriteLine($"SKIPPED: Checkpoint not found: {FluxSingleFilePath}");
            return;
        }
        if (!File.Exists(TokenizerVocabPath) || !File.Exists(TokenizerMergesPath))
        {
            _output.WriteLine("SKIPPED: CLIP tokenizer files not found");
            return;
        }
        if (!File.Exists(T5SpieceModelPath))
        {
            _output.WriteLine("SKIPPED: T5 SentencePiece model not found");
            return;
        }

        Stopwatch totalSw = Stopwatch.StartNew();

        // 1. Load and convert checkpoint
        _output.WriteLine($"[1/7] Loading checkpoint: {Path.GetFileName(FluxSingleFilePath)}");
        Stopwatch sw = Stopwatch.StartNew();
        (FluxCheckpointConverter.ConvertedWeights converted, SafeTensorsLoader loader) =
            FluxCheckpointConverter.LoadAndConvert(FluxSingleFilePath);
        sw.Stop();
        _output.WriteLine($"  Loaded in {sw.ElapsedMilliseconds}ms");

        using (loader)
        {
            // FP8 Schnell single-file checkpoint contains FP8 transformer + FP8 T5. Casting all
            // to F32 would explode the 12B FP8 transformer to ~48GB and OOM a 32GB system.
            // Pass FP8/F16 weights as-is — model.LoadWeights auto-casts the few sites that
            // dereference float* (QkNorm, CLIP embeddings, T5 norms) to F32 internally.
            // Linear weights stay native FP8 for cuBLAS auto-cast (CUDA path) or F32-on-demand (CPU path).
            // Only VAE (small, F32) is cast for safety.
            Dictionary<string, Tensor> vaeF32 = CastWeightsToF32(converted.Vae);

            // GPU backend — auto-transfer, no weight preloading (12B params won't fit in VRAM)
            string assemblyDir = Path.GetDirectoryName(typeof(FluxGenerationTests).Assembly.Location)!;
            string ptxDir = Path.Combine(assemblyDir, "Ptx");
            CudaBackend backend = Directory.Exists(ptxDir)
                ? new CudaBackend(deviceOrdinal: 0, ptxDir: ptxDir)
                : new CudaBackend();

            // 2. Detect architecture
            (int doubleBlocks, int singleBlocks, bool hasGuidance) =
                FluxCheckpointConverter.DetectArchitecture(converted.Transformer);
            _output.WriteLine($"[2/7] Architecture: {doubleBlocks} double + {singleBlocks} single blocks, guidance={hasGuidance}");

            FluxConfig config = hasGuidance ? FluxConfig.Dev : FluxConfig.Schnell;

            // 3. Load transformer (FP8 native — auto-casts QkNorm only)
            _output.WriteLine("[3/7] Loading FluxTransformer (FP8 native)...");
            sw.Restart();
            FluxTransformer transformer = new FluxTransformer(config);
            transformer.LoadWeights(converted.Transformer);
            sw.Stop();
            _output.WriteLine($"  Transformer loaded in {sw.ElapsedMilliseconds}ms");

            // 4. Load CLIP-L (auto-casts all layer weights to F32 internally)
            _output.WriteLine("[4/7] Loading CLIP-L...");
            sw.Restart();
            ClipTextEncoder clipL = new ClipTextEncoder(ClipTextEncoderConfig.SdxlClipL);
            clipL.LoadWeights(converted.ClipL, "text_model");
            sw.Stop();
            _output.WriteLine($"  CLIP-L loaded in {sw.ElapsedMilliseconds}ms");

            // 5. Load T5-XXL (FP8 native — auto-casts norms/embeddings to F32, linear weights stay FP8)
            _output.WriteLine("[5/7] Loading T5-XXL (FP8 native)...");
            sw.Restart();
            T5TextEncoder t5 = new T5TextEncoder(T5TextEncoderConfig.Xxl);
            t5.LoadWeights(converted.T5);
            sw.Stop();
            _output.WriteLine($"  T5-XXL loaded in {sw.ElapsedMilliseconds}ms");

            // 6. Load VAE
            _output.WriteLine("[6/7] Loading VAE...");
            sw.Restart();
            VaeDecoder vaeDecoder = new VaeDecoder(VaeConfig.Flux);
            vaeDecoder.LoadWeights(vaeF32);
            sw.Stop();
            _output.WriteLine($"  VAE loaded in {sw.ElapsedMilliseconds}ms");

            // 7. Tokenize and generate
            _output.WriteLine("[7/7] Tokenizing and generating (GPU)...");

            using ClipTokenizer clipTokenizer = new ClipTokenizer(TokenizerVocabPath, TokenizerMergesPath);
            string prompt = "A photograph of an astronaut riding a horse";
            int[] clipTokenIds = clipTokenizer.Encode(prompt);
            int eosPosition = ClipTokenizer.FindEosPosition(clipTokenIds);

            using T5Tokenizer t5Tokenizer = new T5Tokenizer(T5SpieceModelPath, maxLength: 256);
            int[] t5TokenIds = t5Tokenizer.Encode(prompt);
            int[] t5AttentionMask = T5Tokenizer.CreateAttentionMask(t5TokenIds);

            // Schnell is distilled — designed for 1–4 steps with no CFG. 512×512 for direct comparison with Dev FP8 result.
            TextToImageRequest request = new TextToImageRequest
            {
                Prompt = prompt,
                Width = 512,
                Height = 512,
                Steps = 4,
                Seed = 42,
            };

            FluxPipeline pipeline = new FluxPipeline(backend, clipL, t5, transformer, vaeDecoder, config);

            sw.Restart();
            (byte[] rgbData, int width, int height, int seed) = pipeline.GenerateFromTokens(
                clipTokenIds, eosPosition, t5TokenIds, t5AttentionMask,
                request, guidanceScale: 0.0f,
                onProgress: p => _output.WriteLine($"  Step {p.Step}/{p.TotalSteps} ({p.ElapsedMs:F0}ms)"));

            backend.Sync();
            sw.Stop();

            _output.WriteLine($"Generation done in {sw.ElapsedMilliseconds}ms (seed={seed})");

            // Save output
            Directory.CreateDirectory(OutputDir);
            string outputPath = Path.Combine(OutputDir, $"flux_schnell_gpu_{width}x{height}_s{request.Steps}_seed{seed}.bmp");
            ImagePostProcessor.SaveBmp(outputPath, rgbData, width, height);
            _output.WriteLine($"Saved: {outputPath}");

            Assert.Equal(512, width);
            Assert.Equal(512, height);
            Assert.Equal(512 * 512 * 3, rgbData.Length);

            ValidateImageNotDegenerate(rgbData);

            totalSw.Stop();
            _output.WriteLine($"\nTotal test time: {totalSw.ElapsedMilliseconds}ms");

            pipeline.Dispose();
            transformer.Dispose();
            t5.Dispose();
            backend.Dispose();
        }
    }

    /// <summary>
    /// GPU end-to-end Flux Dev FP8 image generation. Uses CudaBackend with auto-transfer
    /// pattern (no weight preloading). FP8 weights stay native in RAM (1 byte/element),
    /// CudaBackend auto-casts F8→F16 per-GEMM. Model LoadWeights auto-casts weights
    /// accessed via float* DataPointer (QkNorm, CLIP embeddings, T5 norms/embeddings) to F32.
    /// </summary>
    [Fact]
    public void Dev_Fp8_GenerateImage_Gpu()
    {
        if (!File.Exists(FluxDevFp8Path))
        {
            _output.WriteLine($"SKIPPED: Checkpoint not found: {FluxDevFp8Path}");
            return;
        }
        if (!File.Exists(TokenizerVocabPath) || !File.Exists(TokenizerMergesPath))
        {
            _output.WriteLine("SKIPPED: CLIP tokenizer files not found");
            return;
        }
        if (!File.Exists(T5SpieceModelPath))
        {
            _output.WriteLine("SKIPPED: T5 SentencePiece model not found");
            return;
        }

        // Find PTX directory
        string assemblyDir = Path.GetDirectoryName(typeof(FluxGenerationTests).Assembly.Location)!;
        string ptxDir = Path.Combine(assemblyDir, "Ptx");
        if (!Directory.Exists(ptxDir))
        {
            _output.WriteLine($"SKIPPED: PTX directory not found: {ptxDir}");
            return;
        }

        Stopwatch totalSw = Stopwatch.StartNew();

        // 1. Load and convert checkpoint
        _output.WriteLine($"[1/7] Loading FP8 checkpoint: {Path.GetFileName(FluxDevFp8Path)}");
        Stopwatch sw = Stopwatch.StartNew();
        (FluxCheckpointConverter.ConvertedWeights converted, SafeTensorsLoader loader) =
            FluxCheckpointConverter.LoadAndConvert(FluxDevFp8Path);
        sw.Stop();
        _output.WriteLine($"  Loaded in {sw.ElapsedMilliseconds}ms");
        _output.WriteLine($"  Transformer: {converted.Transformer.Count} keys");
        _output.WriteLine($"  CLIP-L: {converted.ClipL.Count} keys");
        _output.WriteLine($"  T5: {converted.T5.Count} keys");
        _output.WriteLine($"  VAE: {converted.Vae.Count} keys");

        // Log dtype distribution
        LogDTypeDistribution("Transformer", converted.Transformer);
        LogDTypeDistribution("CLIP-L", converted.ClipL);
        LogDTypeDistribution("T5", converted.T5);
        LogDTypeDistribution("VAE", converted.Vae);

        using (loader)
        {
            // FP8 weights: pass as-is. Model LoadWeights auto-casts weights accessed via float*
            // (QkNorm, CLIP embeddings/projections/layer weights, T5 norms/embeddings).
            // Linear weights stay native FP8 — CudaBackend auto-casts F8→F16 per-GEMM.
            // VAE weights cast to F32 for safety (VAE is small ~160MB).
            Dictionary<string, Tensor> vaeF32 = CastWeightsToF32(converted.Vae);

            // GPU backend — auto-transfer, no weight preloading (12B params won't fit in VRAM)
            CudaBackend backend = new CudaBackend(deviceOrdinal: 0, ptxDir: ptxDir);

            // 2. Detect architecture
            (int doubleBlocks, int singleBlocks, bool hasGuidance) =
                FluxCheckpointConverter.DetectArchitecture(converted.Transformer);
            _output.WriteLine($"[2/7] Architecture: {doubleBlocks} double + {singleBlocks} single blocks, guidance={hasGuidance}");
            Assert.True(hasGuidance, "Expected Dev architecture with guidance embedding");

            FluxConfig config = FluxConfig.Dev;

            // 3. Load transformer (FP8 weights — auto-casts QkNorm only)
            _output.WriteLine("[3/7] Loading FluxTransformer (FP8)...");
            sw.Restart();
            FluxTransformer transformer = new FluxTransformer(config);
            transformer.LoadWeights(converted.Transformer);
            sw.Stop();
            _output.WriteLine($"  Transformer loaded in {sw.ElapsedMilliseconds}ms");

            // 4. Load CLIP-L (auto-casts all layer weights to F32 internally)
            _output.WriteLine("[4/7] Loading CLIP-L...");
            sw.Restart();
            ClipTextEncoder clipL = new ClipTextEncoder(ClipTextEncoderConfig.SdxlClipL);
            clipL.LoadWeights(converted.ClipL, "text_model");
            sw.Stop();
            _output.WriteLine($"  CLIP-L loaded in {sw.ElapsedMilliseconds}ms");

            // 5. Load T5-XXL (auto-casts norms/embeddings to F32, linear weights stay FP8)
            _output.WriteLine("[5/7] Loading T5-XXL (FP8)...");
            sw.Restart();
            T5TextEncoder t5 = new T5TextEncoder(T5TextEncoderConfig.Xxl);
            t5.LoadWeights(converted.T5);
            sw.Stop();
            _output.WriteLine($"  T5-XXL loaded in {sw.ElapsedMilliseconds}ms");

            // 6. Load VAE (F32 — VAE must never be FP8)
            _output.WriteLine("[6/7] Loading VAE...");
            sw.Restart();
            VaeDecoder vaeDecoder = new VaeDecoder(VaeConfig.Flux);
            vaeDecoder.LoadWeights(vaeF32);
            sw.Stop();
            _output.WriteLine($"  VAE loaded in {sw.ElapsedMilliseconds}ms");

            // 7. Tokenize and generate
            _output.WriteLine("[7/7] Tokenizing and generating (GPU, FP8)...");

            using ClipTokenizer clipTokenizer = new ClipTokenizer(TokenizerVocabPath, TokenizerMergesPath);
            string prompt = "A photograph of an astronaut riding a horse";
            int[] clipTokenIds = clipTokenizer.Encode(prompt);
            int eosPosition = ClipTokenizer.FindEosPosition(clipTokenIds);

            // T5 with 512 max length for Dev
            using T5Tokenizer t5Tokenizer = new T5Tokenizer(T5SpieceModelPath, maxLength: 512);
            int[] t5TokenIds = t5Tokenizer.Encode(prompt);
            int[] t5AttentionMask = T5Tokenizer.CreateAttentionMask(t5TokenIds);

            // Dev needs 20+ steps (4 is for Schnell only). 512x512 minimum for coherent output.
            // Steps reduced to 10 for faster diagnostic run.
            TextToImageRequest request = new TextToImageRequest
            {
                Prompt = prompt,
                Width = 512,
                Height = 512,
                Steps = 10,
                Seed = 42,
            };

            FluxPipeline pipeline = new FluxPipeline(backend, clipL, t5, transformer, vaeDecoder, config);

            // Enable debug latent dump for Python cross-validation
            Environment.SetEnvironmentVariable("FLUX_DEBUG_DIR", OutputDir);

            sw.Restart();
            (byte[] rgbData, int width, int height, int seed) = pipeline.GenerateFromTokens(
                clipTokenIds, eosPosition, t5TokenIds, t5AttentionMask,
                request, guidanceScale: 3.5f,
                onProgress: p => _output.WriteLine($"  Step {p.Step}/{p.TotalSteps} ({p.ElapsedMs:F0}ms)"));

            backend.Sync();
            sw.Stop();

            _output.WriteLine($"Generation done in {sw.ElapsedMilliseconds}ms (seed={seed})");

            // Save output
            Directory.CreateDirectory(OutputDir);
            string outputPath = Path.Combine(OutputDir, $"flux_dev_fp8_gpu_{width}x{height}_s{request.Steps}_seed{seed}.bmp");
            ImagePostProcessor.SaveBmp(outputPath, rgbData, width, height);
            _output.WriteLine($"Saved: {outputPath}");

            Assert.Equal(512, width);
            Assert.Equal(512, height);
            Assert.Equal(512 * 512 * 3, rgbData.Length);

            ValidateImageNotDegenerate(rgbData);
            LogPerChannelRgbStats(rgbData, width, height);

            totalSw.Stop();
            _output.WriteLine($"\nTotal test time: {totalSw.ElapsedMilliseconds}ms");

            pipeline.Dispose();
            transformer.Dispose();
            t5.Dispose();
            backend.Dispose();
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

    /// <summary>
    /// Flux.1 Krea Dev (photoreal fine-tune of Dev) using ComfyUI's <c>fp8_scaled</c> single-file
    /// transformer. The Krea checkpoint stores only the transformer (no CLIP-L / T5 / VAE), so we
    /// pair it with the all-in-one Dev FP8 file for the encoders + VAE. The fp8_scaled format adds
    /// per-tensor scalar <c>scale_weight</c> companions that <see cref="FluxCheckpointConverter"/>
    /// folds into <see cref="Tensor.Fp8ScaleFactor"/> on each weight, then <c>CudaBackend.Linear</c>
    /// applies the scale via cuBLAS' <c>alpha</c> parameter at GEMM time (zero memory overhead vs
    /// dequanting to F16 at load). Uses FluxConfig.Dev (Krea has guidance embedding).
    /// </summary>
    [Fact]
    public void Krea_Fp8Scaled_GenerateImage_Gpu()
    {
        if (!File.Exists(FluxKreaFp8ScaledPath))
        {
            _output.WriteLine($"SKIPPED: Krea checkpoint not found: {FluxKreaFp8ScaledPath}");
            return;
        }
        if (!File.Exists(FluxDevFp8Path))
        {
            _output.WriteLine($"SKIPPED: Dev FP8 (for shared encoders+VAE) not found: {FluxDevFp8Path}");
            return;
        }
        if (!File.Exists(TokenizerVocabPath) || !File.Exists(TokenizerMergesPath))
        {
            _output.WriteLine("SKIPPED: CLIP tokenizer files not found");
            return;
        }
        if (!File.Exists(T5SpieceModelPath))
        {
            _output.WriteLine("SKIPPED: T5 SentencePiece model not found");
            return;
        }

        string assemblyDir = Path.GetDirectoryName(typeof(FluxGenerationTests).Assembly.Location)!;
        string ptxDir = Path.Combine(assemblyDir, "Ptx");
        if (!Directory.Exists(ptxDir))
        {
            _output.WriteLine($"SKIPPED: PTX directory not found: {ptxDir}");
            return;
        }

        Stopwatch totalSw = Stopwatch.StartNew();

        // 1a. Load + convert Krea (transformer-only, fp8_scaled — converter folds scale_weight into Fp8ScaleFactor)
        _output.WriteLine($"[1/7] Loading Krea checkpoint: {Path.GetFileName(FluxKreaFp8ScaledPath)}");
        Stopwatch sw = Stopwatch.StartNew();
        (FluxCheckpointConverter.ConvertedWeights kreaConverted, SafeTensorsLoader kreaLoader) =
            FluxCheckpointConverter.LoadAndConvert(FluxKreaFp8ScaledPath);
        sw.Stop();
        _output.WriteLine($"  Krea loaded in {sw.ElapsedMilliseconds}ms ({kreaConverted.Transformer.Count} transformer keys)");
        LogDTypeDistribution("Krea Transformer (FP8 native + scale_weight folded into Fp8ScaleFactor)", kreaConverted.Transformer);

        // 1b. Load Dev FP8 for the shared encoders + VAE
        _output.WriteLine($"[1b/7] Loading Dev FP8 for shared encoders+VAE: {Path.GetFileName(FluxDevFp8Path)}");
        sw.Restart();
        (FluxCheckpointConverter.ConvertedWeights devConverted, SafeTensorsLoader devLoader) =
            FluxCheckpointConverter.LoadAndConvert(FluxDevFp8Path);
        sw.Stop();
        _output.WriteLine($"  Dev FP8 loaded in {sw.ElapsedMilliseconds}ms");

        using (kreaLoader)
        using (devLoader)
        {
            Dictionary<string, Tensor> vaeF32 = CastWeightsToF32(devConverted.Vae);

            CudaBackend backend = new CudaBackend(deviceOrdinal: 0, ptxDir: ptxDir);

            // 2. Detect architecture from Krea — should be Dev-style with guidance embedder
            (int doubleBlocks, int singleBlocks, bool hasGuidance) =
                FluxCheckpointConverter.DetectArchitecture(kreaConverted.Transformer);
            _output.WriteLine($"[2/7] Architecture: {doubleBlocks} double + {singleBlocks} single blocks, guidance={hasGuidance}");
            Assert.True(hasGuidance, "Expected Krea to be a Dev-architecture variant with guidance embedding");

            FluxConfig config = FluxConfig.Dev;

            // 3. Load Krea transformer (FP8 native; scale_weight stored on each tensor's Fp8ScaleFactor)
            _output.WriteLine("[3/7] Loading FluxTransformer (Krea FP8 native + per-tensor scale)...");
            sw.Restart();
            FluxTransformer transformer = new FluxTransformer(config);
            transformer.LoadWeights(kreaConverted.Transformer);
            sw.Stop();
            _output.WriteLine($"  Transformer loaded in {sw.ElapsedMilliseconds}ms");

            // 4. Load CLIP-L from Dev (auto-casts internally)
            _output.WriteLine("[4/7] Loading CLIP-L from Dev FP8...");
            sw.Restart();
            ClipTextEncoder clipL = new ClipTextEncoder(ClipTextEncoderConfig.SdxlClipL);
            clipL.LoadWeights(devConverted.ClipL, "text_model");
            sw.Stop();
            _output.WriteLine($"  CLIP-L loaded in {sw.ElapsedMilliseconds}ms");

            // 5. Load T5-XXL from Dev (FP8 native — model auto-casts internally)
            _output.WriteLine("[5/7] Loading T5-XXL from Dev FP8...");
            sw.Restart();
            T5TextEncoder t5 = new T5TextEncoder(T5TextEncoderConfig.Xxl);
            t5.LoadWeights(devConverted.T5);
            sw.Stop();
            _output.WriteLine($"  T5-XXL loaded in {sw.ElapsedMilliseconds}ms");

            // 6. Load VAE from Dev
            _output.WriteLine("[6/7] Loading VAE from Dev FP8...");
            sw.Restart();
            VaeDecoder vaeDecoder = new VaeDecoder(VaeConfig.Flux);
            vaeDecoder.LoadWeights(vaeF32);
            sw.Stop();
            _output.WriteLine($"  VAE loaded in {sw.ElapsedMilliseconds}ms");

            // 7. Tokenize + generate
            _output.WriteLine("[7/7] Tokenizing and generating (GPU, Krea fp8_scaled → F16)...");

            using ClipTokenizer clipTokenizer = new ClipTokenizer(TokenizerVocabPath, TokenizerMergesPath);
            string prompt = "A photograph of an astronaut riding a horse";
            int[] clipTokenIds = clipTokenizer.Encode(prompt);
            int eosPosition = ClipTokenizer.FindEosPosition(clipTokenIds);

            using T5Tokenizer t5Tokenizer = new T5Tokenizer(T5SpieceModelPath, maxLength: 512);
            int[] t5TokenIds = t5Tokenizer.Encode(prompt);
            int[] t5AttentionMask = T5Tokenizer.CreateAttentionMask(t5TokenIds);

            // Krea is a Dev fine-tune — same step count and guidance recommendations.
            TextToImageRequest request = new TextToImageRequest
            {
                Prompt = prompt,
                Width = 512,
                Height = 512,
                Steps = 10,
                Seed = 42,
            };

            FluxPipeline pipeline = new FluxPipeline(backend, clipL, t5, transformer, vaeDecoder, config);

            sw.Restart();
            (byte[] rgbData, int width, int height, int seed) = pipeline.GenerateFromTokens(
                clipTokenIds, eosPosition, t5TokenIds, t5AttentionMask,
                request, guidanceScale: 3.5f,
                onProgress: p => _output.WriteLine($"  Step {p.Step}/{p.TotalSteps} ({p.ElapsedMs:F0}ms)"));

            backend.Sync();
            sw.Stop();

            _output.WriteLine($"Generation done in {sw.ElapsedMilliseconds}ms (seed={seed})");

            Directory.CreateDirectory(OutputDir);
            string outputPath = Path.Combine(OutputDir, $"flux_krea_fp8scaled_gpu_{width}x{height}_s{request.Steps}_seed{seed}.bmp");
            ImagePostProcessor.SaveBmp(outputPath, rgbData, width, height);
            _output.WriteLine($"Saved: {outputPath}");

            Assert.Equal(512, width);
            Assert.Equal(512, height);
            Assert.Equal(512 * 512 * 3, rgbData.Length);

            ValidateImageNotDegenerate(rgbData);
            LogPerChannelRgbStats(rgbData, width, height);

            totalSw.Stop();
            _output.WriteLine($"\nTotal test time: {totalSw.ElapsedMilliseconds}ms");

            pipeline.Dispose();
            transformer.Dispose();
            t5.Dispose();
            backend.Dispose();
        }
    }

    /// <summary>
    /// Same as Dev_Fp8 but casts ALL weights to F32 before loading.
    /// This isolates whether the issue is FP8-specific or a general Flux model bug.
    /// </summary>
    [Fact]
    public void Dev_Fp8_AllF32_GenerateImage_Gpu()
    {
        if (!File.Exists(FluxDevFp8Path))
        {
            _output.WriteLine($"SKIPPED: Checkpoint not found: {FluxDevFp8Path}");
            return;
        }
        if (!File.Exists(TokenizerVocabPath) || !File.Exists(TokenizerMergesPath) || !File.Exists(T5SpieceModelPath))
        {
            _output.WriteLine("SKIPPED: Tokenizer files not found");
            return;
        }

        string assemblyDir = Path.GetDirectoryName(typeof(FluxGenerationTests).Assembly.Location)!;
        string ptxDir = Path.Combine(assemblyDir, "Ptx");
        if (!Directory.Exists(ptxDir))
        {
            _output.WriteLine($"SKIPPED: PTX directory not found: {ptxDir}");
            return;
        }

        Stopwatch totalSw = Stopwatch.StartNew();

        _output.WriteLine($"[1/7] Loading FP8 checkpoint (will cast ALL to F32): {Path.GetFileName(FluxDevFp8Path)}");
        Stopwatch sw = Stopwatch.StartNew();
        (FluxCheckpointConverter.ConvertedWeights converted, SafeTensorsLoader loader) =
            FluxCheckpointConverter.LoadAndConvert(FluxDevFp8Path);
        sw.Stop();
        _output.WriteLine($"  Loaded in {sw.ElapsedMilliseconds}ms");

        using (loader)
        {
            // Cast ALL weights to F32 — this eliminates FP8 as a variable
            _output.WriteLine("  Casting ALL weights to F32...");
            Dictionary<string, Tensor> transformerF32 = CastWeightsToF32(converted.Transformer);
            Dictionary<string, Tensor> clipLF32 = CastWeightsToF32(converted.ClipL);
            Dictionary<string, Tensor> t5F32 = CastWeightsToF32(converted.T5);
            Dictionary<string, Tensor> vaeF32 = CastWeightsToF32(converted.Vae);
            _output.WriteLine("  All weights cast to F32");

            CudaBackend backend = new CudaBackend(deviceOrdinal: 0, ptxDir: ptxDir);

            (int doubleBlocks, int singleBlocks, bool hasGuidance) =
                FluxCheckpointConverter.DetectArchitecture(converted.Transformer);
            _output.WriteLine($"[2/7] Architecture: {doubleBlocks} double + {singleBlocks} single blocks, guidance={hasGuidance}");

            FluxConfig config = FluxConfig.Dev;

            _output.WriteLine("[3/7] Loading FluxTransformer (F32)...");
            FluxTransformer transformer = new FluxTransformer(config);
            transformer.LoadWeights(transformerF32);

            _output.WriteLine("[4/7] Loading CLIP-L (F32)...");
            ClipTextEncoder clipL = new ClipTextEncoder(ClipTextEncoderConfig.SdxlClipL);
            clipL.LoadWeights(clipLF32, "text_model");

            _output.WriteLine("[5/7] Loading T5-XXL (F32)...");
            T5TextEncoder t5 = new T5TextEncoder(T5TextEncoderConfig.Xxl);
            t5.LoadWeights(t5F32);

            _output.WriteLine("[6/7] Loading VAE (F32)...");
            VaeDecoder vaeDecoder = new VaeDecoder(VaeConfig.Flux);
            vaeDecoder.LoadWeights(vaeF32);

            _output.WriteLine("[7/7] Tokenizing and generating (GPU, ALL F32)...");

            using ClipTokenizer clipTokenizer = new ClipTokenizer(TokenizerVocabPath, TokenizerMergesPath);
            string prompt = "A photograph of an astronaut riding a horse";
            int[] clipTokenIds = clipTokenizer.Encode(prompt);
            int eosPosition = ClipTokenizer.FindEosPosition(clipTokenIds);

            using T5Tokenizer t5Tokenizer = new T5Tokenizer(T5SpieceModelPath, maxLength: 512);
            int[] t5TokenIds = t5Tokenizer.Encode(prompt);
            int[] t5AttentionMask = T5Tokenizer.CreateAttentionMask(t5TokenIds);

            // Small resolution + few steps for quick comparison
            TextToImageRequest request = new TextToImageRequest
            {
                Prompt = prompt,
                Width = 256,
                Height = 256,
                Steps = 4,
                Seed = 42,
            };

            FluxPipeline pipeline = new FluxPipeline(backend, clipL, t5, transformer, vaeDecoder, config);

            sw.Restart();
            (byte[] rgbData, int width, int height, int seed) = pipeline.GenerateFromTokens(
                clipTokenIds, eosPosition, t5TokenIds, t5AttentionMask,
                request, guidanceScale: 3.5f,
                onProgress: p => _output.WriteLine($"  Step {p.Step}/{p.TotalSteps} ({p.ElapsedMs:F0}ms)"));

            backend.Sync();
            sw.Stop();

            _output.WriteLine($"Generation done in {sw.ElapsedMilliseconds}ms (seed={seed})");

            Directory.CreateDirectory(OutputDir);
            string outputPath = Path.Combine(OutputDir, $"flux_dev_allf32_gpu_{width}x{height}_s{request.Steps}_seed{seed}.bmp");
            ImagePostProcessor.SaveBmp(outputPath, rgbData, width, height);
            _output.WriteLine($"Saved: {outputPath}");

            ValidateImageNotDegenerate(rgbData);
            LogPerChannelRgbStats(rgbData, width, height);

            totalSw.Stop();
            _output.WriteLine($"\nTotal test time: {totalSw.ElapsedMilliseconds}ms");

            pipeline.Dispose();
            transformer.Dispose();
            t5.Dispose();
            backend.Dispose();
        }
    }

    private void LogPerChannelRgbStats(byte[] rgbData, int width, int height)
    {
        int pixels = width * height;
        long[] sums = [0, 0, 0];
        int[] mins = [255, 255, 255];
        int[] maxs = [0, 0, 0];
        for (int i = 0; i < pixels; i++)
        {
            for (int c = 0; c < 3; c++)
            {
                byte v = rgbData[i * 3 + c];
                sums[c] += v;
                if (v < mins[c]) mins[c] = v;
                if (v > maxs[c]) maxs[c] = v;
            }
        }
        string[] names = ["R", "G", "B"];
        for (int c = 0; c < 3; c++)
        {
            float mean = (float)sums[c] / pixels;
            _output.WriteLine($"  {names[c]}: min={mins[c]} max={maxs[c]} mean={mean:F1}");
        }
    }
}
