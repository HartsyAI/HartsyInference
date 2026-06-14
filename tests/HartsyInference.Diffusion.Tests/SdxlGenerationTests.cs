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
using HartsyInference.Diffusion.Schedulers;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.ModelHandler.CheckpointConverters;
using HartsyInference.ModelHandler.SafeTensors;
using HartsyInference.Tests.Common;
using HartsyInference.Tokenizers;

namespace HartsyInference.Diffusion.Tests;

/// <summary>
/// End-to-end SDXL image generation tests using single-file checkpoints.
/// Loads JuggernautXL, runs the full pipeline (dual CLIP → UNet denoise → VAE decode),
/// and saves the output as a BMP file.
///
/// WARNING: These tests are SLOW on CPU. A 256x256 image with 5 steps takes ~60+ minutes.
/// Set SDXL_SINGLE_FILE_PATH and CLIP_TOKENIZER_DIR environment variables or use defaults.
/// </summary>
public class SdxlGenerationTests
{
    private static string SdxlCheckpointPath => TestPaths.Sdxl.SingleFile;
    private static string TokenizerVocabPath => TestPaths.Tokenizers.ClipVocab;
    private static string TokenizerMergesPath => TestPaths.Tokenizers.ClipMerges;
    private static string OutputDir => TestPaths.OutputDir;

    private readonly ITestOutputHelper _output;

    public SdxlGenerationTests(ITestOutputHelper output) => _output = output;

    /// <summary>
    /// Full end-to-end SDXL image generation from a single-file checkpoint.
    /// Loads all components, runs dual CLIP encoding, UNet denoising loop, VAE decode, saves BMP.
    /// Uses small resolution (128x128) and minimal steps (3) to keep CPU time manageable.
    /// </summary>
    [Fact]
    public void SingleFile_GenerateImage_Small()
    {
        if (!File.Exists(SdxlCheckpointPath))
        {
            _output.WriteLine($"SKIPPED: Checkpoint not found: {SdxlCheckpointPath}");
            return;
        }
        if (!File.Exists(TokenizerVocabPath) || !File.Exists(TokenizerMergesPath))
        {
            _output.WriteLine("SKIPPED: Tokenizer files not found");
            return;
        }

        Stopwatch totalSw = Stopwatch.StartNew();

        // 1. Load and convert checkpoint
        _output.WriteLine($"[1/6] Loading checkpoint: {Path.GetFileName(SdxlCheckpointPath)}");
        Stopwatch sw = Stopwatch.StartNew();
        (SdxlCheckpointConverter.ConvertedWeights converted, SafeTensorsLoader loader) =
            SdxlCheckpointConverter.LoadAndConvert(SdxlCheckpointPath);
        sw.Stop();
        _output.WriteLine($"  Checkpoint loaded and converted in {sw.ElapsedMilliseconds}ms");

        using (loader)
        {
            Dictionary<string, Tensor> unetF32 = CastWeightsToF32(converted.UNet);
            Dictionary<string, Tensor> clipLF32 = CastWeightsToF32(converted.ClipL);
            Dictionary<string, Tensor> clipGF32 = CastWeightsToF32(converted.ClipG);
            Dictionary<string, Tensor> vaeF32 = CastWeightsToF32(converted.Vae);

            // 2. Tokenize
            _output.WriteLine("[2/6] Tokenizing prompt...");
            using ClipTokenizer tokenizer = new(TokenizerVocabPath, TokenizerMergesPath);

            string prompt = "a majestic lion in a field of sunflowers, golden hour, photorealistic";
            string negPrompt = "blurry, low quality, deformed";

            int[] promptTokensL = tokenizer.Encode(prompt);
            int[] negTokensL = tokenizer.Encode(negPrompt);
            int[] promptTokensG = tokenizer.Encode(prompt);
            int[] negTokensG = tokenizer.Encode(negPrompt);

            int promptEosG = ClipTokenizer.FindEosPosition(promptTokensG);
            int negEosG = ClipTokenizer.FindEosPosition(negTokensG);
            _output.WriteLine($"  Prompt EOS position: {promptEosG}, Negative EOS: {negEosG}");

            // 3. Load CLIP-L
            _output.WriteLine("[3/6] Loading CLIP-L...");
            sw.Restart();
            ClipTextEncoder clipL = new(ClipTextEncoderConfig.SdxlClipL);
            clipL.LoadWeights(clipLF32, "text_model");
            sw.Stop();
            _output.WriteLine($"  CLIP-L loaded in {sw.ElapsedMilliseconds}ms");

            // 4. Load CLIP-G
            _output.WriteLine("[4/6] Loading CLIP-G...");
            sw.Restart();
            ClipTextEncoder clipG = new(ClipTextEncoderConfig.SdxlClipG);
            clipG.LoadWeights(clipGF32, "text_model");
            sw.Stop();
            _output.WriteLine($"  CLIP-G loaded in {sw.ElapsedMilliseconds}ms");

            // 5. Load UNet
            _output.WriteLine("[5/6] Loading UNet...");
            sw.Restart();
            UNet unet = new(UNetConfig.SdxlBase);
            unet.LoadWeights(unetF32);
            sw.Stop();
            _output.WriteLine($"  UNet loaded in {sw.ElapsedMilliseconds}ms");

            // 6. Load VAE
            _output.WriteLine("[6/6] Loading VAE...");
            sw.Restart();
            VaeDecoder vae = new(VaeConfig.Sdxl);
            vae.LoadWeights(vaeF32);
            sw.Stop();
            _output.WriteLine($"  VAE loaded in {sw.ElapsedMilliseconds}ms");

            // Create pipeline and generate
            using CpuBackend backend = new();
            using SdxlPipeline pipeline = new(backend, clipL, clipG, unet, vae);

            TextToImageRequest request = new()
            {
                Prompt = prompt,
                NegativePrompt = negPrompt,
                Width = 128,
                Height = 128,
                Steps = 3,
                CfgScale = 7.0f,
                Seed = 42,
            };

            _output.WriteLine($"\nGenerating {request.Width}x{request.Height} image, {request.Steps} steps, cfg={request.CfgScale}, seed=42...");
            Stopwatch genSw = Stopwatch.StartNew();

            (byte[] rgbData, int width, int height, int seed) = pipeline.GenerateFromTokens(
                promptTokensL, negTokensL,
                promptTokensG, negTokensG,
                promptEosG, negEosG,
                request,
                progress => _output.WriteLine($"  Step {progress.Step}/{progress.TotalSteps} ({progress.ElapsedMs:F0}ms)"));

            genSw.Stop();
            _output.WriteLine($"\nGeneration complete in {genSw.Elapsed.TotalMinutes:F1} minutes (seed={seed})");

            // Validate output
            Assert.Equal(128, width);
            Assert.Equal(128, height);
            Assert.Equal(128 * 128 * 3, rgbData.Length);

            // Check not all black or all white
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

            // Save output
            Directory.CreateDirectory(OutputDir);
            string outputPath = Path.Combine(OutputDir, $"sdxl_test_{DateTime.Now:yyyyMMdd_HHmmss}.bmp");
            ImagePostProcessor.SaveBmp(outputPath, rgbData, width, height);
            _output.WriteLine($"  Image saved to: {outputPath}");

            totalSw.Stop();
            _output.WriteLine($"\nTotal time: {totalSw.Elapsed.TotalMinutes:F1} minutes");
        }
    }

    /// <summary>
    /// Full end-to-end SDXL image generation on GPU at native resolution.
    /// Loads JuggernautXL, runs dual CLIP → UNet denoise → VAE decode on CUDA backend.
    /// 1024x1024 with 20 steps — should complete in minutes on a modern GPU.
    /// </summary>
    [Fact]
    public void Gpu_SingleFile_GenerateImage_1024()
    {
        if (!File.Exists(SdxlCheckpointPath))
        {
            _output.WriteLine($"SKIPPED: Checkpoint not found: {SdxlCheckpointPath}");
            return;
        }
        if (!File.Exists(TokenizerVocabPath) || !File.Exists(TokenizerMergesPath))
        {
            _output.WriteLine("SKIPPED: Tokenizer files not found");
            return;
        }

        // Find PTX directory relative to test assembly output
        string assemblyDir = Path.GetDirectoryName(typeof(SdxlGenerationTests).Assembly.Location)!;
        string ptxDir = Path.Combine(assemblyDir, "Ptx");
        if (!Directory.Exists(ptxDir))
        {
            _output.WriteLine($"SKIPPED: PTX directory not found: {ptxDir}");
            return;
        }

        Stopwatch totalSw = Stopwatch.StartNew();

        // 1. Load and convert checkpoint
        _output.WriteLine($"[1/7] Loading checkpoint: {Path.GetFileName(SdxlCheckpointPath)}");
        Stopwatch sw = Stopwatch.StartNew();
        (SdxlCheckpointConverter.ConvertedWeights converted, SafeTensorsLoader loader) =
            SdxlCheckpointConverter.LoadAndConvert(SdxlCheckpointPath);
        sw.Stop();
        _output.WriteLine($"  Checkpoint loaded and converted in {sw.ElapsedMilliseconds}ms");

        using (loader)
        {
            Dictionary<string, Tensor> unetF32 = CastWeightsToF32(converted.UNet);
            Dictionary<string, Tensor> clipLF32 = CastWeightsToF32(converted.ClipL);
            Dictionary<string, Tensor> clipGF32 = CastWeightsToF32(converted.ClipG);
            Dictionary<string, Tensor> vaeF32 = CastWeightsToF32(converted.Vae);

            // 2. Tokenize
            _output.WriteLine("[2/7] Tokenizing prompt...");
            using ClipTokenizer tokenizer = new(TokenizerVocabPath, TokenizerMergesPath);

            string prompt = "a majestic lion in a field of sunflowers, golden hour, photorealistic";
            string negPrompt = "blurry, low quality, deformed";

            int[] promptTokensL = tokenizer.Encode(prompt);
            int[] negTokensL = tokenizer.Encode(negPrompt);
            int[] promptTokensG = tokenizer.Encode(prompt);
            int[] negTokensG = tokenizer.Encode(negPrompt);

            int promptEosG = ClipTokenizer.FindEosPosition(promptTokensG);
            int negEosG = ClipTokenizer.FindEosPosition(negTokensG);
            _output.WriteLine($"  Prompt EOS position: {promptEosG}, Negative EOS: {negEosG}");

            // 3. Load CLIP-L
            _output.WriteLine("[3/7] Loading CLIP-L...");
            sw.Restart();
            ClipTextEncoder clipL = new(ClipTextEncoderConfig.SdxlClipL);
            clipL.LoadWeights(clipLF32, "text_model");
            sw.Stop();
            _output.WriteLine($"  CLIP-L loaded in {sw.ElapsedMilliseconds}ms");

            // 4. Load CLIP-G
            _output.WriteLine("[4/7] Loading CLIP-G...");
            sw.Restart();
            ClipTextEncoder clipG = new(ClipTextEncoderConfig.SdxlClipG);
            clipG.LoadWeights(clipGF32, "text_model");
            sw.Stop();
            _output.WriteLine($"  CLIP-G loaded in {sw.ElapsedMilliseconds}ms");

            // 5. Load UNet
            _output.WriteLine("[5/7] Loading UNet...");
            sw.Restart();
            UNet unet = new(UNetConfig.SdxlBase);
            unet.LoadWeights(unetF32);
            sw.Stop();
            _output.WriteLine($"  UNet loaded in {sw.ElapsedMilliseconds}ms");

            // 6. Load VAE
            _output.WriteLine("[6/7] Loading VAE...");
            sw.Restart();
            VaeDecoder vae = new(VaeConfig.Sdxl);
            vae.LoadWeights(vaeF32);
            sw.Stop();
            _output.WriteLine($"  VAE loaded in {sw.ElapsedMilliseconds}ms");

            // 7. Create CUDA backend and pipeline
            _output.WriteLine("[7/7] Initializing CUDA backend...");
            sw.Restart();
            using CudaBackend backend = new(deviceOrdinal: 0, ptxDir: ptxDir);
            sw.Stop();
            _output.WriteLine($"  CUDA backend initialized in {sw.ElapsedMilliseconds}ms");

            using SdxlPipeline pipeline = new(backend, clipL, clipG, unet, vae);

            // Preload UNet + VAE weights to GPU, free CPU copies
            _output.WriteLine("\n[GPU] Preloading UNet + VAE weights to GPU...");
            sw.Restart();
            backend.PreloadWeights(unet.EnumerateWeights());
            backend.PreloadWeights(vae.EnumerateWeights());
            (long cachedBytes, long _, long _) = backend.GetGpuCacheStats();
            sw.Stop();
            _output.WriteLine($"  Preloaded {cachedBytes / 1024.0 / 1024.0:F1} MB to GPU in {sw.ElapsedMilliseconds}ms");

            // Free CPU weight memory (GPU cache holds copies; disposed tensors still work via cache)
            foreach (Tensor tensor in unetF32.Values) tensor.Dispose();
            foreach (Tensor tensor in vaeF32.Values) tensor.Dispose();
            unetF32.Clear();
            vaeF32.Clear();
            _output.WriteLine("  CPU weight memory freed (CLIP retained on CPU)");

            TextToImageRequest request = new()
            {
                Prompt = prompt,
                NegativePrompt = negPrompt,
                Width = 1024,
                Height = 1024,
                Steps = 20,
                CfgScale = 7.0f,
                Seed = 42,
            };

            _output.WriteLine($"\nGenerating {request.Width}x{request.Height} image, {request.Steps} steps, cfg={request.CfgScale}, seed=42 [GPU]...");
            Stopwatch genSw = Stopwatch.StartNew();

            (byte[] rgbData, int width, int height, int seed) = pipeline.GenerateFromTokens(
                promptTokensL, negTokensL,
                promptTokensG, negTokensG,
                promptEosG, negEosG,
                request,
                progress => _output.WriteLine($"  Step {progress.Step}/{progress.TotalSteps} ({progress.ElapsedMs:F0}ms)"));

            genSw.Stop();
            (long finalCached, long hits, long misses) = backend.GetGpuCacheStats();
            _output.WriteLine($"\nGeneration complete in {genSw.Elapsed.TotalSeconds:F1} seconds (seed={seed})");
            _output.WriteLine($"  GPU cache: {finalCached / 1024.0 / 1024.0:F1} MB, hits={hits}, misses={misses}");

            // Free GPU weight cache
            backend.FreePreloadedWeights();

            // Validate output
            Assert.Equal(1024, width);
            Assert.Equal(1024, height);
            Assert.Equal(1024 * 1024 * 3, rgbData.Length);

            // Check not all black or all white
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

            // Save output
            Directory.CreateDirectory(OutputDir);
            string outputPath = Path.Combine(OutputDir, $"sdxl_gpu_1024_{DateTime.Now:yyyyMMdd_HHmmss}.bmp");
            ImagePostProcessor.SaveBmp(outputPath, rgbData, width, height);
            _output.WriteLine($"  Image saved to: {outputPath}");

            totalSw.Stop();
            _output.WriteLine($"\nTotal time: {totalSw.Elapsed.TotalSeconds:F1} seconds");
        }
    }

    /// <summary>
    /// SDXL GPU generation at small resolution for fast iteration.
    /// 256x256 with 5 steps using auto-transfer CudaBackend.
    /// All weights stay on CPU; each IBackend op transparently copies H2D, executes, copies D2H.
    /// </summary>
    [Fact]
    public void Gpu_SingleFile_GenerateImage_Small()
    {
        if (!File.Exists(SdxlCheckpointPath))
        {
            _output.WriteLine($"SKIPPED: Checkpoint not found: {SdxlCheckpointPath}");
            return;
        }
        if (!File.Exists(TokenizerVocabPath) || !File.Exists(TokenizerMergesPath))
        {
            _output.WriteLine("SKIPPED: Tokenizer files not found");
            return;
        }

        string assemblyDir = Path.GetDirectoryName(typeof(SdxlGenerationTests).Assembly.Location)!;
        string ptxDir = Path.Combine(assemblyDir, "Ptx");
        if (!Directory.Exists(ptxDir))
        {
            _output.WriteLine($"SKIPPED: PTX directory not found: {ptxDir}");
            return;
        }

        Stopwatch totalSw = Stopwatch.StartNew();

        _output.WriteLine($"[1/7] Loading checkpoint: {Path.GetFileName(SdxlCheckpointPath)}");
        Stopwatch sw = Stopwatch.StartNew();
        (SdxlCheckpointConverter.ConvertedWeights converted, SafeTensorsLoader loader) =
            SdxlCheckpointConverter.LoadAndConvert(SdxlCheckpointPath);
        sw.Stop();
        _output.WriteLine($"  Checkpoint loaded and converted in {sw.ElapsedMilliseconds}ms");

        using (loader)
        {
            Dictionary<string, Tensor> unetF32 = CastWeightsToF32(converted.UNet);
            Dictionary<string, Tensor> clipLF32 = CastWeightsToF32(converted.ClipL);
            Dictionary<string, Tensor> clipGF32 = CastWeightsToF32(converted.ClipG);
            Dictionary<string, Tensor> vaeF32 = CastWeightsToF32(converted.Vae);

            _output.WriteLine("[2/7] Tokenizing prompt...");
            using ClipTokenizer tokenizer = new(TokenizerVocabPath, TokenizerMergesPath);

            string prompt = "a majestic lion in a field of sunflowers, golden hour, photorealistic";
            string negPrompt = "blurry, low quality, deformed";

            int[] promptTokensL = tokenizer.Encode(prompt);
            int[] negTokensL = tokenizer.Encode(negPrompt);
            int[] promptTokensG = tokenizer.Encode(prompt);
            int[] negTokensG = tokenizer.Encode(negPrompt);

            int promptEosG = ClipTokenizer.FindEosPosition(promptTokensG);
            int negEosG = ClipTokenizer.FindEosPosition(negTokensG);
            _output.WriteLine($"  Prompt EOS position: {promptEosG}, Negative EOS: {negEosG}");

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

            _output.WriteLine("[7/7] Initializing CUDA backend (auto-transfer mode)...");
            sw.Restart();
            using CudaBackend backend = new(deviceOrdinal: 0, ptxDir: ptxDir);
            sw.Stop();
            _output.WriteLine($"  CUDA backend initialized in {sw.ElapsedMilliseconds}ms");
            _output.WriteLine($"  Device: {backend.Capabilities.Name}");

            using SdxlPipeline pipeline = new(backend, clipL, clipG, unet, vae);

            TextToImageRequest request = new()
            {
                Prompt = prompt,
                NegativePrompt = negPrompt,
                Width = 256,
                Height = 256,
                Steps = 10,
                CfgScale = 7.0f,
                Seed = 42,
            };

            _output.WriteLine($"\nGenerating {request.Width}x{request.Height} image, {request.Steps} step, cfg={request.CfgScale}, seed=42 [GPU auto-transfer]...");
            Stopwatch genSw = Stopwatch.StartNew();

            (byte[] rgbData, int width, int height, int seed) = pipeline.GenerateFromTokens(
                promptTokensL, negTokensL,
                promptTokensG, negTokensG,
                promptEosG, negEosG,
                request,
                progress => _output.WriteLine($"  Step {progress.Step}/{progress.TotalSteps} ({progress.ElapsedMs:F0}ms)"));

            genSw.Stop();
            _output.WriteLine($"\nGeneration complete in {genSw.Elapsed.TotalSeconds:F1} seconds (seed={seed})");

            Assert.Equal(256, width);
            Assert.Equal(256, height);
            Assert.Equal(256 * 256 * 3, rgbData.Length);

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

            Directory.CreateDirectory(OutputDir);
            string outputPath = Path.Combine(OutputDir, $"sdxl_gpu_256_{DateTime.Now:yyyyMMdd_HHmmss}.bmp");
            ImagePostProcessor.SaveBmp(outputPath, rgbData, width, height);
            _output.WriteLine($"  Image saved to: {outputPath}");

            totalSw.Stop();
            _output.WriteLine($"\nTotal time: {totalSw.Elapsed.TotalSeconds:F1} seconds");
        }
    }

    /// <summary>
    /// FP16 GPU generation test: passes native F16 weights to UNet/VAE (no F32 cast).
    /// CLIP stays F32 (CPU-side). Tests the full Phase 3 FP16 inference path.
    /// 256x256 with 10 steps for quick smoke test.
    /// </summary>
    [Fact]
    public void Gpu_F16_GenerateImage_256()
    {
        if (!File.Exists(SdxlCheckpointPath))
        {
            _output.WriteLine($"SKIPPED: Checkpoint not found: {SdxlCheckpointPath}");
            return;
        }
        if (!File.Exists(TokenizerVocabPath) || !File.Exists(TokenizerMergesPath))
        {
            _output.WriteLine("SKIPPED: Tokenizer files not found");
            return;
        }

        string assemblyDir = Path.GetDirectoryName(typeof(SdxlGenerationTests).Assembly.Location)!;
        string ptxDir = Path.Combine(assemblyDir, "Ptx");
        if (!Directory.Exists(ptxDir))
        {
            _output.WriteLine($"SKIPPED: PTX directory not found: {ptxDir}");
            return;
        }

        Stopwatch totalSw = Stopwatch.StartNew();

        _output.WriteLine($"[1/7] Loading checkpoint: {Path.GetFileName(SdxlCheckpointPath)}");
        Stopwatch sw = Stopwatch.StartNew();
        (SdxlCheckpointConverter.ConvertedWeights converted, SafeTensorsLoader loader) =
            SdxlCheckpointConverter.LoadAndConvert(SdxlCheckpointPath);
        sw.Stop();
        _output.WriteLine($"  Checkpoint loaded and converted in {sw.ElapsedMilliseconds}ms");

        using (loader)
        {
            // CLIP stays F32 (CPU-side text encoding)
            Dictionary<string, Tensor> clipLF32 = CastWeightsToF32(converted.ClipL);
            Dictionary<string, Tensor> clipGF32 = CastWeightsToF32(converted.ClipG);

            // UNet and VAE: cast ALL weights to F16 (checkpoints may have mixed F16/F32)
            Dictionary<string, Tensor> unetWeights = CastWeightsToF16(converted.UNet);
            Dictionary<string, Tensor> vaeWeights = CastWeightsToF16(converted.Vae);

            // Count how many weights needed casting
            int unetCastCount = converted.UNet.Count(kv => kv.Value.DType != DType.F16);
            int vaeCastCount = converted.Vae.Count(kv => kv.Value.DType != DType.F16);
            _output.WriteLine($"  UNet: {unetCastCount}/{converted.UNet.Count} weights cast to F16");
            _output.WriteLine($"  VAE: {vaeCastCount}/{converted.Vae.Count} weights cast to F16");

            _output.WriteLine("[2/7] Tokenizing prompt...");
            using ClipTokenizer tokenizer = new(TokenizerVocabPath, TokenizerMergesPath);

            string prompt = "a majestic lion in a field of sunflowers, golden hour, photorealistic";
            string negPrompt = "blurry, low quality, deformed";

            int[] promptTokensL = tokenizer.Encode(prompt);
            int[] negTokensL = tokenizer.Encode(negPrompt);
            int[] promptTokensG = tokenizer.Encode(prompt);
            int[] negTokensG = tokenizer.Encode(negPrompt);

            int promptEosG = ClipTokenizer.FindEosPosition(promptTokensG);
            int negEosG = ClipTokenizer.FindEosPosition(negTokensG);

            _output.WriteLine("[3/7] Loading CLIP-L (F32)...");
            sw.Restart();
            ClipTextEncoder clipL = new(ClipTextEncoderConfig.SdxlClipL);
            clipL.LoadWeights(clipLF32, "text_model");
            sw.Stop();
            _output.WriteLine($"  CLIP-L loaded in {sw.ElapsedMilliseconds}ms");

            _output.WriteLine("[4/7] Loading CLIP-G (F32)...");
            sw.Restart();
            ClipTextEncoder clipG = new(ClipTextEncoderConfig.SdxlClipG);
            clipG.LoadWeights(clipGF32, "text_model");
            sw.Stop();
            _output.WriteLine($"  CLIP-G loaded in {sw.ElapsedMilliseconds}ms");

            _output.WriteLine("[5/7] Loading UNet (F16 weights)...");
            sw.Restart();
            UNet unet = new(UNetConfig.SdxlBase);
            unet.LoadWeights(unetWeights);
            sw.Stop();
            _output.WriteLine($"  UNet loaded in {sw.ElapsedMilliseconds}ms");

            _output.WriteLine("[6/7] Loading VAE (F16 weights)...");
            sw.Restart();
            VaeDecoder vae = new(VaeConfig.Sdxl);
            vae.LoadWeights(vaeWeights);
            sw.Stop();
            _output.WriteLine($"  VAE loaded in {sw.ElapsedMilliseconds}ms");

            _output.WriteLine("[7/7] Initializing CUDA backend...");
            sw.Restart();
            using CudaBackend backend = new(deviceOrdinal: 0, ptxDir: ptxDir);
            sw.Stop();
            _output.WriteLine($"  CUDA backend initialized in {sw.ElapsedMilliseconds}ms");

            using SdxlPipeline pipeline = new(backend, clipL, clipG, unet, vae);

            // Preload F16 weights to GPU
            _output.WriteLine("\n[GPU] Preloading F16 UNet + VAE weights to GPU...");
            sw.Restart();
            backend.PreloadWeights(unet.EnumerateWeights());
            backend.PreloadWeights(vae.EnumerateWeights());
            (long cachedBytes, long _, long _) = backend.GetGpuCacheStats();
            sw.Stop();
            _output.WriteLine($"  Preloaded {cachedBytes / 1024.0 / 1024.0:F1} MB to GPU in {sw.ElapsedMilliseconds}ms (should be ~half of F32)");

            TextToImageRequest request = new()
            {
                Prompt = prompt,
                NegativePrompt = negPrompt,
                Width = 256,
                Height = 256,
                Steps = 10,
                CfgScale = 7.0f,
                Seed = 42,
            };

            _output.WriteLine($"\nGenerating {request.Width}x{request.Height} image, {request.Steps} steps, cfg={request.CfgScale}, seed=42 [GPU F16]...");
            Stopwatch genSw = Stopwatch.StartNew();

            (byte[] rgbData, int width, int height, int seed) = pipeline.GenerateFromTokens(
                promptTokensL, negTokensL,
                promptTokensG, negTokensG,
                promptEosG, negEosG,
                request,
                progress => _output.WriteLine($"  Step {progress.Step}/{progress.TotalSteps} ({progress.ElapsedMs:F0}ms)"));

            genSw.Stop();
            (long finalCached, long hits, long misses) = backend.GetGpuCacheStats();
            _output.WriteLine($"\nGeneration complete in {genSw.Elapsed.TotalSeconds:F1} seconds (seed={seed})");
            _output.WriteLine($"  GPU cache: {finalCached / 1024.0 / 1024.0:F1} MB, hits={hits}, misses={misses}");

            backend.FreePreloadedWeights();

            // Validate output
            Assert.Equal(256, width);
            Assert.Equal(256, height);
            Assert.Equal(256 * 256 * 3, rgbData.Length);

            // Check not all black or all white (NaN/Inf would produce all-zero or all-255)
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
            Assert.True(nonZeroPct > 10, "Image appears to be all black (possible NaN/Inf in F16 path)");
            Assert.True(nonFFPct > 10, "Image appears to be all white (possible F16 overflow)");

            // Save output
            Directory.CreateDirectory(OutputDir);
            string outputPath = Path.Combine(OutputDir, $"sdxl_gpu_f16_256_{DateTime.Now:yyyyMMdd_HHmmss}.bmp");
            ImagePostProcessor.SaveBmp(outputPath, rgbData, width, height);
            _output.WriteLine($"  Image saved to: {outputPath}");

            totalSw.Stop();
            _output.WriteLine($"\nTotal time: {totalSw.Elapsed.TotalSeconds:F1} seconds");
        }
    }

    /// <summary>
    /// FP16 GPU generation at full 1024x1024 resolution, 20 steps.
    /// Tests Phase 3 FP16 inference at production resolution.
    /// </summary>
    [Fact]
    public void Gpu_F16_GenerateImage_1024()
    {
        if (!File.Exists(SdxlCheckpointPath))
        {
            _output.WriteLine($"SKIPPED: Checkpoint not found: {SdxlCheckpointPath}");
            return;
        }
        if (!File.Exists(TokenizerVocabPath) || !File.Exists(TokenizerMergesPath))
        {
            _output.WriteLine("SKIPPED: Tokenizer files not found");
            return;
        }

        string assemblyDir = Path.GetDirectoryName(typeof(SdxlGenerationTests).Assembly.Location)!;
        string ptxDir = Path.Combine(assemblyDir, "Ptx");
        if (!Directory.Exists(ptxDir))
        {
            _output.WriteLine($"SKIPPED: PTX directory not found: {ptxDir}");
            return;
        }

        Stopwatch totalSw = Stopwatch.StartNew();

        _output.WriteLine($"[1/7] Loading checkpoint: {Path.GetFileName(SdxlCheckpointPath)}");
        Stopwatch sw = Stopwatch.StartNew();
        (SdxlCheckpointConverter.ConvertedWeights converted, SafeTensorsLoader loader) =
            SdxlCheckpointConverter.LoadAndConvert(SdxlCheckpointPath);
        sw.Stop();
        _output.WriteLine($"  Checkpoint loaded and converted in {sw.ElapsedMilliseconds}ms");

        using (loader)
        {
            // CLIP stays F32 (CPU-side text encoding)
            Dictionary<string, Tensor> clipLF32 = CastWeightsToF32(converted.ClipL);
            Dictionary<string, Tensor> clipGF32 = CastWeightsToF32(converted.ClipG);

            // UNet and VAE: cast ALL weights to F16 (checkpoints may have mixed F16/F32)
            Dictionary<string, Tensor> unetWeights = CastWeightsToF16(converted.UNet);
            Dictionary<string, Tensor> vaeWeights = CastWeightsToF16(converted.Vae);

            int unetCastCount = converted.UNet.Count(kv => kv.Value.DType != DType.F16);
            int vaeCastCount = converted.Vae.Count(kv => kv.Value.DType != DType.F16);
            _output.WriteLine($"  UNet: {unetCastCount}/{converted.UNet.Count} weights cast to F16");
            _output.WriteLine($"  VAE: {vaeCastCount}/{converted.Vae.Count} weights cast to F16");

            _output.WriteLine("[2/7] Tokenizing prompt...");
            using ClipTokenizer tokenizer = new(TokenizerVocabPath, TokenizerMergesPath);

            string prompt = "a majestic lion in a field of sunflowers, golden hour, photorealistic";
            string negPrompt = "blurry, low quality, deformed";

            int[] promptTokensL = tokenizer.Encode(prompt);
            int[] negTokensL = tokenizer.Encode(negPrompt);
            int[] promptTokensG = tokenizer.Encode(prompt);
            int[] negTokensG = tokenizer.Encode(negPrompt);

            int promptEosG = ClipTokenizer.FindEosPosition(promptTokensG);
            int negEosG = ClipTokenizer.FindEosPosition(negTokensG);

            _output.WriteLine("[3/7] Loading CLIP-L (F32)...");
            ClipTextEncoder clipL = new(ClipTextEncoderConfig.SdxlClipL);
            clipL.LoadWeights(clipLF32, "text_model");

            _output.WriteLine("[4/7] Loading CLIP-G (F32)...");
            ClipTextEncoder clipG = new(ClipTextEncoderConfig.SdxlClipG);
            clipG.LoadWeights(clipGF32, "text_model");

            _output.WriteLine("[5/7] Loading UNet (F16 weights)...");
            UNet unet = new(UNetConfig.SdxlBase);
            unet.LoadWeights(unetWeights);

            _output.WriteLine("[6/7] Loading VAE...");
            VaeDecoder vae = new(VaeConfig.Sdxl);
            vae.LoadWeights(vaeWeights);

            _output.WriteLine("[7/7] Initializing CUDA backend...");
            using CudaBackend backend = new(deviceOrdinal: 0, ptxDir: ptxDir);

            using SdxlPipeline pipeline = new(backend, clipL, clipG, unet, vae);

            // Preload weights to GPU
            _output.WriteLine("\n[GPU] Preloading UNet + VAE weights to GPU...");
            sw.Restart();
            backend.PreloadWeights(unet.EnumerateWeights());
            backend.PreloadWeights(vae.EnumerateWeights());
            (long cachedBytes, long _, long _) = backend.GetGpuCacheStats();
            sw.Stop();
            _output.WriteLine($"  Preloaded {cachedBytes / 1024.0 / 1024.0:F1} MB to GPU in {sw.ElapsedMilliseconds}ms");

            TextToImageRequest request = new()
            {
                Prompt = prompt,
                NegativePrompt = negPrompt,
                Width = 1024,
                Height = 1024,
                Steps = 20,
                CfgScale = 7.0f,
                Seed = 42,
            };

            _output.WriteLine($"\nGenerating {request.Width}x{request.Height} image, {request.Steps} steps, cfg={request.CfgScale}, seed=42 [GPU F16]...");
            Stopwatch genSw = Stopwatch.StartNew();

            (byte[] rgbData, int width, int height, int seed) = pipeline.GenerateFromTokens(
                promptTokensL, negTokensL,
                promptTokensG, negTokensG,
                promptEosG, negEosG,
                request,
                progress => _output.WriteLine($"  Step {progress.Step}/{progress.TotalSteps} ({progress.ElapsedMs:F0}ms)"));

            genSw.Stop();
            (long finalCached, long hits, long misses) = backend.GetGpuCacheStats();
            _output.WriteLine($"\nGeneration complete in {genSw.Elapsed.TotalSeconds:F1} seconds (seed={seed})");
            _output.WriteLine($"  GPU cache: {finalCached / 1024.0 / 1024.0:F1} MB, hits={hits}, misses={misses}");

            backend.FreePreloadedWeights();

            Assert.Equal(1024, width);
            Assert.Equal(1024, height);
            Assert.Equal(1024 * 1024 * 3, rgbData.Length);

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

            Directory.CreateDirectory(OutputDir);
            string outputPath = Path.Combine(OutputDir, $"sdxl_gpu_f16_1024_{DateTime.Now:yyyyMMdd_HHmmss}.bmp");
            ImagePostProcessor.SaveBmp(outputPath, rgbData, width, height);
            _output.WriteLine($"  Image saved to: {outputPath}");

            totalSw.Stop();
            _output.WriteLine($"\nTotal time: {totalSw.Elapsed.TotalSeconds:F1} seconds");
        }
    }

    /// <summary>
    /// Diagnostic: dumps text embedding and per-step latent stats for comparison with Python reference.
    /// Runs on CPU to avoid GPU-specific issues. Prints stats in same format as Python dump_sdxl_reference_stats.py.
    /// </summary>
    [Fact]
    public unsafe void Diagnostic_DumpPipelineStats()
    {
        if (!File.Exists(SdxlCheckpointPath))
        {
            _output.WriteLine($"SKIPPED: Checkpoint not found: {SdxlCheckpointPath}");
            return;
        }
        if (!File.Exists(TokenizerVocabPath) || !File.Exists(TokenizerMergesPath))
        {
            _output.WriteLine("SKIPPED: Tokenizer files not found");
            return;
        }

        // Load checkpoint
        _output.WriteLine("Loading checkpoint...");
        (SdxlCheckpointConverter.ConvertedWeights converted, SafeTensorsLoader loader) =
            SdxlCheckpointConverter.LoadAndConvert(SdxlCheckpointPath);

        using (loader)
        {
            Dictionary<string, Tensor> clipLF32 = CastWeightsToF32(converted.ClipL);
            Dictionary<string, Tensor> clipGF32 = CastWeightsToF32(converted.ClipG);
            Dictionary<string, Tensor> unetF32 = CastWeightsToF32(converted.UNet);
            Dictionary<string, Tensor> vaeF32 = CastWeightsToF32(converted.Vae);

            // Tokenize
            using ClipTokenizer tokenizer = new(TokenizerVocabPath, TokenizerMergesPath);
            string prompt = "a majestic lion in a field of sunflowers, golden hour, photorealistic";
            string negPrompt = "blurry, low quality, deformed";

            int[] promptTokensL = tokenizer.Encode(prompt);
            int[] negTokensL = tokenizer.Encode(negPrompt);
            int[] promptTokensG = tokenizer.Encode(prompt);
            int[] negTokensG = tokenizer.Encode(negPrompt);

            int promptEosG = ClipTokenizer.FindEosPosition(promptTokensG);
            int negEosG = ClipTokenizer.FindEosPosition(negTokensG);

            _output.WriteLine($"=== Token IDs ===");
            _output.WriteLine($"CLIP-L prompt: [{string.Join(", ", promptTokensL.Take(15))}...]");
            _output.WriteLine($"CLIP-L neg:    [{string.Join(", ", negTokensL.Take(10))}...]");
            _output.WriteLine($"CLIP-G EOS positions: prompt={promptEosG}, neg={negEosG}");

            // Load encoders
            using CpuBackend backend = new();

            ClipTextEncoder clipL = new(ClipTextEncoderConfig.SdxlClipL);
            clipL.LoadWeights(clipLF32, "text_model");

            ClipTextEncoder clipG = new(ClipTextEncoderConfig.SdxlClipG);
            clipG.LoadWeights(clipGF32, "text_model");

            // Encode with CLIP-L: batch [neg, pos]
            _output.WriteLine($"\n=== CLIP-L Encoding ===");
            int[][] batchTokenIdsL = [negTokensL, promptTokensL];
            (Tensor clipLHidden, _) = clipL.EncodePenultimate(backend, batchTokenIdsL, [0, 0]);
            DumpTensorStats(_output, "clip_l_penultimate", clipLHidden);

            // Encode with CLIP-G: batch [neg, pos]
            _output.WriteLine($"\n=== CLIP-G Encoding ===");
            int[][] batchTokenIdsG = [negTokensG, promptTokensG];
            int[] eosPositions = [negEosG, promptEosG];
            (Tensor clipGHidden, Tensor? pooledOutput) = clipG.EncodePenultimate(backend, batchTokenIdsG, eosPositions);
            DumpTensorStats(_output, "clip_g_penultimate", clipGHidden);
            if (pooledOutput != null)
                DumpTensorStats(_output, "clip_g_pooled", pooledOutput);

            // Concatenate hidden states
            _output.WriteLine($"\n=== Concatenated Text Embeddings ===");
            Tensor textEmbeddings = ConcatAlongLastDimDiag(clipLHidden, clipGHidden);
            DumpTensorStats(_output, "text_embeddings_concat", textEmbeddings);

            clipLHidden.Dispose();
            clipGHidden.Dispose();

            // Check text_projection weight shape
            if (clipGF32.TryGetValue("text_projection.weight", out Tensor? tpWeight))
            {
                _output.WriteLine($"\n=== text_projection weight ===");
                _output.WriteLine($"  shape: [{tpWeight.Shape[0]}, {tpWeight.Shape[1]}]");
                DumpTensorStats(_output, "text_projection_weight", tpWeight);
            }

            // Scheduler setup
            _output.WriteLine($"\n=== Scheduler ===");
            EulerDiscreteScheduler scheduler = new();
            scheduler.SetTimesteps(10);
            ReadOnlySpan<float> timesteps = scheduler.Timesteps;
            _output.WriteLine($"  timesteps: [{string.Join(", ", timesteps.ToArray().Select(t => t.ToString("F1")))}]");
            _output.WriteLine($"  init_noise_sigma: {scheduler.InitialNoiseSigma}");

            // Initial noise
            _output.WriteLine($"\n=== Initial Noise ===");
            TensorShape latentShape = new TensorShape(1, 4, 32, 32); // 256x256 / 8
            Tensor latent = SeedGenerator.CreateNoise(latentShape, 42);
            DumpTensorStats(_output, "initial_noise", latent);

            // Scale noise
            float initSigma = scheduler.InitialNoiseSigma;
            _output.WriteLine($"  Scaling by init_noise_sigma={initSigma}");
            Tensor scaled = new Tensor(latentShape, DType.F32);
            backend.Scale(scaled, latent, initSigma);
            latent.Dispose();
            latent = scaled;
            DumpTensorStats(_output, "scaled_noise", latent);

            // Load UNet for a single step diagnostic
            _output.WriteLine($"\n=== Loading UNet ===");
            UNet unet = new(UNetConfig.SdxlBase);
            unet.LoadWeights(unetF32);

            // ADM conditioning
            float[] sizeCondition = [256f, 256f, 0f, 0f, 256f, 256f];

            // Single UNet step (step 0)
            _output.WriteLine($"\n=== UNet Step 0 (t={timesteps[0]:F1}) ===");
            float inputScale = scheduler.ScaleModelInput(0);
            _output.WriteLine($"  ScaleModelInput: {inputScale}");

            Tensor scaledLatent;
            if (MathF.Abs(inputScale - 1.0f) > 1e-6f)
            {
                scaledLatent = new Tensor(latentShape, DType.F32);
                backend.Scale(scaledLatent, latent, inputScale);
            }
            else
            {
                scaledLatent = latent;
            }
            DumpTensorStats(_output, "step0_scaled_input", scaledLatent);

            // Extract uncond/cond text embeddings
            int seqLen = (int)textEmbeddings.Shape[1];
            int hiddenSize = (int)textEmbeddings.Shape[2];
            int pooledDim = (int)pooledOutput!.Shape[1];

            Tensor uncondEmb = SliceBatchElement(textEmbeddings, 0, seqLen, hiddenSize);
            Tensor condEmb = SliceBatchElement(textEmbeddings, 1, seqLen, hiddenSize);
            Tensor uncondPooled = SliceBatchElement1D(pooledOutput, 0, pooledDim);
            Tensor condPooled = SliceBatchElement1D(pooledOutput, 1, pooledDim);

            // UNet forward - uncond
            _output.WriteLine($"\n  Running UNet uncond...");
            Tensor uncondNoise = unet.Forward(backend, scaledLatent, timesteps[0], uncondEmb, uncondPooled, sizeCondition);
            DumpTensorStats(_output, "step0_noise_pred_uncond", uncondNoise);

            // UNet forward - cond
            _output.WriteLine($"\n  Running UNet cond...");
            Tensor condNoise = unet.Forward(backend, scaledLatent, timesteps[0], condEmb, condPooled, sizeCondition);
            DumpTensorStats(_output, "step0_noise_pred_cond", condNoise);

            // CFG blend
            float cfgScale = 7.0f;
            Tensor cfgNoise = new Tensor(latentShape, DType.F32);
            float* uncPtr = (float*)uncondNoise.DataPointer;
            float* conPtr = (float*)condNoise.DataPointer;
            float* outPtr = (float*)cfgNoise.DataPointer;
            int count = (int)latentShape.ElementCount;
            for (int i = 0; i < count; i++)
                outPtr[i] = uncPtr[i] + cfgScale * (conPtr[i] - uncPtr[i]);
            DumpTensorStats(_output, "step0_noise_pred_cfg", cfgNoise);

            // Scheduler step
            Tensor newLatent = new Tensor(latentShape, DType.F32);
            scheduler.Step(newLatent, cfgNoise, latent, 0);
            DumpTensorStats(_output, "step0_latents_after", newLatent);

            // Cleanup
            uncondEmb.Dispose();
            condEmb.Dispose();
            uncondPooled.Dispose();
            condPooled.Dispose();
            uncondNoise.Dispose();
            condNoise.Dispose();
            cfgNoise.Dispose();
            if (scaledLatent != latent) scaledLatent.Dispose();
            textEmbeddings.Dispose();
            pooledOutput.Dispose();
            latent.Dispose();
            newLatent.Dispose();

            _output.WriteLine("\n=== Done ===");
        }
    }

    private static unsafe void DumpTensorStats(ITestOutputHelper output, string name, Tensor t)
    {
        float* ptr = (float*)t.DataPointer;
        long count = t.ElementCount;

        double sum = 0, sumSq = 0;
        float min = float.MaxValue, max = float.MinValue;
        for (long i = 0; i < count; i++)
        {
            float v = ptr[i];
            sum += v;
            sumSq += (double)v * v;
            if (v < min) min = v;
            if (v > max) max = v;
        }
        double mean = sum / count;
        double variance = sumSq / count - mean * mean;
        double std = Math.Sqrt(Math.Max(0, variance));

        string shapeStr = t.Shape.Rank switch
        {
            1 => $"[{t.Shape[0]}]",
            2 => $"[{t.Shape[0]}, {t.Shape[1]}]",
            3 => $"[{t.Shape[0]}, {t.Shape[1]}, {t.Shape[2]}]",
            4 => $"[{t.Shape[0]}, {t.Shape[1]}, {t.Shape[2]}, {t.Shape[3]}]",
            _ => "[]"
        };

        string first8 = string.Join(", ", Enumerable.Range(0, (int)Math.Min(8, count)).Select(i => ptr[i].ToString("G6")));

        output.WriteLine($"  {name}: shape={shapeStr}, mean={mean:G6}, std={std:G6}, min={min:G6}, max={max:G6}");
        output.WriteLine($"    first_8: [{first8}]");
    }

    private static unsafe Tensor ConcatAlongLastDimDiag(Tensor a, Tensor b)
    {
        int batch = (int)a.Shape[0];
        int seqLen = (int)a.Shape[1];
        int dimA = (int)a.Shape[2];
        int dimB = (int)b.Shape[2];
        int dimOut = dimA + dimB;

        TensorShape outShape = new TensorShape(batch, seqLen, dimOut);
        Tensor output = new Tensor(outShape, DType.F32);

        float* aPtr = (float*)a.DataPointer;
        float* bPtr = (float*)b.DataPointer;
        float* outPtr = (float*)output.DataPointer;

        for (int bIdx = 0; bIdx < batch; bIdx++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                int aOffset = (bIdx * seqLen + s) * dimA;
                int bOffset = (bIdx * seqLen + s) * dimB;
                int outOffset = (bIdx * seqLen + s) * dimOut;

                for (int d = 0; d < dimA; d++)
                    outPtr[outOffset + d] = aPtr[aOffset + d];
                for (int d = 0; d < dimB; d++)
                    outPtr[outOffset + dimA + d] = bPtr[bOffset + d];
            }
        }

        return output;
    }

    private static unsafe Tensor SliceBatchElement(Tensor tensor, int batchIdx, int seqLen, int hiddenSize)
    {
        TensorShape shape = new TensorShape(1, seqLen, hiddenSize);
        Tensor slice = new Tensor(shape, DType.F32);
        float* srcPtr = (float*)tensor.DataPointer;
        float* dstPtr = (float*)slice.DataPointer;
        int elements = seqLen * hiddenSize;
        int srcOffset = batchIdx * elements;
        for (int i = 0; i < elements; i++)
            dstPtr[i] = srcPtr[srcOffset + i];
        return slice;
    }

    private static unsafe Tensor SliceBatchElement1D(Tensor tensor, int batchIdx, int dim)
    {
        TensorShape shape = new TensorShape(1, dim);
        Tensor slice = new Tensor(shape, DType.F32);
        float* srcPtr = (float*)tensor.DataPointer;
        float* dstPtr = (float*)slice.DataPointer;
        int srcOffset = batchIdx * dim;
        for (int i = 0; i < dim; i++)
            dstPtr[i] = srcPtr[srcOffset + i];
        return slice;
    }

    /// <summary>
    /// Cross-runtime validation: loads Python reference tensors and runs a single UNet forward
    /// pass on CPU, comparing output against Python's noise prediction.
    /// This eliminates RNG and tokenizer differences to isolate UNet bugs.
    /// </summary>
    [Fact]
    public unsafe void CrossRuntime_SingleUNetPassMatchesPython()
    {
        string refDir = Path.Combine(RepoRoot.Path, "tests", "python-reference", "sdxl_reference_tensors");
        if (!Directory.Exists(refDir))
        {
            _output.WriteLine($"SKIPPED: Reference tensors not found: {refDir}");
            return;
        }
        if (!File.Exists(SdxlCheckpointPath))
        {
            _output.WriteLine($"SKIPPED: Checkpoint not found: {SdxlCheckpointPath}");
            return;
        }

        // Load Python reference tensors
        _output.WriteLine("Loading Python reference tensors...");
        Tensor pyScaledInput = LoadBinaryTensor(Path.Combine(refDir, "step0_scaled_input.bin"), new TensorShape(1, 4, 32, 32));
        Tensor pyTextEmb = LoadBinaryTensor(Path.Combine(refDir, "text_embeddings.bin"), new TensorShape(2, 77, 2048));
        Tensor pyPooled = LoadBinaryTensor(Path.Combine(refDir, "clip_g_pooled.bin"), new TensorShape(2, 1280));
        Tensor pyNoiseUncond = LoadBinaryTensor(Path.Combine(refDir, "step0_noise_pred_uncond.bin"), new TensorShape(1, 4, 32, 32));
        Tensor pyNoiseCond = LoadBinaryTensor(Path.Combine(refDir, "step0_noise_pred_cond.bin"), new TensorShape(1, 4, 32, 32));

        DumpTensorStats(_output, "py_scaled_input", pyScaledInput);
        DumpTensorStats(_output, "py_text_emb", pyTextEmb);
        DumpTensorStats(_output, "py_pooled", pyPooled);
        DumpTensorStats(_output, "py_noise_uncond", pyNoiseUncond);
        DumpTensorStats(_output, "py_noise_cond", pyNoiseCond);

        // Load checkpoint and UNet
        _output.WriteLine("\nLoading checkpoint and UNet...");
        (SdxlCheckpointConverter.ConvertedWeights converted, SafeTensorsLoader loader) =
            SdxlCheckpointConverter.LoadAndConvert(SdxlCheckpointPath);

        using (loader)
        {
            Dictionary<string, Tensor> unetF32 = CastWeightsToF32(converted.UNet);

            UNet unet = new(UNetConfig.SdxlBase);
            unet.LoadWeights(unetF32);

            using CpuBackend backend = new();

            // ADM conditioning (same as Python: 256x256 target, no crop)
            float[] sizeCondition = [256f, 256f, 0f, 0f, 256f, 256f];

            // Split text embeddings: [0] = neg (uncond), [1] = pos (cond)
            int seqLen = (int)pyTextEmb.Shape[1];
            int hiddenSize = (int)pyTextEmb.Shape[2];
            int pooledDim = (int)pyPooled.Shape[1];

            Tensor uncondEmb = SliceBatchElement(pyTextEmb, 0, seqLen, hiddenSize);
            Tensor condEmb = SliceBatchElement(pyTextEmb, 1, seqLen, hiddenSize);
            Tensor uncondPooled = SliceBatchElement1D(pyPooled, 0, pooledDim);
            Tensor condPooled = SliceBatchElement1D(pyPooled, 1, pooledDim);

            // Run UNet forward - unconditional
            _output.WriteLine("\nRunning UNet uncond pass (CPU)...");
            Stopwatch sw = Stopwatch.StartNew();
            Tensor csNoiseUncond = unet.Forward(backend, pyScaledInput, 900.0f, uncondEmb, uncondPooled, sizeCondition);
            sw.Stop();
            _output.WriteLine($"  Done in {sw.Elapsed.TotalSeconds:F1}s");
            DumpTensorStats(_output, "cs_noise_uncond", csNoiseUncond);

            // Compare with Python reference
            _output.WriteLine("\n=== Comparison: uncond noise prediction ===");
            CompareWithReference(_output, "noise_pred_uncond", csNoiseUncond, pyNoiseUncond);

            // Run UNet forward - conditional
            _output.WriteLine("\nRunning UNet cond pass (CPU)...");
            sw.Restart();
            Tensor csNoiseCond = unet.Forward(backend, pyScaledInput, 900.0f, condEmb, condPooled, sizeCondition);
            sw.Stop();
            _output.WriteLine($"  Done in {sw.Elapsed.TotalSeconds:F1}s");
            DumpTensorStats(_output, "cs_noise_cond", csNoiseCond);

            _output.WriteLine("\n=== Comparison: cond noise prediction ===");
            CompareWithReference(_output, "noise_pred_cond", csNoiseCond, pyNoiseCond);

            // Cleanup
            uncondEmb.Dispose();
            condEmb.Dispose();
            uncondPooled.Dispose();
            condPooled.Dispose();
            csNoiseUncond.Dispose();
            csNoiseCond.Dispose();
        }

        pyScaledInput.Dispose();
        pyTextEmb.Dispose();
        pyPooled.Dispose();
        pyNoiseUncond.Dispose();
        pyNoiseCond.Dispose();

        _output.WriteLine("\n=== Done ===");
    }

    [Fact]
    public void CrossRuntime_SingleUNetPassMatchesPython_GPU()
    {
        // GPU variant: same test as CPU cross-runtime but using CudaBackend
        // This confirms whether the GPU auto-transfer backend produces correct results

        if (!File.Exists(SdxlCheckpointPath))
        {
            _output.WriteLine($"SKIPPED: Checkpoint not found: {SdxlCheckpointPath}");
            return;
        }

        string assemblyDir = Path.GetDirectoryName(typeof(SdxlGenerationTests).Assembly.Location)!;
        string ptxDir = Path.Combine(assemblyDir, "Ptx");
        if (!Directory.Exists(ptxDir))
        {
            _output.WriteLine($"SKIPPED: PTX directory not found: {ptxDir}");
            return;
        }

        string refDir = Path.Combine(
            Path.GetDirectoryName(typeof(SdxlGenerationTests).Assembly.Location)!,
            "..", "..", "..", "..", "python-reference", "sdxl_reference_tensors");
        refDir = Path.GetFullPath(refDir);

        if (!Directory.Exists(refDir))
        {
            _output.WriteLine($"SKIPPED: Python reference tensors not found: {refDir}");
            return;
        }

        _output.WriteLine($"Loading Python reference tensors from: {refDir}");

        Tensor pyScaledInput = LoadBinaryTensor(Path.Combine(refDir, "step0_scaled_input.bin"), new TensorShape(1, 4, 32, 32));
        Tensor pyTextEmb = LoadBinaryTensor(Path.Combine(refDir, "text_embeddings.bin"), new TensorShape(2, 77, 2048));
        Tensor pyPooled = LoadBinaryTensor(Path.Combine(refDir, "clip_g_pooled.bin"), new TensorShape(2, 1280));
        Tensor pyNoiseUncond = LoadBinaryTensor(Path.Combine(refDir, "step0_noise_pred_uncond.bin"), new TensorShape(1, 4, 32, 32));
        Tensor pyNoiseCond = LoadBinaryTensor(Path.Combine(refDir, "step0_noise_pred_cond.bin"), new TensorShape(1, 4, 32, 32));

        DumpTensorStats(_output, "py_scaled_input", pyScaledInput);
        DumpTensorStats(_output, "py_text_emb", pyTextEmb);
        DumpTensorStats(_output, "py_pooled", pyPooled);

        (SdxlCheckpointConverter.ConvertedWeights converted, SafeTensorsLoader loader) =
            SdxlCheckpointConverter.LoadAndConvert(SdxlCheckpointPath);

        using (loader)
        {
            Dictionary<string, Tensor> unetF32 = CastWeightsToF32(converted.UNet);

            UNet unet = new(UNetConfig.SdxlBase);
            unet.LoadWeights(unetF32);

            using CudaBackend backend = new(deviceOrdinal: 0, ptxDir: ptxDir);

            float[] sizeCondition = [256f, 256f, 0f, 0f, 256f, 256f];

            int seqLen = (int)pyTextEmb.Shape[1];
            int hiddenSize = (int)pyTextEmb.Shape[2];
            int pooledDim = (int)pyPooled.Shape[1];

            Tensor uncondEmb = SliceBatchElement(pyTextEmb, 0, seqLen, hiddenSize);
            Tensor condEmb = SliceBatchElement(pyTextEmb, 1, seqLen, hiddenSize);
            Tensor uncondPooled = SliceBatchElement1D(pyPooled, 0, pooledDim);
            Tensor condPooled = SliceBatchElement1D(pyPooled, 1, pooledDim);

            // Run UNet forward - unconditional (GPU)
            _output.WriteLine("\nRunning UNet uncond pass (GPU)...");
            Stopwatch sw = Stopwatch.StartNew();
            Tensor csNoiseUncond = unet.Forward(backend, pyScaledInput, 900.0f, uncondEmb, uncondPooled, sizeCondition);
            sw.Stop();
            _output.WriteLine($"  Done in {sw.Elapsed.TotalSeconds:F1}s");
            DumpTensorStats(_output, "gpu_noise_uncond", csNoiseUncond);

            _output.WriteLine("\n=== GPU vs Python: uncond noise prediction ===");
            CompareWithReference(_output, "gpu_noise_pred_uncond", csNoiseUncond, pyNoiseUncond);

            // Run UNet forward - conditional (GPU)
            _output.WriteLine("\nRunning UNet cond pass (GPU)...");
            sw.Restart();
            Tensor csNoiseCond = unet.Forward(backend, pyScaledInput, 900.0f, condEmb, condPooled, sizeCondition);
            sw.Stop();
            _output.WriteLine($"  Done in {sw.Elapsed.TotalSeconds:F1}s");
            DumpTensorStats(_output, "gpu_noise_cond", csNoiseCond);

            _output.WriteLine("\n=== GPU vs Python: cond noise prediction ===");
            CompareWithReference(_output, "gpu_noise_pred_cond", csNoiseCond, pyNoiseCond);

            uncondEmb.Dispose();
            condEmb.Dispose();
            uncondPooled.Dispose();
            condPooled.Dispose();
            csNoiseUncond.Dispose();
            csNoiseCond.Dispose();
        }

        pyScaledInput.Dispose();
        pyTextEmb.Dispose();
        pyPooled.Dispose();
        pyNoiseUncond.Dispose();
        pyNoiseCond.Dispose();

        _output.WriteLine("\n=== Done ===");
    }

    [Fact]
    public void GpuVsCpu_UNetDirectComparison()
    {
        // Runs the same UNet forward pass on CPU and GPU backends, compares outputs directly.
        // This eliminates Python reference as a variable.
        if (!File.Exists(SdxlCheckpointPath))
        {
            _output.WriteLine($"SKIPPED: Checkpoint not found: {SdxlCheckpointPath}");
            return;
        }

        string assemblyDir = Path.GetDirectoryName(typeof(SdxlGenerationTests).Assembly.Location)!;
        string ptxDir = Path.Combine(assemblyDir, "Ptx");
        if (!Directory.Exists(ptxDir))
        {
            _output.WriteLine($"SKIPPED: PTX directory not found: {ptxDir}");
            return;
        }

        string refDir = Path.Combine(assemblyDir, "..", "..", "..", "..", "python-reference", "sdxl_reference_tensors");
        refDir = Path.GetFullPath(refDir);
        if (!Directory.Exists(refDir))
        {
            _output.WriteLine($"SKIPPED: Python reference tensors not found: {refDir}");
            return;
        }

        Tensor pyScaledInput = LoadBinaryTensor(Path.Combine(refDir, "step0_scaled_input.bin"), new TensorShape(1, 4, 32, 32));
        Tensor pyTextEmb = LoadBinaryTensor(Path.Combine(refDir, "text_embeddings.bin"), new TensorShape(2, 77, 2048));
        Tensor pyPooled = LoadBinaryTensor(Path.Combine(refDir, "clip_g_pooled.bin"), new TensorShape(2, 1280));

        (SdxlCheckpointConverter.ConvertedWeights converted, SafeTensorsLoader loader) =
            SdxlCheckpointConverter.LoadAndConvert(SdxlCheckpointPath);

        using (loader)
        {
            Dictionary<string, Tensor> unetF32 = CastWeightsToF32(converted.UNet);

            // Create TWO UNet instances with the same weights
            UNet cpuUnet = new(UNetConfig.SdxlBase);
            cpuUnet.LoadWeights(unetF32);
            UNet gpuUnet = new(UNetConfig.SdxlBase);
            gpuUnet.LoadWeights(unetF32);

            float[] sizeCondition = [256f, 256f, 0f, 0f, 256f, 256f];
            int seqLen = (int)pyTextEmb.Shape[1];
            int hiddenSize = (int)pyTextEmb.Shape[2];
            int pooledDim = (int)pyPooled.Shape[1];

            Tensor uncondEmb = SliceBatchElement(pyTextEmb, 0, seqLen, hiddenSize);
            Tensor condEmb = SliceBatchElement(pyTextEmb, 1, seqLen, hiddenSize);
            Tensor uncondPooled = SliceBatchElement1D(pyPooled, 0, pooledDim);
            Tensor condPooled = SliceBatchElement1D(pyPooled, 1, pooledDim);

            // Run CPU
            using CpuBackend cpuBackend = new();
            _output.WriteLine("Running UNet uncond pass (CPU)...");
            Tensor cpuResult = cpuUnet.Forward(cpuBackend, pyScaledInput, 900.0f, uncondEmb, uncondPooled, sizeCondition);
            DumpTensorStats(_output, "cpu_result", cpuResult);

            // Run GPU
            using CudaBackend gpuBackend = new(deviceOrdinal: 0, ptxDir: ptxDir);
            _output.WriteLine("Running UNet uncond pass (GPU)...");
            Tensor gpuResult = gpuUnet.Forward(gpuBackend, pyScaledInput, 900.0f, uncondEmb, uncondPooled, sizeCondition);
            DumpTensorStats(_output, "gpu_result", gpuResult);

            _output.WriteLine("\n=== GPU vs CPU: direct UNet comparison ===");
            CompareWithReference(_output, "gpu_vs_cpu", gpuResult, cpuResult);

            cpuResult.Dispose();
            gpuResult.Dispose();
            uncondEmb.Dispose();
            condEmb.Dispose();
            uncondPooled.Dispose();
            condPooled.Dispose();
        }

        pyScaledInput.Dispose();
        pyTextEmb.Dispose();
        pyPooled.Dispose();

        _output.WriteLine("\n=== Done ===");
    }

    [Fact]
    public unsafe void GpuGelu_DiagnosticValues()
    {
        string assemblyDir = Path.GetDirectoryName(typeof(SdxlGenerationTests).Assembly.Location)!;
        string ptxDir = Path.Combine(assemblyDir, "Ptx");
        if (!Directory.Exists(ptxDir))
        {
            _output.WriteLine($"SKIPPED: PTX directory not found: {ptxDir}");
            return;
        }
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: no CUDA driver available (running outside the host? try a non-Flatpak terminal)");
            return;
        }

        using CpuBackend cpu = new();
        using CudaBackend gpu = new(deviceOrdinal: 0, ptxDir: ptxDir);

        // Test 1: Small known values
        float[] testValues = [0.0f, 0.5f, -0.5f, 1.0f, -1.0f, 2.0f, -2.0f, 3.0f];
        TensorShape smallShape = new TensorShape(testValues.Length);
        Tensor cpuIn = new Tensor(smallShape, DType.F32);
        Tensor gpuIn = new Tensor(smallShape, DType.F32);
        Tensor cpuOut = new Tensor(smallShape, DType.F32);
        Tensor gpuOut = new Tensor(smallShape, DType.F32);

        for (int i = 0; i < testValues.Length; i++)
        {
            ((float*)cpuIn.DataPointer)[i] = testValues[i];
            ((float*)gpuIn.DataPointer)[i] = testValues[i];
        }

        cpu.Gelu(cpuOut, cpuIn);
        gpu.Gelu(gpuOut, gpuIn);

        _output.WriteLine("GELU diagnostic (small): input → CPU → GPU → diff");
        for (int i = 0; i < testValues.Length; i++)
        {
            float c = ((float*)cpuOut.DataPointer)[i];
            float g = ((float*)gpuOut.DataPointer)[i];
            _output.WriteLine($"  x={testValues[i],6:F1} → cpu={c,10:F6} gpu={g,10:F6} diff={g - c:E3}");
        }
        cpuIn.Dispose(); gpuIn.Dispose(); cpuOut.Dispose(); gpuOut.Dispose();

        // Test 2: Large tensor [2,77,1280] in isolation (same as per-op test)
        _output.WriteLine("\nGELU diagnostic (large [2,77,1280]):");
        TensorShape largeShape = new TensorShape(2, 77, 1280);
        Random rng = new(42);
        Tensor largeIn = new Tensor(largeShape, DType.F32);
        float* p = (float*)largeIn.DataPointer;
        for (long i = 0; i < largeIn.ElementCount; i++)
            p[i] = (float)(rng.NextDouble() * 2 - 1);

        Tensor largeCpuOut = new Tensor(largeShape, DType.F32);
        Tensor largeGpuOut = new Tensor(largeShape, DType.F32);
        Tensor largeGpuIn = new Tensor(largeShape, DType.F32);
        Buffer.MemoryCopy((void*)largeIn.DataPointer, (void*)largeGpuIn.DataPointer,
            largeIn.ElementCount * 4, largeIn.ElementCount * 4);

        cpu.Gelu(largeCpuOut, largeIn);
        gpu.Gelu(largeGpuOut, largeGpuIn);

        float* cPtr = (float*)largeCpuOut.DataPointer;
        float* gPtr = (float*)largeGpuOut.DataPointer;
        double sumAbsErr = 0, maxAbsErr = 0;
        for (long i = 0; i < largeIn.ElementCount; i++)
        {
            double err = Math.Abs(cPtr[i] - gPtr[i]);
            sumAbsErr += err;
            if (err > maxAbsErr) maxAbsErr = err;
        }
        double avgErr = sumAbsErr / largeIn.ElementCount;
        _output.WriteLine($"  avg_err={avgErr:E3}, max_err={maxAbsErr:E3}");
        _output.WriteLine($"  first_8 cpu: [{string.Join(", ", Enumerable.Range(0, 8).Select(i => cPtr[i].ToString("G6")))}]");
        _output.WriteLine($"  first_8 gpu: [{string.Join(", ", Enumerable.Range(0, 8).Select(i => gPtr[i].ToString("G6")))}]");

        largeIn.Dispose(); largeCpuOut.Dispose(); largeGpuOut.Dispose(); largeGpuIn.Dispose();
        gpu.EvictGpuCache();
    }

    [Fact]
    public unsafe void GpuVsCpu_PerOperationComparison()
    {
        string assemblyDir = Path.GetDirectoryName(typeof(SdxlGenerationTests).Assembly.Location)!;
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

        using CpuBackend cpu = new();
        using CudaBackend gpu = new(deviceOrdinal: 0, ptxDir: ptxDir);

        Random rng = new(42);

        // Helper: create random tensor
        Tensor MakeRandom(TensorShape shape)
        {
            Tensor t = new Tensor(shape, DType.F32);
            float* p = (float*)t.DataPointer;
            for (long i = 0; i < t.ElementCount; i++)
                p[i] = (float)(rng.NextDouble() * 2 - 1);
            return t;
        }

        // Helper: clone tensor
        Tensor Clone(Tensor src)
        {
            Tensor dst = new Tensor(src.Shape, DType.F32);
            Buffer.MemoryCopy((void*)src.DataPointer, (void*)dst.DataPointer,
                src.ElementCount * 4, src.ElementCount * 4);
            return dst;
        }

        // Helper: compare
        void Compare(string name, Tensor cpuOut, Tensor gpuOut)
        {
            float* cPtr = (float*)cpuOut.DataPointer;
            float* gPtr = (float*)gpuOut.DataPointer;
            long count = cpuOut.ElementCount;
            double sumAbsErr = 0, maxAbsErr = 0;
            for (long i = 0; i < count; i++)
            {
                double err = Math.Abs(cPtr[i] - gPtr[i]);
                sumAbsErr += err;
                if (err > maxAbsErr) maxAbsErr = err;
            }
            double avgErr = sumAbsErr / count;
            string status = avgErr < 1e-4 ? "OK" : "FAIL";
            _output.WriteLine($"  {name}: avg_err={avgErr:E3}, max_err={maxAbsErr:E3} [{status}]");
        }

        _output.WriteLine("=== Per-Operation GPU vs CPU Comparison ===\n");

        // 1. Linear
        {
            TensorShape inShape = new TensorShape(2, 77, 320);
            TensorShape wShape = new TensorShape(640, 320);
            TensorShape bShape = new TensorShape(640);
            TensorShape outShape = new TensorShape(2, 77, 640);
            Tensor input = MakeRandom(inShape);
            Tensor weight = MakeRandom(wShape);
            Tensor bias = MakeRandom(bShape);
            Tensor cpuOut = new Tensor(outShape, DType.F32);
            Tensor gpuOut = new Tensor(outShape, DType.F32);
            Tensor gpuIn = Clone(input);
            cpu.Linear(cpuOut, input, weight, bias);
            gpu.Linear(gpuOut, gpuIn, weight, bias);
            Compare("Linear [2,77,320]→[2,77,640]", cpuOut, gpuOut);
            input.Dispose(); weight.Dispose(); bias.Dispose();
            cpuOut.Dispose(); gpuOut.Dispose(); gpuIn.Dispose();
        }

        // 2. GroupNorm
        {
            TensorShape inShape = new TensorShape(1, 320, 32, 32);
            TensorShape wShape = new TensorShape(320);
            Tensor input = MakeRandom(inShape);
            Tensor weight = MakeRandom(wShape);
            Tensor bias = MakeRandom(wShape);
            Tensor cpuOut = new Tensor(inShape, DType.F32);
            Tensor gpuOut = new Tensor(inShape, DType.F32);
            Tensor gpuIn = Clone(input);
            cpu.GroupNorm(cpuOut, input, weight, bias, 32, 1e-5f);
            gpu.GroupNorm(gpuOut, gpuIn, weight, bias, 32, 1e-5f);
            Compare("GroupNorm [1,320,32,32] g=32", cpuOut, gpuOut);
            input.Dispose(); weight.Dispose(); bias.Dispose();
            cpuOut.Dispose(); gpuOut.Dispose(); gpuIn.Dispose();
        }

        // 3. LayerNorm
        {
            TensorShape inShape = new TensorShape(2, 77, 320);
            TensorShape wShape = new TensorShape(320);
            Tensor input = MakeRandom(inShape);
            Tensor weight = MakeRandom(wShape);
            Tensor bias = MakeRandom(wShape);
            Tensor cpuOut = new Tensor(inShape, DType.F32);
            Tensor gpuOut = new Tensor(inShape, DType.F32);
            Tensor gpuIn = Clone(input);
            cpu.LayerNorm(cpuOut, input, weight, bias, 1e-5f);
            gpu.LayerNorm(gpuOut, gpuIn, weight, bias, 1e-5f);
            Compare("LayerNorm [2,77,320]", cpuOut, gpuOut);
            input.Dispose(); weight.Dispose(); bias.Dispose();
            cpuOut.Dispose(); gpuOut.Dispose(); gpuIn.Dispose();
        }

        // 4. Conv2D (3x3, pad=1)
        {
            TensorShape inShape = new TensorShape(1, 320, 32, 32);
            TensorShape wShape = new TensorShape(320, 320, 3, 3);
            TensorShape bShape = new TensorShape(320);
            TensorShape outShape = new TensorShape(1, 320, 32, 32);
            Tensor input = MakeRandom(inShape);
            Tensor weight = MakeRandom(wShape);
            Tensor bias = MakeRandom(bShape);
            Tensor cpuOut = new Tensor(outShape, DType.F32);
            Tensor gpuOut = new Tensor(outShape, DType.F32);
            Tensor gpuIn = Clone(input);
            cpu.Conv2D(cpuOut, input, weight, bias, 1, 1, 1, 1);
            gpu.Conv2D(gpuOut, gpuIn, weight, bias, 1, 1, 1, 1);
            Compare("Conv2D [1,320,32,32] 3x3 pad=1", cpuOut, gpuOut);
            input.Dispose(); weight.Dispose(); bias.Dispose();
            cpuOut.Dispose(); gpuOut.Dispose(); gpuIn.Dispose();
        }

        // 5. SiLU
        {
            TensorShape shape = new TensorShape(1, 320, 32, 32);
            Tensor input = MakeRandom(shape);
            Tensor cpuOut = new Tensor(shape, DType.F32);
            Tensor gpuOut = new Tensor(shape, DType.F32);
            Tensor gpuIn = Clone(input);
            cpu.Silu(cpuOut, input);
            gpu.Silu(gpuOut, gpuIn);
            Compare("SiLU [1,320,32,32]", cpuOut, gpuOut);
            input.Dispose(); cpuOut.Dispose(); gpuOut.Dispose(); gpuIn.Dispose();
        }

        // 6. GELU
        {
            TensorShape shape = new TensorShape(2, 77, 1280);
            Tensor input = MakeRandom(shape);
            Tensor cpuOut = new Tensor(shape, DType.F32);
            Tensor gpuOut = new Tensor(shape, DType.F32);
            Tensor gpuIn = Clone(input);
            cpu.Gelu(cpuOut, input);
            gpu.Gelu(gpuOut, gpuIn);
            Compare("GELU [2,77,1280]", cpuOut, gpuOut);
            input.Dispose(); cpuOut.Dispose(); gpuOut.Dispose(); gpuIn.Dispose();
        }

        // 7. Add
        {
            TensorShape shape = new TensorShape(1, 320, 32, 32);
            Tensor a = MakeRandom(shape);
            Tensor b = MakeRandom(shape);
            Tensor cpuOut = new Tensor(shape, DType.F32);
            Tensor gpuOut = new Tensor(shape, DType.F32);
            Tensor gpuA = Clone(a);
            Tensor gpuB = Clone(b);
            cpu.Add(cpuOut, a, b);
            gpu.Add(gpuOut, gpuA, gpuB);
            Compare("Add [1,320,32,32]", cpuOut, gpuOut);
            a.Dispose(); b.Dispose(); cpuOut.Dispose(); gpuOut.Dispose();
            gpuA.Dispose(); gpuB.Dispose();
        }

        // 8. Scale
        {
            TensorShape shape = new TensorShape(1, 4, 32, 32);
            Tensor input = MakeRandom(shape);
            Tensor cpuOut = new Tensor(shape, DType.F32);
            Tensor gpuOut = new Tensor(shape, DType.F32);
            Tensor gpuIn = Clone(input);
            cpu.Scale(cpuOut, input, 14.6146f);
            gpu.Scale(gpuOut, gpuIn, 14.6146f);
            Compare("Scale [1,4,32,32] * 14.6146", cpuOut, gpuOut);
            input.Dispose(); cpuOut.Dispose(); gpuOut.Dispose(); gpuIn.Dispose();
        }

        // 9. SDPA (multi-head attention)
        {
            int B = 1, H = 10, Sq = 77, D = 64;
            TensorShape qShape = new TensorShape(B, H, Sq, D);
            TensorShape kvShape = new TensorShape(B, H, Sq, D);
            Tensor q = MakeRandom(qShape);
            Tensor k = MakeRandom(kvShape);
            Tensor v = MakeRandom(kvShape);
            Tensor cpuOut = new Tensor(qShape, DType.F32);
            Tensor gpuOut = new Tensor(qShape, DType.F32);
            Tensor gpuQ = Clone(q);
            Tensor gpuK = Clone(k);
            Tensor gpuV = Clone(v);
            float scale = 1.0f / MathF.Sqrt(D);
            cpu.ScaledDotProductAttention(cpuOut, q, k, v, null, scale);
            gpu.ScaledDotProductAttention(gpuOut, gpuQ, gpuK, gpuV, null, scale);
            Compare("SDPA [1,10,77,64]", cpuOut, gpuOut);
            q.Dispose(); k.Dispose(); v.Dispose();
            cpuOut.Dispose(); gpuOut.Dispose();
            gpuQ.Dispose(); gpuK.Dispose(); gpuV.Dispose();
        }

        // 10. BatchedMatMul
        {
            TensorShape aShape = new TensorShape(2, 77, 320);
            TensorShape bShape = new TensorShape(320, 640);
            TensorShape outShape = new TensorShape(2, 77, 640);
            Tensor a = MakeRandom(aShape);
            Tensor b = MakeRandom(bShape);
            Tensor cpuOut = new Tensor(outShape, DType.F32);
            Tensor gpuOut = new Tensor(outShape, DType.F32);
            Tensor gpuA = Clone(a);
            cpu.BatchedMatMul(cpuOut, a, b);
            gpu.BatchedMatMul(gpuOut, gpuA, b);
            Compare("BatchedMatMul [2,77,320]x[320,640]", cpuOut, gpuOut);
            a.Dispose(); b.Dispose(); cpuOut.Dispose(); gpuOut.Dispose(); gpuA.Dispose();
        }

        // 11. Concat dim=1
        {
            TensorShape s1 = new TensorShape(1, 320, 32, 32);
            TensorShape s2 = new TensorShape(1, 320, 32, 32);
            TensorShape outShape = new TensorShape(1, 640, 32, 32);
            Tensor t1 = MakeRandom(s1);
            Tensor t2 = MakeRandom(s2);
            Tensor cpuOut = new Tensor(outShape, DType.F32);
            Tensor gpuOut = new Tensor(outShape, DType.F32);
            Tensor gt1 = Clone(t1);
            Tensor gt2 = Clone(t2);
            cpu.Concat(cpuOut, [t1, t2], 1);
            gpu.Concat(gpuOut, [gt1, gt2], 1);
            Compare("Concat dim=1 [1,320+320,32,32]", cpuOut, gpuOut);
            t1.Dispose(); t2.Dispose(); cpuOut.Dispose(); gpuOut.Dispose();
            gt1.Dispose(); gt2.Dispose();
        }

        // 12. UpsampleNearest2D
        {
            TensorShape inShape = new TensorShape(1, 320, 16, 16);
            TensorShape outShape = new TensorShape(1, 320, 32, 32);
            Tensor input = MakeRandom(inShape);
            Tensor cpuOut = new Tensor(outShape, DType.F32);
            Tensor gpuOut = new Tensor(outShape, DType.F32);
            Tensor gpuIn = Clone(input);
            cpu.UpsampleNearest2D(cpuOut, input, 2, 2);
            gpu.UpsampleNearest2D(gpuOut, gpuIn, 2, 2);
            Compare("UpsampleNearest2D 16→32", cpuOut, gpuOut);
            input.Dispose(); cpuOut.Dispose(); gpuOut.Dispose(); gpuIn.Dispose();
        }

        gpu.EvictGpuCache();
        _output.WriteLine("\n=== Done ===");
    }

    [Fact]
    public unsafe void GpuVsCpu_SdxlRealisticShapes()
    {
        string assemblyDir = Path.GetDirectoryName(typeof(SdxlGenerationTests).Assembly.Location)!;
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

        using CpuBackend cpu = new();
        using CudaBackend gpu = new(deviceOrdinal: 0, ptxDir: ptxDir);

        Random rng = new(42);

        Tensor MakeRandom(TensorShape shape)
        {
            Tensor t = new Tensor(shape, DType.F32);
            float* p = (float*)t.DataPointer;
            for (long i = 0; i < t.ElementCount; i++)
                p[i] = (float)(rng.NextDouble() * 2 - 1);
            return t;
        }

        Tensor Clone(Tensor src)
        {
            Tensor dst = new Tensor(src.Shape, DType.F32);
            Buffer.MemoryCopy((void*)src.DataPointer, (void*)dst.DataPointer,
                src.ElementCount * 4, src.ElementCount * 4);
            return dst;
        }

        void Compare(string name, Tensor cpuOut, Tensor gpuOut)
        {
            float* cPtr = (float*)cpuOut.DataPointer;
            float* gPtr = (float*)gpuOut.DataPointer;
            long count = cpuOut.ElementCount;
            double sumAbsErr = 0, maxAbsErr = 0;
            for (long i = 0; i < count; i++)
            {
                double err = Math.Abs(cPtr[i] - gPtr[i]);
                sumAbsErr += err;
                if (err > maxAbsErr) maxAbsErr = err;
            }
            double avgErr = sumAbsErr / count;
            string status = avgErr < 1e-4 ? "OK" : "FAIL";
            _output.WriteLine($"  {name}: avg_err={avgErr:E3}, max_err={maxAbsErr:E3} [{status}]");
        }

        _output.WriteLine("=== SDXL Realistic Shapes: GPU vs CPU ===\n");

        // 1. Linear - attention Q/K/V projection (large inner dim)
        {
            TensorShape inShape = new TensorShape(1, 1024, 640);
            TensorShape wShape = new TensorShape(1920, 640);
            TensorShape bShape = new TensorShape(1920);
            TensorShape outShape = new TensorShape(1, 1024, 1920);
            Tensor input = MakeRandom(inShape); Tensor weight = MakeRandom(wShape);
            Tensor bias = MakeRandom(bShape);
            Tensor cpuOut = new Tensor(outShape, DType.F32);
            Tensor gpuOut = new Tensor(outShape, DType.F32);
            Tensor gpuIn = Clone(input);
            cpu.Linear(cpuOut, input, weight, bias);
            gpu.Linear(gpuOut, gpuIn, weight, bias);
            Compare("Linear [1,1024,640]→[1,1024,1920]", cpuOut, gpuOut);
            input.Dispose(); weight.Dispose(); bias.Dispose();
            cpuOut.Dispose(); gpuOut.Dispose(); gpuIn.Dispose();
        }

        // 2. Linear - deep block (1280→5120 for GEGLU)
        {
            TensorShape inShape = new TensorShape(1, 64, 1280);
            TensorShape wShape = new TensorShape(5120, 1280);
            TensorShape bShape = new TensorShape(5120);
            TensorShape outShape = new TensorShape(1, 64, 5120);
            Tensor input = MakeRandom(inShape); Tensor weight = MakeRandom(wShape);
            Tensor bias = MakeRandom(bShape);
            Tensor cpuOut = new Tensor(outShape, DType.F32);
            Tensor gpuOut = new Tensor(outShape, DType.F32);
            Tensor gpuIn = Clone(input);
            cpu.Linear(cpuOut, input, weight, bias);
            gpu.Linear(gpuOut, gpuIn, weight, bias);
            Compare("Linear [1,64,1280]→[1,64,5120]", cpuOut, gpuOut);
            input.Dispose(); weight.Dispose(); bias.Dispose();
            cpuOut.Dispose(); gpuOut.Dispose(); gpuIn.Dispose();
        }

        // 3. Conv2D - deep block (1280ch, 8x8 spatial)
        {
            TensorShape inShape = new TensorShape(1, 1280, 8, 8);
            TensorShape wShape = new TensorShape(1280, 1280, 3, 3);
            TensorShape bShape = new TensorShape(1280);
            TensorShape outShape = new TensorShape(1, 1280, 8, 8);
            Tensor input = MakeRandom(inShape); Tensor weight = MakeRandom(wShape);
            Tensor bias = MakeRandom(bShape);
            Tensor cpuOut = new Tensor(outShape, DType.F32);
            Tensor gpuOut = new Tensor(outShape, DType.F32);
            Tensor gpuIn = Clone(input);
            cpu.Conv2D(cpuOut, input, weight, bias, 1, 1, 1, 1);
            gpu.Conv2D(gpuOut, gpuIn, weight, bias, 1, 1, 1, 1);
            Compare("Conv2D [1,1280,8,8] 3x3", cpuOut, gpuOut);
            input.Dispose(); weight.Dispose(); bias.Dispose();
            cpuOut.Dispose(); gpuOut.Dispose(); gpuIn.Dispose();
        }

        // 4. Conv2D - 1x1 (used for skip connections)
        {
            TensorShape inShape = new TensorShape(1, 640, 16, 16);
            TensorShape wShape = new TensorShape(1280, 640, 1, 1);
            TensorShape bShape = new TensorShape(1280);
            TensorShape outShape = new TensorShape(1, 1280, 16, 16);
            Tensor input = MakeRandom(inShape); Tensor weight = MakeRandom(wShape);
            Tensor bias = MakeRandom(bShape);
            Tensor cpuOut = new Tensor(outShape, DType.F32);
            Tensor gpuOut = new Tensor(outShape, DType.F32);
            Tensor gpuIn = Clone(input);
            cpu.Conv2D(cpuOut, input, weight, bias, 1, 1, 0, 0);
            gpu.Conv2D(gpuOut, gpuIn, weight, bias, 1, 1, 0, 0);
            Compare("Conv2D [1,640,16,16]→[1,1280,16,16] 1x1", cpuOut, gpuOut);
            input.Dispose(); weight.Dispose(); bias.Dispose();
            cpuOut.Dispose(); gpuOut.Dispose(); gpuIn.Dispose();
        }

        // 5. GroupNorm - deep block
        {
            TensorShape inShape = new TensorShape(1, 1280, 8, 8);
            TensorShape wShape = new TensorShape(1280);
            Tensor input = MakeRandom(inShape); Tensor weight = MakeRandom(wShape);
            Tensor bias = MakeRandom(wShape);
            Tensor cpuOut = new Tensor(inShape, DType.F32);
            Tensor gpuOut = new Tensor(inShape, DType.F32);
            Tensor gpuIn = Clone(input);
            cpu.GroupNorm(cpuOut, input, weight, bias, 32, 1e-5f);
            gpu.GroupNorm(gpuOut, gpuIn, weight, bias, 32, 1e-5f);
            Compare("GroupNorm [1,1280,8,8] g=32", cpuOut, gpuOut);
            input.Dispose(); weight.Dispose(); bias.Dispose();
            cpuOut.Dispose(); gpuOut.Dispose(); gpuIn.Dispose();
        }

        // 6. SDPA - spatial self-attention (1024 tokens = 32*32)
        {
            int B = 1, H = 10, Sq = 1024, D = 64;
            TensorShape qShape = new TensorShape(B, H, Sq, D);
            Tensor q = MakeRandom(qShape); Tensor k = MakeRandom(qShape); Tensor v = MakeRandom(qShape);
            Tensor cpuOut = new Tensor(qShape, DType.F32);
            Tensor gpuOut = new Tensor(qShape, DType.F32);
            Tensor gpuQ = Clone(q); Tensor gpuK = Clone(k); Tensor gpuV = Clone(v);
            float scale = 1.0f / MathF.Sqrt(D);
            cpu.ScaledDotProductAttention(cpuOut, q, k, v, null, scale);
            gpu.ScaledDotProductAttention(gpuOut, gpuQ, gpuK, gpuV, null, scale);
            Compare("SDPA [1,10,1024,64] (spatial)", cpuOut, gpuOut);
            q.Dispose(); k.Dispose(); v.Dispose();
            cpuOut.Dispose(); gpuOut.Dispose();
            gpuQ.Dispose(); gpuK.Dispose(); gpuV.Dispose();
        }

        // 7. SDPA - cross attention (1024 Q tokens, 77 KV tokens)
        {
            int B = 1, H = 10, Sq = 1024, Skv = 77, D = 64;
            TensorShape qShape = new TensorShape(B, H, Sq, D);
            TensorShape kvShape = new TensorShape(B, H, Skv, D);
            TensorShape outShape = new TensorShape(B, H, Sq, D);
            Tensor q = MakeRandom(qShape); Tensor k = MakeRandom(kvShape); Tensor v = MakeRandom(kvShape);
            Tensor cpuOut = new Tensor(outShape, DType.F32);
            Tensor gpuOut = new Tensor(outShape, DType.F32);
            Tensor gpuQ = Clone(q); Tensor gpuK = Clone(k); Tensor gpuV = Clone(v);
            float scale = 1.0f / MathF.Sqrt(D);
            cpu.ScaledDotProductAttention(cpuOut, q, k, v, null, scale);
            gpu.ScaledDotProductAttention(gpuOut, gpuQ, gpuK, gpuV, null, scale);
            Compare("SDPA [1,10,1024,64]×[1,10,77,64] (cross)", cpuOut, gpuOut);
            q.Dispose(); k.Dispose(); v.Dispose();
            cpuOut.Dispose(); gpuOut.Dispose();
            gpuQ.Dispose(); gpuK.Dispose(); gpuV.Dispose();
        }

        // 8. SDPA - deep block spatial (20 heads)
        {
            int B = 1, H = 20, Sq = 64, D = 64;
            TensorShape qShape = new TensorShape(B, H, Sq, D);
            Tensor q = MakeRandom(qShape); Tensor k = MakeRandom(qShape); Tensor v = MakeRandom(qShape);
            Tensor cpuOut = new Tensor(qShape, DType.F32);
            Tensor gpuOut = new Tensor(qShape, DType.F32);
            Tensor gpuQ = Clone(q); Tensor gpuK = Clone(k); Tensor gpuV = Clone(v);
            float scale = 1.0f / MathF.Sqrt(D);
            cpu.ScaledDotProductAttention(cpuOut, q, k, v, null, scale);
            gpu.ScaledDotProductAttention(gpuOut, gpuQ, gpuK, gpuV, null, scale);
            Compare("SDPA [1,20,64,64] (deep spatial)", cpuOut, gpuOut);
            q.Dispose(); k.Dispose(); v.Dispose();
            cpuOut.Dispose(); gpuOut.Dispose();
            gpuQ.Dispose(); gpuK.Dispose(); gpuV.Dispose();
        }

        // 9. Conv2D - downsample stride=2
        {
            TensorShape inShape = new TensorShape(1, 320, 32, 32);
            TensorShape wShape = new TensorShape(320, 320, 3, 3);
            TensorShape bShape = new TensorShape(320);
            TensorShape outShape = new TensorShape(1, 320, 16, 16);
            Tensor input = MakeRandom(inShape); Tensor weight = MakeRandom(wShape);
            Tensor bias = MakeRandom(bShape);
            Tensor cpuOut = new Tensor(outShape, DType.F32);
            Tensor gpuOut = new Tensor(outShape, DType.F32);
            Tensor gpuIn = Clone(input);
            cpu.Conv2D(cpuOut, input, weight, bias, 2, 2, 1, 1);
            gpu.Conv2D(gpuOut, gpuIn, weight, bias, 2, 2, 1, 1);
            Compare("Conv2D downsample [1,320,32,32]→[1,320,16,16] s=2", cpuOut, gpuOut);
            input.Dispose(); weight.Dispose(); bias.Dispose();
            cpuOut.Dispose(); gpuOut.Dispose(); gpuIn.Dispose();
        }

        // 10. Chain: GroupNorm → SiLU → Conv2D (first ResNet block pattern)
        {
            TensorShape shape = new TensorShape(1, 320, 32, 32);
            TensorShape wShape = new TensorShape(320);
            TensorShape cWShape = new TensorShape(320, 320, 3, 3);
            TensorShape cBShape = new TensorShape(320);

            Tensor input = MakeRandom(shape);
            Tensor gnW = MakeRandom(wShape); Tensor gnB = MakeRandom(wShape);
            Tensor convW = MakeRandom(cWShape); Tensor convB = MakeRandom(cBShape);

            // CPU chain
            Tensor cpuGn = new Tensor(shape, DType.F32);
            cpu.GroupNorm(cpuGn, input, gnW, gnB, 32, 1e-5f);
            Tensor cpuSilu = new Tensor(shape, DType.F32);
            cpu.Silu(cpuSilu, cpuGn);
            Tensor cpuConv = new Tensor(shape, DType.F32);
            cpu.Conv2D(cpuConv, cpuSilu, convW, convB, 1, 1, 1, 1);

            // GPU chain
            Tensor gpuInput = Clone(input);
            Tensor gpuGn = new Tensor(shape, DType.F32);
            gpu.GroupNorm(gpuGn, gpuInput, gnW, gnB, 32, 1e-5f);
            Tensor gpuSilu = new Tensor(shape, DType.F32);
            gpu.Silu(gpuSilu, gpuGn);
            Tensor gpuConv = new Tensor(shape, DType.F32);
            gpu.Conv2D(gpuConv, gpuSilu, convW, convB, 1, 1, 1, 1);

            Compare("Chain: GN→SiLU→Conv2D [1,320,32,32]", cpuConv, gpuConv);

            input.Dispose(); gnW.Dispose(); gnB.Dispose(); convW.Dispose(); convB.Dispose();
            cpuGn.Dispose(); cpuSilu.Dispose(); cpuConv.Dispose();
            gpuInput.Dispose(); gpuGn.Dispose(); gpuSilu.Dispose(); gpuConv.Dispose();
        }

        _output.WriteLine("\n=== Done ===");
    }

    private static unsafe Tensor LoadBinaryTensor(string path, TensorShape shape)
    {
        byte[] bytes = File.ReadAllBytes(path);
        Tensor tensor = new Tensor(shape, DType.F32);
        fixed (byte* srcPtr = bytes)
        {
            Buffer.MemoryCopy(srcPtr, (void*)tensor.DataPointer, bytes.Length, bytes.Length);
        }
        return tensor;
    }

    private static unsafe void CompareWithReference(ITestOutputHelper output, string name, Tensor actual, Tensor expected)
    {
        float* aPtr = (float*)actual.DataPointer;
        float* ePtr = (float*)expected.DataPointer;
        long count = actual.ElementCount;

        double sumErr = 0, sumAbsErr = 0, maxAbsErr = 0;
        int maxErrIdx = 0;
        for (long i = 0; i < count; i++)
        {
            double err = aPtr[i] - ePtr[i];
            double absErr = Math.Abs(err);
            sumErr += err;
            sumAbsErr += absErr;
            if (absErr > maxAbsErr)
            {
                maxAbsErr = absErr;
                maxErrIdx = (int)i;
            }
        }

        double meanErr = sumErr / count;
        double avgAbsErr = sumAbsErr / count;

        output.WriteLine($"  {name}: avg_err={avgAbsErr:E3}, max_err={maxAbsErr:E3} (at idx={maxErrIdx})");
        output.WriteLine($"    mean_bias={meanErr:E3}");
        output.WriteLine($"    actual[{maxErrIdx}]={aPtr[maxErrIdx]:G6}, expected[{maxErrIdx}]={ePtr[maxErrIdx]:G6}");

        // Sample first 8 elements
        output.WriteLine($"    actual  first_8: [{string.Join(", ", Enumerable.Range(0, 8).Select(i => aPtr[i].ToString("G6")))}]");
        output.WriteLine($"    expected first_8: [{string.Join(", ", Enumerable.Range(0, 8).Select(i => ePtr[i].ToString("G6")))}]");
    }

    private static Dictionary<string, Tensor> CastWeightsToF32(Dictionary<string, Tensor> weights)
    {
        Dictionary<string, Tensor> f32 = new(weights.Count);
        foreach (KeyValuePair<string, Tensor> kvp in weights)
        {
            f32[kvp.Key] = (kvp.Value.DType == DType.F16 || kvp.Value.DType == DType.BF16)
                ? kvp.Value.CastTo(DType.F32)
                : kvp.Value;
        }
        return f32;
    }

    private static Dictionary<string, Tensor> CastWeightsToF16(Dictionary<string, Tensor> weights)
    {
        Dictionary<string, Tensor> f16 = new(weights.Count);
        foreach (KeyValuePair<string, Tensor> kvp in weights)
        {
            f16[kvp.Key] = (kvp.Value.DType != DType.F16)
                ? kvp.Value.CastTo(DType.F16)
                : kvp.Value;
        }
        return f16;
    }
}
