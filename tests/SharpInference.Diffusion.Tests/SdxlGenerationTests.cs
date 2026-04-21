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
/// End-to-end SDXL image generation tests using single-file checkpoints.
/// Loads JuggernautXL, runs the full pipeline (dual CLIP → UNet denoise → VAE decode),
/// and saves the output as a BMP file.
///
/// WARNING: These tests are SLOW on CPU. A 256x256 image with 5 steps takes ~60+ minutes.
/// Set SDXL_SINGLE_FILE_PATH and CLIP_TOKENIZER_DIR environment variables or use defaults.
/// </summary>
public class SdxlGenerationTests
{
    private static readonly string SdxlCheckpointPath =
        Environment.GetEnvironmentVariable("SDXL_SINGLE_FILE_PATH")
        ?? @"C:\Users\AI Overlord\Desktop\Projects\SwarmUI\Models\Stable-Diffusion\juggernautXL_ragnarokBy.safetensors";

    // CLIP-L and CLIP-G use the same OpenAI CLIP vocabulary (49408 tokens)
    private static readonly string TokenizerVocabPath =
        Environment.GetEnvironmentVariable("CLIP_VOCAB_PATH")
        ?? @"C:\Users\AI Overlord\Desktop\Projects\SharpInference\tests\test-models\sd15\tokenizer\vocab.json";

    private static readonly string TokenizerMergesPath =
        Environment.GetEnvironmentVariable("CLIP_MERGES_PATH")
        ?? @"C:\Users\AI Overlord\Desktop\Projects\SharpInference\tests\test-models\sd15\tokenizer\merges.txt";

    private static readonly string OutputDir =
        Environment.GetEnvironmentVariable("SDXL_OUTPUT_DIR")
        ?? @"C:\Users\AI Overlord\Desktop\Projects\SharpInference\Output";

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
            _output.WriteLine($"\nGeneration complete in {genSw.Elapsed.TotalSeconds:F1} seconds (seed={seed})");

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
}
