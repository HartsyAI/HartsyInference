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
using SharpInference.Tests.Common;
using SharpInference.Tokenizers;

namespace SharpInference.Diffusion.Tests;

/// <summary>
/// End-to-end SD3 image generation tests using single-file checkpoints.
/// Loads an SD3 Medium checkpoint, runs the full pipeline (triple CLIP/T5 encoding → MMDiT denoise → VAE decode),
/// and saves the output as a BMP file.
///
/// WARNING: These tests are SLOW on CPU. Use GPU backend for reasonable run times.
/// Set SD3_SINGLE_FILE_PATH, CLIP_VOCAB_PATH, CLIP_MERGES_PATH, and T5_SPIECE_MODEL_PATH
/// environment variables or use defaults.
/// </summary>
public class Sd3GenerationTests
{
    private static string Sd3CheckpointPath => TestPaths.Sd3.SingleFile;
    private static string TokenizerVocabPath => TestPaths.Tokenizers.ClipVocab;
    private static string TokenizerMergesPath => TestPaths.Tokenizers.ClipMerges;
    private static string T5SpieceModelPath => TestPaths.Tokenizers.T5Spiece;
    private static string OutputDir => TestPaths.OutputDir;

    private readonly ITestOutputHelper _output;

    public Sd3GenerationTests(ITestOutputHelper output) => _output = output;

    /// <summary>
    /// Full end-to-end SD3 image generation from a single-file checkpoint on CPU.
    /// Small resolution (128x128) and minimal steps (3) to keep CPU time manageable.
    /// Uses only CLIP-L + CLIP-G (no T5) for reduced memory.
    /// </summary>
    [Fact]
    public void SingleFile_GenerateImage_Small_NoT5()
    {
        if (!File.Exists(Sd3CheckpointPath))
        {
            _output.WriteLine($"SKIPPED: Checkpoint not found: {Sd3CheckpointPath}");
            return;
        }
        if (!File.Exists(TokenizerVocabPath) || !File.Exists(TokenizerMergesPath))
        {
            _output.WriteLine("SKIPPED: CLIP tokenizer files not found");
            return;
        }

        Stopwatch totalSw = Stopwatch.StartNew();

        // 1. Load and convert checkpoint
        _output.WriteLine($"[1/6] Loading checkpoint: {Path.GetFileName(Sd3CheckpointPath)}");
        Stopwatch sw = Stopwatch.StartNew();
        (Sd3CheckpointConverter.ConvertedWeights converted, SafeTensorsLoader loader) =
            Sd3CheckpointConverter.LoadAndConvert(Sd3CheckpointPath);
        sw.Stop();
        _output.WriteLine($"  Checkpoint loaded and converted in {sw.ElapsedMilliseconds}ms");

        using (loader)
        {
            Dictionary<string, Tensor> transformerF32 = CastWeightsToF32(converted.Transformer);
            Dictionary<string, Tensor> clipLF32 = CastWeightsToF32(converted.ClipL);
            Dictionary<string, Tensor> clipGF32 = CastWeightsToF32(converted.ClipG);
            Dictionary<string, Tensor> vaeF32 = CastWeightsToF32(converted.Vae);

            // 2. Tokenize (CLIP only, no T5)
            _output.WriteLine("[2/6] Tokenizing prompt...");
            using ClipTokenizer tokenizer = new(TokenizerVocabPath, TokenizerMergesPath);

            string prompt = "A photograph of an astronaut riding a horse";
            string negPrompt = "";

            int[] promptTokensL = tokenizer.Encode(prompt);
            int[] negTokensL = tokenizer.Encode(negPrompt);
            int[] promptTokensG = tokenizer.Encode(prompt);
            int[] negTokensG = tokenizer.Encode(negPrompt);

            int promptEosL = ClipTokenizer.FindEosPosition(promptTokensL);
            int negEosL = ClipTokenizer.FindEosPosition(negTokensL);
            int promptEosG = ClipTokenizer.FindEosPosition(promptTokensG);
            int negEosG = ClipTokenizer.FindEosPosition(negTokensG);
            _output.WriteLine($"  Prompt token count: {promptTokensL.Length}, EOS-L: {promptEosL}, EOS-G: {promptEosG}");

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

            // 5. Load SD3 Transformer
            _output.WriteLine("[5/6] Loading SD3 MMDiT transformer...");
            sw.Restart();
            int depth = Sd3CheckpointConverter.DetectDepth(transformerF32);
            _output.WriteLine($"  Detected depth: {depth}");
            Sd3Config config = Sd3Config.Medium;
            Sd3Transformer transformer = new Sd3Transformer(config);
            transformer.LoadWeights(transformerF32);
            sw.Stop();
            _output.WriteLine($"  Transformer loaded in {sw.ElapsedMilliseconds}ms");

            // 6. Load VAE
            _output.WriteLine("[6/6] Loading VAE...");
            sw.Restart();
            VaeDecoder vae = new(VaeConfig.Sd3);
            vae.LoadWeights(vaeF32);
            sw.Stop();
            _output.WriteLine($"  VAE loaded in {sw.ElapsedMilliseconds}ms");

            // Create pipeline and generate (no T5)
            using CpuBackend backend = new();
            using Sd3Pipeline pipeline = new(backend, clipL, clipG, null, transformer, vae, 3.0f);

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

            _output.WriteLine($"\nGenerating {request.Width}x{request.Height} image, {request.Steps} steps, cfg={request.CfgScale}, seed=42 (no T5)...");
            Stopwatch genSw = Stopwatch.StartNew();

            (byte[] rgbData, int width, int height, int seed) = pipeline.GenerateFromTokens(
                promptTokensL, negTokensL,
                promptTokensG, negTokensG,
                promptEosL, negEosL,
                promptEosG, negEosG,
                null, null, null, null,
                request,
                progress => _output.WriteLine($"  Step {progress.Step}/{progress.TotalSteps} ({progress.ElapsedMs:F0}ms)"));

            genSw.Stop();
            _output.WriteLine($"\nGeneration complete in {genSw.Elapsed.TotalMinutes:F1} minutes (seed={seed})");

            // Validate output
            Assert.Equal(128, width);
            Assert.Equal(128, height);
            Assert.Equal(128 * 128 * 3, rgbData.Length);

            // Check not all black or all white
            ValidateImageNotDegenerate(rgbData);

            // Save output
            Directory.CreateDirectory(OutputDir);
            string outputPath = Path.Combine(OutputDir, $"sd3_cpu_128_noT5_{DateTime.Now:yyyyMMdd_HHmmss}.bmp");
            ImagePostProcessor.SaveBmp(outputPath, rgbData, width, height);
            _output.WriteLine($"  Image saved to: {outputPath}");

            totalSw.Stop();
            _output.WriteLine($"\nTotal time: {totalSw.Elapsed.TotalMinutes:F1} minutes");

            transformer.Dispose();
        }
    }

    /// <summary>
    /// Full end-to-end SD3 image generation on GPU at native 1024x1024 resolution.
    /// Includes T5-XXL text encoder for maximum quality.
    /// </summary>
    [Fact]
    public void Gpu_SingleFile_GenerateImage_1024_WithT5()
    {
        if (!File.Exists(Sd3CheckpointPath))
        {
            _output.WriteLine($"SKIPPED: Checkpoint not found: {Sd3CheckpointPath}");
            return;
        }
        if (!File.Exists(TokenizerVocabPath) || !File.Exists(TokenizerMergesPath))
        {
            _output.WriteLine("SKIPPED: CLIP tokenizer files not found");
            return;
        }

        string assemblyDir = Path.GetDirectoryName(typeof(Sd3GenerationTests).Assembly.Location)!;
        string ptxDir = Path.Combine(assemblyDir, "Ptx");
        if (!Directory.Exists(ptxDir))
        {
            _output.WriteLine($"SKIPPED: PTX directory not found: {ptxDir}");
            return;
        }

        Stopwatch totalSw = Stopwatch.StartNew();

        // 1. Load and convert checkpoint
        _output.WriteLine($"[1/8] Loading checkpoint: {Path.GetFileName(Sd3CheckpointPath)}");
        Stopwatch sw = Stopwatch.StartNew();
        (Sd3CheckpointConverter.ConvertedWeights converted, SafeTensorsLoader loader) =
            Sd3CheckpointConverter.LoadAndConvert(Sd3CheckpointPath);
        sw.Stop();
        _output.WriteLine($"  Checkpoint loaded and converted in {sw.ElapsedMilliseconds}ms");

        using (loader)
        {
            Dictionary<string, Tensor> transformerF32 = CastWeightsToF32(converted.Transformer);
            Dictionary<string, Tensor> clipLF32 = CastWeightsToF32(converted.ClipL);
            Dictionary<string, Tensor> clipGF32 = CastWeightsToF32(converted.ClipG);
            Dictionary<string, Tensor> t5F32 = CastWeightsToF32(converted.T5);
            Dictionary<string, Tensor> vaeF32 = CastWeightsToF32(converted.Vae);

            // 2. Tokenize
            _output.WriteLine("[2/8] Tokenizing prompt...");
            using ClipTokenizer clipTokenizer = new(TokenizerVocabPath, TokenizerMergesPath);

            string prompt = "A photograph of an astronaut riding a horse";
            string negPrompt = "";

            int[] promptTokensL = clipTokenizer.Encode(prompt);
            int[] negTokensL = clipTokenizer.Encode(negPrompt);
            int[] promptTokensG = clipTokenizer.Encode(prompt);
            int[] negTokensG = clipTokenizer.Encode(negPrompt);

            int promptEosL = ClipTokenizer.FindEosPosition(promptTokensL);
            int negEosL = ClipTokenizer.FindEosPosition(negTokensL);
            int promptEosG = ClipTokenizer.FindEosPosition(promptTokensG);
            int negEosG = ClipTokenizer.FindEosPosition(negTokensG);

            int[]? promptTokensT5 = null;
            int[]? negTokensT5 = null;
            int[]? promptMaskT5 = null;
            int[]? negMaskT5 = null;
            T5TextEncoder? t5Encoder = null;

            if (File.Exists(T5SpieceModelPath) && t5F32.Count > 0)
            {
                _output.WriteLine("  Tokenizing with T5...");
                using T5Tokenizer t5Tokenizer = new(T5SpieceModelPath);
                promptTokensT5 = t5Tokenizer.Encode(prompt);
                negTokensT5 = t5Tokenizer.Encode(negPrompt);

                // Attention mask: 1 for non-pad, 0 for pad
                promptMaskT5 = CreateAttentionMask(promptTokensT5);
                negMaskT5 = CreateAttentionMask(negTokensT5);

                _output.WriteLine($"  T5 prompt tokens: {promptTokensT5.Length}");

                // Load T5 encoder
                _output.WriteLine("[3/8] Loading T5-XXL encoder...");
                sw.Restart();
                t5Encoder = new T5TextEncoder(T5TextEncoderConfig.Xxl);
                t5Encoder.LoadWeights(t5F32);
                sw.Stop();
                _output.WriteLine($"  T5-XXL loaded in {sw.ElapsedMilliseconds}ms ({t5F32.Count} tensors)");
            }
            else
            {
                _output.WriteLine("[3/8] SKIPPING T5 (tokenizer or weights not available)");
            }

            // 4. Load CLIP-L
            _output.WriteLine("[4/8] Loading CLIP-L...");
            sw.Restart();
            ClipTextEncoder clipL = new(ClipTextEncoderConfig.SdxlClipL);
            clipL.LoadWeights(clipLF32, "text_model");
            sw.Stop();
            _output.WriteLine($"  CLIP-L loaded in {sw.ElapsedMilliseconds}ms");

            // 5. Load CLIP-G
            _output.WriteLine("[5/8] Loading CLIP-G...");
            sw.Restart();
            ClipTextEncoder clipG = new(ClipTextEncoderConfig.SdxlClipG);
            clipG.LoadWeights(clipGF32, "text_model");
            sw.Stop();
            _output.WriteLine($"  CLIP-G loaded in {sw.ElapsedMilliseconds}ms");

            // 6. Load SD3 Transformer
            _output.WriteLine("[6/8] Loading SD3 MMDiT transformer...");
            sw.Restart();
            int depth = Sd3CheckpointConverter.DetectDepth(transformerF32);
            _output.WriteLine($"  Detected depth: {depth}");
            Sd3Config config = Sd3Config.Medium;
            Sd3Transformer transformer = new Sd3Transformer(config);
            transformer.LoadWeights(transformerF32);
            sw.Stop();
            _output.WriteLine($"  Transformer loaded in {sw.ElapsedMilliseconds}ms");

            // 7. Load VAE
            _output.WriteLine("[7/8] Loading VAE...");
            sw.Restart();
            VaeDecoder vae = new(VaeConfig.Sd3);
            vae.LoadWeights(vaeF32);
            sw.Stop();
            _output.WriteLine($"  VAE loaded in {sw.ElapsedMilliseconds}ms");

            // 8. Create CUDA backend and pipeline
            _output.WriteLine("[8/8] Initializing CUDA backend...");
            sw.Restart();
            using CudaBackend backend = new(deviceOrdinal: 0, ptxDir: ptxDir);
            sw.Stop();
            _output.WriteLine($"  CUDA backend initialized in {sw.ElapsedMilliseconds}ms");
            _output.WriteLine($"  Device: {backend.Capabilities.Name}");

            using Sd3Pipeline pipeline = new(backend, clipL, clipG, t5Encoder, transformer, vae, 3.0f);

            TextToImageRequest request = new()
            {
                Prompt = prompt,
                NegativePrompt = negPrompt,
                Width = 1024,
                Height = 1024,
                Steps = 28,
                CfgScale = 7.0f,
                Seed = 42,
            };

            _output.WriteLine($"\nGenerating {request.Width}x{request.Height} image, {request.Steps} steps, cfg={request.CfgScale}, seed=42 [GPU]...");
            Stopwatch genSw = Stopwatch.StartNew();

            (byte[] rgbData, int width, int height, int seed) = pipeline.GenerateFromTokens(
                promptTokensL, negTokensL,
                promptTokensG, negTokensG,
                promptEosL, negEosL,
                promptEosG, negEosG,
                promptTokensT5, negTokensT5,
                promptMaskT5, negMaskT5,
                request,
                progress => _output.WriteLine($"  Step {progress.Step}/{progress.TotalSteps} ({progress.ElapsedMs:F0}ms)"));

            genSw.Stop();
            _output.WriteLine($"\nGeneration complete in {genSw.Elapsed.TotalSeconds:F1} seconds (seed={seed})");

            // Validate output
            Assert.Equal(1024, width);
            Assert.Equal(1024, height);
            Assert.Equal(1024 * 1024 * 3, rgbData.Length);

            ValidateImageNotDegenerate(rgbData);

            // Save output
            Directory.CreateDirectory(OutputDir);
            string t5Suffix = t5Encoder is not null ? "withT5" : "noT5";
            string outputPath = Path.Combine(OutputDir, $"sd3_gpu_1024_{t5Suffix}_{DateTime.Now:yyyyMMdd_HHmmss}.bmp");
            ImagePostProcessor.SaveBmp(outputPath, rgbData, width, height);
            _output.WriteLine($"  Image saved to: {outputPath}");

            totalSw.Stop();
            _output.WriteLine($"\nTotal time: {totalSw.Elapsed.TotalSeconds:F1} seconds");

            transformer.Dispose();
            t5Encoder?.Dispose();
        }
    }

    /// <summary>
    /// SD3 GPU generation at small resolution for fast iteration.
    /// 256x256 with 5 steps, no CFG for single forward pass per step.
    /// </summary>
    [Fact]
    public void Gpu_SingleFile_GenerateImage_Small()
    {
        if (!File.Exists(Sd3CheckpointPath))
        {
            _output.WriteLine($"SKIPPED: Checkpoint not found: {Sd3CheckpointPath}");
            return;
        }
        if (!File.Exists(TokenizerVocabPath) || !File.Exists(TokenizerMergesPath))
        {
            _output.WriteLine("SKIPPED: CLIP tokenizer files not found");
            return;
        }

        string assemblyDir = Path.GetDirectoryName(typeof(Sd3GenerationTests).Assembly.Location)!;
        string ptxDir = Path.Combine(assemblyDir, "Ptx");
        if (!Directory.Exists(ptxDir))
        {
            _output.WriteLine($"SKIPPED: PTX directory not found: {ptxDir}");
            return;
        }

        Stopwatch totalSw = Stopwatch.StartNew();

        _output.WriteLine($"[1/7] Loading checkpoint: {Path.GetFileName(Sd3CheckpointPath)}");
        Stopwatch sw = Stopwatch.StartNew();
        (Sd3CheckpointConverter.ConvertedWeights converted, SafeTensorsLoader loader) =
            Sd3CheckpointConverter.LoadAndConvert(Sd3CheckpointPath);
        sw.Stop();
        _output.WriteLine($"  Checkpoint loaded and converted in {sw.ElapsedMilliseconds}ms");

        using (loader)
        {
            Dictionary<string, Tensor> transformerF32 = CastWeightsToF32(converted.Transformer);
            Dictionary<string, Tensor> clipLF32 = CastWeightsToF32(converted.ClipL);
            Dictionary<string, Tensor> clipGF32 = CastWeightsToF32(converted.ClipG);
            Dictionary<string, Tensor> vaeF32 = CastWeightsToF32(converted.Vae);

            _output.WriteLine("[2/7] Tokenizing prompt...");
            using ClipTokenizer tokenizer = new(TokenizerVocabPath, TokenizerMergesPath);

            string prompt = "A photograph of an astronaut riding a horse";
            string negPrompt = "";

            int[] promptTokensL = tokenizer.Encode(prompt);
            int[] negTokensL = tokenizer.Encode(negPrompt);
            int[] promptTokensG = tokenizer.Encode(prompt);
            int[] negTokensG = tokenizer.Encode(negPrompt);

            int promptEosL = ClipTokenizer.FindEosPosition(promptTokensL);
            int negEosL = ClipTokenizer.FindEosPosition(negTokensL);
            int promptEosG = ClipTokenizer.FindEosPosition(promptTokensG);
            int negEosG = ClipTokenizer.FindEosPosition(negTokensG);

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

            _output.WriteLine("[5/7] Loading SD3 MMDiT transformer...");
            sw.Restart();
            Sd3Config config = Sd3Config.Medium;
            Sd3Transformer transformer = new Sd3Transformer(config);
            transformer.LoadWeights(transformerF32);
            sw.Stop();
            _output.WriteLine($"  Transformer loaded in {sw.ElapsedMilliseconds}ms");

            _output.WriteLine("[6/7] Loading VAE...");
            sw.Restart();
            VaeDecoder vae = new(VaeConfig.Sd3);
            vae.LoadWeights(vaeF32);
            sw.Stop();
            _output.WriteLine($"  VAE loaded in {sw.ElapsedMilliseconds}ms");

            _output.WriteLine("[7/7] Initializing CUDA backend...");
            sw.Restart();
            using CudaBackend backend = new(deviceOrdinal: 0, ptxDir: ptxDir);
            sw.Stop();
            _output.WriteLine($"  CUDA backend initialized in {sw.ElapsedMilliseconds}ms");
            _output.WriteLine($"  Device: {backend.Capabilities.Name}");

            using Sd3Pipeline pipeline = new(backend, clipL, clipG, null, transformer, vae, 3.0f);

            TextToImageRequest request = new()
            {
                Prompt = prompt,
                NegativePrompt = negPrompt,
                Width = 256,
                Height = 256,
                Steps = 5,
                CfgScale = 1.0f,
                Seed = 42,
            };

            _output.WriteLine($"\nGenerating {request.Width}x{request.Height} image, {request.Steps} steps, cfg={request.CfgScale}, seed=42 [GPU]...");
            Stopwatch genSw = Stopwatch.StartNew();

            (byte[] rgbData, int width, int height, int seed) = pipeline.GenerateFromTokens(
                promptTokensL, negTokensL,
                promptTokensG, negTokensG,
                promptEosL, negEosL,
                promptEosG, negEosG,
                null, null, null, null,
                request,
                progress => _output.WriteLine($"  Step {progress.Step}/{progress.TotalSteps} ({progress.ElapsedMs:F0}ms)"));

            genSw.Stop();
            _output.WriteLine($"\nGeneration complete in {genSw.Elapsed.TotalSeconds:F1} seconds (seed={seed})");

            Assert.Equal(256, width);
            Assert.Equal(256, height);
            Assert.Equal(256 * 256 * 3, rgbData.Length);

            ValidateImageNotDegenerate(rgbData);

            Directory.CreateDirectory(OutputDir);
            string outputPath = Path.Combine(OutputDir, $"sd3_gpu_256_{DateTime.Now:yyyyMMdd_HHmmss}.bmp");
            ImagePostProcessor.SaveBmp(outputPath, rgbData, width, height);
            _output.WriteLine($"  Image saved to: {outputPath}");

            totalSw.Stop();
            _output.WriteLine($"\nTotal time: {totalSw.Elapsed.TotalSeconds:F1} seconds");

            transformer.Dispose();
        }
    }

    /// <summary>Validates the image has meaningful pixel content (not all black or all white).</summary>
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

    /// <summary>Creates a T5 attention mask (1 for non-pad tokens, 0 for pad tokens).</summary>
    private static int[] CreateAttentionMask(int[] tokenIds)
    {
        int[] mask = new int[tokenIds.Length];
        for (int i = 0; i < tokenIds.Length; i++)
        {
            mask[i] = tokenIds[i] != T5Tokenizer.PadTokenId ? 1 : 0;
        }
        return mask;
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
