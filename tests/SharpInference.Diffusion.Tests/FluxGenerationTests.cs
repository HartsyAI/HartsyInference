using System.Diagnostics;
using Xunit;
using Xunit.Abstractions;
using SharpInference.Core.Tensors;
using SharpInference.Cpu;
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
    private static readonly string FluxSingleFilePath =
        Environment.GetEnvironmentVariable("FLUX_SINGLE_FILE_PATH")
        ?? @"C:\Users\kaleb\Desktop\Projects\SwarmUI\Models\Stable-Diffusion\Flux\flux1-schnell.safetensors";

    private static readonly string TokenizerVocabPath =
        Environment.GetEnvironmentVariable("CLIP_VOCAB_PATH")
        ?? @"C:\Users\kaleb\Desktop\projects\SharpInference\tests\test-models\clip_vocab.json";

    private static readonly string TokenizerMergesPath =
        Environment.GetEnvironmentVariable("CLIP_MERGES_PATH")
        ?? @"C:\Users\kaleb\Desktop\projects\SharpInference\tests\test-models\clip_merges.txt";

    private static readonly string T5SpieceModelPath =
        Environment.GetEnvironmentVariable("T5_SPIECE_MODEL_PATH")
        ?? @"C:\Users\kaleb\Desktop\projects\SharpInference\tests\test-models\t5_spiece.model";

    private static readonly string OutputDir =
        Environment.GetEnvironmentVariable("FLUX_OUTPUT_DIR")
        ?? @"C:\Users\kaleb\Desktop\projects\SharpInference\Output";

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
            f32[kvp.Key] = (kvp.Value.DType == DType.F16 || kvp.Value.DType == DType.BF16)
                ? kvp.Value.CastTo(DType.F32)
                : kvp.Value;
        }
        return f32;
    }
}
