using System.Diagnostics;
using SharpInference.Core.Tensors;
using SharpInference.Cpu;
using SharpInference.Diffusion.Models.Denoisers;
using SharpInference.Diffusion.Models.TextEncoders;
using SharpInference.Diffusion.Models.Vae;
using SharpInference.Diffusion.Pipelines;
using SharpInference.Diffusion.Requests;
using SharpInference.Diffusion.Utilities;
using SharpInference.ModelHandler.CheckpointConverters;
using SharpInference.ModelHandler.Lora;
using SharpInference.ModelHandler.SafeTensors;
using SharpInference.Tokenizers;
using Xunit;
using Xunit.Abstractions;

namespace SharpInference.Diffusion.Tests;

/// <summary>End-to-end SDXL+LoRA generation test. Loads an SDXL checkpoint, merges a LoRA into the UNet/CLIP weights via LoraStack, runs the full CPU pipeline, and verifies the output is non-degenerate. Skips cleanly when SDXL_LORA_PATH is not set or files are missing.</summary>
public class SdxlLoraGenerationTests
{
    private static readonly string SdxlCheckpointPath =
        Environment.GetEnvironmentVariable("SDXL_SINGLE_FILE_PATH")
        ?? @"C:\Users\kaleb\Desktop\Projects\SwarmUI\Models\Stable-Diffusion\SDXL\Juggernaut_XL_-_Ragnarok_by_RunDiffusion.safetensors";

    private static readonly string? SdxlLoraPath =
        Environment.GetEnvironmentVariable("SDXL_LORA_PATH");

    private static readonly string TokenizerVocabPath =
        Environment.GetEnvironmentVariable("CLIP_VOCAB_PATH")
        ?? @"C:\Users\kaleb\Desktop\projects\SharpInference\tests\test-models\clip_vocab.json";

    private static readonly string TokenizerMergesPath =
        Environment.GetEnvironmentVariable("CLIP_MERGES_PATH")
        ?? @"C:\Users\kaleb\Desktop\projects\SharpInference\tests\test-models\clip_merges.txt";

    private static readonly string OutputDir =
        Environment.GetEnvironmentVariable("SDXL_OUTPUT_DIR")
        ?? @"C:\Users\kaleb\Desktop\projects\SharpInference\Output";

    private readonly ITestOutputHelper _output;

    public SdxlLoraGenerationTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void SingleFile_WithLora_GenerateImage_Cpu_Small()
    {
        if (string.IsNullOrEmpty(SdxlLoraPath) || !File.Exists(SdxlLoraPath))
        {
            _output.WriteLine($"SKIPPED: Set SDXL_LORA_PATH to a Kohya-format SDXL LoRA .safetensors file (got '{SdxlLoraPath}').");
            return;
        }
        if (!File.Exists(SdxlCheckpointPath))
        {
            _output.WriteLine($"SKIPPED: SDXL checkpoint not found: {SdxlCheckpointPath}");
            return;
        }
        if (!File.Exists(TokenizerVocabPath) || !File.Exists(TokenizerMergesPath))
        {
            _output.WriteLine("SKIPPED: tokenizer files not found.");
            return;
        }

        Stopwatch totalSw = Stopwatch.StartNew();

        _output.WriteLine($"[1/7] Loading checkpoint: {Path.GetFileName(SdxlCheckpointPath)}");
        Stopwatch sw = Stopwatch.StartNew();
        (SdxlCheckpointConverter.ConvertedWeights converted, SafeTensorsLoader loader) =
            SdxlCheckpointConverter.LoadAndConvert(SdxlCheckpointPath);
        sw.Stop();
        _output.WriteLine($"  Checkpoint loaded + converted in {sw.ElapsedMilliseconds}ms");

        using (loader)
        {
            Dictionary<string, Tensor> unetF32 = CastWeightsToF32(converted.UNet);
            Dictionary<string, Tensor> clipLF32 = CastWeightsToF32(converted.ClipL);
            Dictionary<string, Tensor> clipGF32 = CastWeightsToF32(converted.ClipG);
            Dictionary<string, Tensor> vaeF32 = CastWeightsToF32(converted.Vae);

            using CpuBackend backend = new();

            _output.WriteLine($"[2/7] Loading LoRA: {Path.GetFileName(SdxlLoraPath)}");
            sw.Restart();
            using LoraFile lora = LoraFile.Load(SdxlLoraPath!);
            using LoraStack stack = new();
            stack.Add(lora, strength: 1.0f);
            sw.Stop();
            _output.WriteLine($"  LoRA loaded in {sw.ElapsedMilliseconds}ms (format={lora.Format}, layers={lora.Layers.Count})");

            _output.WriteLine("[3/7] Merging LoRA deltas...");
            sw.Restart();
            int unetMerged = stack.ApplyTo(unetF32, LoraTarget.UNet, backend);
            int clipLMerged = stack.ApplyTo(clipLF32, LoraTarget.ClipL, backend);
            int clipGMerged = stack.ApplyTo(clipGF32, LoraTarget.ClipG, backend);
            sw.Stop();
            _output.WriteLine($"  Merged {unetMerged} UNet + {clipLMerged} CLIP-L + {clipGMerged} CLIP-G weights in {sw.ElapsedMilliseconds}ms");

            _output.WriteLine("[4/7] Tokenizing...");
            using ClipTokenizer tokenizer = new(TokenizerVocabPath, TokenizerMergesPath);
            string prompt = "a majestic lion in a field of sunflowers, golden hour, photorealistic";
            string negPrompt = "blurry, low quality, deformed";
            int[] promptTokensL = tokenizer.Encode(prompt);
            int[] negTokensL = tokenizer.Encode(negPrompt);
            int[] promptTokensG = tokenizer.Encode(prompt);
            int[] negTokensG = tokenizer.Encode(negPrompt);
            int promptEosG = ClipTokenizer.FindEosPosition(promptTokensG);
            int negEosG = ClipTokenizer.FindEosPosition(negTokensG);

            _output.WriteLine("[5/7] Building model components...");
            sw.Restart();
            ClipTextEncoder clipL = new(ClipTextEncoderConfig.SdxlClipL);
            clipL.LoadWeights(clipLF32, "text_model");
            ClipTextEncoder clipG = new(ClipTextEncoderConfig.SdxlClipG);
            clipG.LoadWeights(clipGF32, "text_model");
            UNet unet = new(UNetConfig.SdxlBase);
            unet.LoadWeights(unetF32);
            VaeDecoder vae = new(VaeConfig.Sdxl);
            vae.LoadWeights(vaeF32);
            sw.Stop();
            _output.WriteLine($"  Models built in {sw.ElapsedMilliseconds}ms");

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

            _output.WriteLine($"[6/7] Generating {request.Width}x{request.Height}, {request.Steps} steps, seed={request.Seed}...");
            Stopwatch genSw = Stopwatch.StartNew();
            (byte[] rgbData, int width, int height, int seed) = pipeline.GenerateFromTokens(
                promptTokensL, negTokensL,
                promptTokensG, negTokensG,
                promptEosG, negEosG,
                request,
                progress => _output.WriteLine($"  Step {progress.Step}/{progress.TotalSteps} ({progress.ElapsedMs:F0}ms)"));
            genSw.Stop();
            _output.WriteLine($"  Generation: {genSw.Elapsed.TotalMinutes:F1} min");

            Assert.Equal(128, width);
            Assert.Equal(128, height);
            Assert.Equal(128 * 128 * 3, rgbData.Length);

            int nonZero = 0, nonFF = 0;
            foreach (byte b in rgbData) { if (b != 0) nonZero++; if (b != 255) nonFF++; }
            float nzPct = nonZero / (float)rgbData.Length * 100;
            float nfPct = nonFF / (float)rgbData.Length * 100;
            _output.WriteLine($"  Non-zero: {nzPct:F1}%, Non-255: {nfPct:F1}%");
            Assert.True(nzPct > 10, "image is all black");
            Assert.True(nfPct > 10, "image is all white");

            _output.WriteLine("[7/7] Saving output...");
            Directory.CreateDirectory(OutputDir);
            string outputPath = Path.Combine(OutputDir, $"sdxl_lora_{DateTime.Now:yyyyMMdd_HHmmss}.bmp");
            ImagePostProcessor.SaveBmp(outputPath, rgbData, width, height);
            _output.WriteLine($"  Saved: {outputPath}");
            _output.WriteLine($"\nTotal: {totalSw.Elapsed.TotalMinutes:F1} min");
        }
    }

    private static Dictionary<string, Tensor> CastWeightsToF32(Dictionary<string, Tensor> weights)
    {
        Dictionary<string, Tensor> result = new(weights.Count);
        foreach ((string k, Tensor t) in weights)
        {
            result[k] = t.DType == DType.F32 ? t : t.CastTo(DType.F32);
        }
        return result;
    }
}
