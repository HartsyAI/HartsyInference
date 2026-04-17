using System.Diagnostics;
using SharpInference.Core.Logging;
using SharpInference.Core.Tensors;
using SharpInference.Cpu;
using SharpInference.Diffusion.Models.Denoisers;
using SharpInference.Diffusion.Models.TextEncoders;
using SharpInference.Diffusion.Models.Vae;
using SharpInference.Diffusion.Pipelines;
using SharpInference.Diffusion.Requests;
using SharpInference.Diffusion.Utilities;
using SharpInference.ModelHandler.SafeTensors;
using SharpInference.Tokenizers;

namespace BasicImageGeneration;

public static class Program
{
    public static void Main(string[] args)
    {
        Logs.MinLevel = LogLevel.Debug;

        string modelDir = args.Length > 0
            ? args[0]
            : Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "tests", "test-models", "sd15");

        modelDir = Path.GetFullPath(modelDir);

        string outputPath = args.Length > 1
            ? args[1]
            : Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Desktop), "sharpinference_output.bmp");

        Console.WriteLine("╔══════════════════════════════════════════════╗");
        Console.WriteLine("║  SharpInference — SD1.5 CPU Image Generation ║");
        Console.WriteLine("╚══════════════════════════════════════════════╝");
        Console.WriteLine();
        Console.WriteLine($"Model directory: {modelDir}");
        Console.WriteLine($"Output path:     {outputPath}");
        Console.WriteLine();

        // Validate files exist
        string tokenizerVocab = Path.Combine(modelDir, "tokenizer", "vocab.json");
        string tokenizerMerges = Path.Combine(modelDir, "tokenizer", "merges.txt");
        string textEncoderPath = Path.Combine(modelDir, "text_encoder", "model.fp16.safetensors");
        string unetPath = Path.Combine(modelDir, "unet", "diffusion_pytorch_model.fp16.safetensors");
        string vaePath = Path.Combine(modelDir, "vae", "diffusion_pytorch_model.fp16.safetensors");

        string[] requiredFiles = [tokenizerVocab, tokenizerMerges, textEncoderPath, unetPath, vaePath];
        foreach (string file in requiredFiles)
        {
            if (!File.Exists(file))
            {
                Console.Error.WriteLine($"ERROR: Required file not found: {file}");
                return;
            }
        }
        Console.WriteLine("All model files found.");

        // Settings — small resolution and few steps for CPU
        int width = 256;
        int height = 256;
        int steps = 20;
        float cfgScale = 7.5f;
        int seed = 42;
        string prompt = "a painting of a cat sitting on a windowsill";
        string negativePrompt = "blurry, bad quality";

        Console.WriteLine($"Prompt: \"{prompt}\"");
        Console.WriteLine($"Size: {width}x{height}, Steps: {steps}, CFG: {cfgScale}, Seed: {seed}");
        Console.WriteLine();

        Stopwatch totalSw = Stopwatch.StartNew();

        // ── 1. Create backend ──
        using CpuBackend backend = new CpuBackend();
        Console.WriteLine("[1/5] CPU backend created.");

        // ── 2. Load tokenizer ──
        Console.WriteLine("[2/5] Loading CLIP tokenizer...");
        using ClipTokenizer tokenizer = new ClipTokenizer(tokenizerVocab, tokenizerMerges);
        int[] promptTokens = tokenizer.Encode(prompt);
        int[] negativeTokens = tokenizer.Encode(negativePrompt);
        Console.WriteLine($"  Prompt tokens: {CountNonZero(promptTokens)} non-padding");
        Console.WriteLine($"  Negative tokens: {CountNonZero(negativeTokens)} non-padding");

        // ── 3. Load text encoder ──
        Console.WriteLine("[3/5] Loading CLIP text encoder...");
        Stopwatch sw = Stopwatch.StartNew();
        ClipTextEncoderConfig clipConfig = ClipTextEncoderConfig.Sd15;
        ClipTextEncoder textEncoder = new ClipTextEncoder(clipConfig);

        using SafeTensorsLoader textEncoderLoader = new SafeTensorsLoader();
        textEncoderLoader.Load(textEncoderPath);
        Dictionary<string, Tensor> textEncoderWeights = CastWeightsToF32(textEncoderLoader.GetAllTensors());
        textEncoder.LoadWeights(textEncoderWeights, "text_model");
        sw.Stop();
        Console.WriteLine($"  Text encoder loaded in {sw.ElapsedMilliseconds}ms ({textEncoderWeights.Count} tensors)");

        // ── 4. Load UNet ──
        Console.WriteLine("[4/5] Loading UNet...");
        sw.Restart();
        UNetConfig unetConfig = UNetConfig.Sd15;
        UNet unet = new UNet(unetConfig);

        using SafeTensorsLoader unetLoader = new SafeTensorsLoader();
        unetLoader.Load(unetPath);
        Dictionary<string, Tensor> unetWeights = CastWeightsToF32(unetLoader.GetAllTensors());
        unet.LoadWeights(unetWeights);
        sw.Stop();
        Console.WriteLine($"  UNet loaded in {sw.ElapsedMilliseconds}ms ({unetWeights.Count} tensors)");

        // ── 5. Load VAE ──
        Console.WriteLine("[5/5] Loading VAE decoder...");
        sw.Restart();
        VaeConfig vaeConfig = VaeConfig.Sd15;
        VaeDecoder vaeDecoder = new VaeDecoder(vaeConfig);

        using SafeTensorsLoader vaeLoader = new SafeTensorsLoader();
        vaeLoader.Load(vaePath);
        Dictionary<string, Tensor> vaeWeights = CastWeightsToF32(vaeLoader.GetAllTensors());
        vaeDecoder.LoadWeights(vaeWeights);
        sw.Stop();
        Console.WriteLine($"  VAE loaded in {sw.ElapsedMilliseconds}ms ({vaeWeights.Count} tensors)");

        Console.WriteLine();
        Console.WriteLine("All models loaded. Starting inference...");
        Console.WriteLine();

        // ── Full pipeline with CFG ──
        using StableDiffusion15Pipeline pipeline = new StableDiffusion15Pipeline(backend, textEncoder, unet, vaeDecoder);

        TextToImageRequest request = new TextToImageRequest
        {
            Prompt = prompt,
            NegativePrompt = negativePrompt,
            Width = width,
            Height = height,
            Steps = steps,
            CfgScale = cfgScale,
            Seed = seed,
        };

        (byte[] rgbData, int outW, int outH, int usedSeed) = pipeline.GenerateFromTokens(
            promptTokens, negativeTokens, request,
            progress => Console.WriteLine($"  Step {progress.Step}/{progress.TotalSteps} ({progress.ElapsedMs:F0}ms)"));

        ImagePostProcessor.SaveBmp(outputPath, rgbData, outW, outH);

        totalSw.Stop();
        Console.WriteLine($"\nGeneration complete in {totalSw.Elapsed.TotalSeconds:F1}s (seed={usedSeed})");
        Console.WriteLine($"Image saved to: {outputPath}");
    }

    /// <summary>Casts all FP16/BF16 tensors to FP32 for CPU inference.</summary>
    private static Dictionary<string, Tensor> CastWeightsToF32(Dictionary<string, Tensor> weights)
    {
        Dictionary<string, Tensor> f32Weights = new Dictionary<string, Tensor>(weights.Count);

        foreach (KeyValuePair<string, Tensor> kvp in weights)
        {
            if (kvp.Value.DType == DType.F16 || kvp.Value.DType == DType.BF16)
            {
                Tensor cast = kvp.Value.CastTo(DType.F32);
                f32Weights[kvp.Key] = cast;
            }
            else
            {
                f32Weights[kvp.Key] = kvp.Value;
            }
        }

        return f32Weights;
    }

    private static int CountNonZero(int[] arr)
    {
        int count = 0;
        foreach (int v in arr)
        {
            if (v != 0) count++;
        }
        return count;
    }
}
