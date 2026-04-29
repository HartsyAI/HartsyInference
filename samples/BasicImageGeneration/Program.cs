using System.Diagnostics;
using SharpInference.Core.Backends;
using SharpInference.Core.Logging;
using SharpInference.Core.Tensors;
using SharpInference.Cpu;
using SharpInference.Cuda;
using SharpInference.Vulkan;
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
    #region Configuration

    private static readonly string SamplesRoot = Path.GetFullPath(
        Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", ".."));

    private static readonly ModelPaths DefaultModelPaths = ModelPaths.FromComfyLayout(
        modelsRoot: Path.Combine(SamplesRoot, "Models"),
        checkpointName: "StabilityAI/sd-v1-5");

    private static readonly OutputSettings DefaultOutputSettings = new OutputSettings
    {
        OutputDir = Path.Combine(SamplesRoot, "Output"),
    };

    #endregion

    #region Generation Settings

    private const int ImageWidth = 256;
    private const int ImageHeight = 256;
    private const int Steps = 20;
    private const float CfgScale = 7.5f;
    private const int Seed = 42;
    private const string Prompt = "a painting of a cat sitting on a windowsill";
    private const string NegativePrompt = "blurry, bad quality";

    #endregion

    public static void Main(string[] args)
    {
        Logs.MinLevel = LogLevel.Debug;

        // --backend cpu|vulkan|cuda  (default: cpu).
        // Positional args still work: <modelDir> <outputDir>.
        string backendChoice = "cpu";
        List<string> positionals = new(args.Length);
        for (int i = 0; i < args.Length; i++)
        {
            string a = args[i];
            if (a == "--backend" && i + 1 < args.Length) { backendChoice = args[++i].ToLowerInvariant(); }
            else if (a.StartsWith("--backend=", StringComparison.Ordinal)) { backendChoice = a["--backend=".Length..].ToLowerInvariant(); }
            else if (a is "-h" or "--help")
            {
                Console.WriteLine("Usage: BasicImageGeneration [--backend cpu|vulkan|cuda] [modelDir] [outputDir]");
                return;
            }
            else positionals.Add(a);
        }

        ModelPaths modelPaths = positionals.Count > 0
            ? ModelPaths.FromHuggingFaceDir(positionals[0])
            : DefaultModelPaths;

        OutputSettings outputSettings = positionals.Count > 1
            ? new OutputSettings { OutputDir = positionals[1] }
            : DefaultOutputSettings;

        Console.WriteLine("╔══════════════════════════════════════════════╗");
        Console.WriteLine("║  SharpInference — SD1.5 Image Generation     ║");
        Console.WriteLine("╚══════════════════════════════════════════════╝");
        Console.WriteLine();
        Console.WriteLine($"Model paths:");
        Console.WriteLine($"  Tokenizer:    {modelPaths.TokenizerDir}");
        Console.WriteLine($"  Text encoder: {modelPaths.TextEncoderPath}");
        Console.WriteLine($"  UNet:         {modelPaths.UNetPath}");
        Console.WriteLine($"  VAE:          {modelPaths.VaePath}");
        Console.WriteLine();
        Console.WriteLine($"Output dir: {outputSettings.OutputDir}");
        Console.WriteLine();

        modelPaths.Validate();
        Console.WriteLine("All model files found.");

        Console.WriteLine($"Prompt: \"{Prompt}\"");
        Console.WriteLine($"Size: {ImageWidth}x{ImageHeight}, Steps: {Steps}, CFG: {CfgScale}, Seed: {Seed}");
        Console.WriteLine();

        Stopwatch totalSw = Stopwatch.StartNew();

        // Create backend
        using IBackend backend = backendChoice switch
        {
            "vulkan" => new VulkanBackend(),
            "cuda" => new CudaBackend(deviceOrdinal: 0,
                ptxDir: Path.Combine(AppContext.BaseDirectory, "Ptx")),
            "cpu" => new CpuBackend(),
            _ => throw new ArgumentException($"Unknown backend '{backendChoice}'. Valid: cpu, vulkan, cuda."),
        };
        Console.WriteLine($"[1/5] Backend: {backend.Capabilities.Name}");

        // Load tokenizer
        Console.WriteLine("[2/5] Loading CLIP tokenizer...");
        using ClipTokenizer tokenizer = new ClipTokenizer(modelPaths.VocabPath, modelPaths.MergesPath);
        int[] promptTokens = tokenizer.Encode(Prompt);
        int[] negativeTokens = tokenizer.Encode(NegativePrompt);
        Console.WriteLine($"  Prompt tokens: {CountNonZero(promptTokens)} non-padding");
        Console.WriteLine($"  Negative tokens: {CountNonZero(negativeTokens)} non-padding");

        // Load text encoder
        Console.WriteLine("[3/5] Loading CLIP text encoder...");
        Stopwatch sw = Stopwatch.StartNew();
        ClipTextEncoderConfig clipConfig = ClipTextEncoderConfig.Sd15;
        ClipTextEncoder textEncoder = new ClipTextEncoder(clipConfig);

        using SafeTensorsLoader textEncoderLoader = new SafeTensorsLoader();
        textEncoderLoader.Load(modelPaths.TextEncoderPath);
        Dictionary<string, Tensor> textEncoderWeights = CastWeightsToF32(textEncoderLoader.GetAllTensors());
        textEncoder.LoadWeights(textEncoderWeights, "text_model");
        sw.Stop();
        Console.WriteLine($"  Text encoder loaded in {sw.ElapsedMilliseconds}ms ({textEncoderWeights.Count} tensors)");

        // Load UNet
        Console.WriteLine("[4/5] Loading UNet...");
        sw.Restart();
        UNetConfig unetConfig = UNetConfig.Sd15;
        UNet unet = new UNet(unetConfig);

        using SafeTensorsLoader unetLoader = new SafeTensorsLoader();
        unetLoader.Load(modelPaths.UNetPath);
        Dictionary<string, Tensor> unetWeights = CastWeightsToF32(unetLoader.GetAllTensors());
        unet.LoadWeights(unetWeights);
        sw.Stop();
        Console.WriteLine($"  UNet loaded in {sw.ElapsedMilliseconds}ms ({unetWeights.Count} tensors)");

        // Load VAE decoder
        Console.WriteLine("[5/5] Loading VAE decoder...");
        sw.Restart();
        VaeConfig vaeConfig = VaeConfig.Sd15;
        VaeDecoder vaeDecoder = new VaeDecoder(vaeConfig);

        using SafeTensorsLoader vaeLoader = new SafeTensorsLoader();
        vaeLoader.Load(modelPaths.VaePath);
        Dictionary<string, Tensor> vaeWeights = CastWeightsToF32(vaeLoader.GetAllTensors());
        vaeDecoder.LoadWeights(vaeWeights);
        sw.Stop();
        Console.WriteLine($"  VAE loaded in {sw.ElapsedMilliseconds}ms ({vaeWeights.Count} tensors)");

        Console.WriteLine();
        Console.WriteLine("All models loaded. Starting inference...");
        Console.WriteLine();

        // Run inference pipeline
        using StableDiffusion15Pipeline pipeline = new StableDiffusion15Pipeline(backend, textEncoder, unet, vaeDecoder);

        TextToImageRequest request = new TextToImageRequest
        {
            Prompt = Prompt,
            NegativePrompt = NegativePrompt,
            Width = ImageWidth,
            Height = ImageHeight,
            Steps = Steps,
            CfgScale = CfgScale,
            Seed = Seed,
        };

        (byte[] rgbData, int outW, int outH, int usedSeed) = pipeline.GenerateFromTokens(
            promptTokens, negativeTokens, request,
            progress => Console.WriteLine($"  Step {progress.Step}/{progress.TotalSteps} ({progress.ElapsedMs:F0}ms)"));

        // Save output
        string promptSlug = Prompt.Length > 30 ? Prompt[..30] : Prompt;
        string outputPath = outputSettings.GetNextOutputPath(promptSlug);
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
