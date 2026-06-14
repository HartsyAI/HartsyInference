using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.ModelHandler.SafeTensors;
using HartsyInference.Tokenizers;
using HartsyInference.Vulkan;

namespace HartsyInference.Cli;

public static class Program
{
    private const int DefaultImageWidth = 256;
    private const int DefaultImageHeight = 256;
    private const int DefaultSteps = 20;
    private const float DefaultCfgScale = 7.5f;
    private const int DefaultSeed = 42;
    private const string DefaultPrompt = "a painting of a cat sitting on a windowsill";
    private const string DefaultNegativePrompt = "blurry, bad quality";

    public static int Main(string[] args)
    {
        Logs.MinLevel = LogLevel.Debug;

        CliOptions options = CliOptions.Parse(args);
        if (options.ShowHelp)
        {
            PrintHelp();
            return 0;
        }

        string repoRoot = ResolveRepoRoot();
        string modelsRoot = options.ModelsRoot ?? Path.Combine(repoRoot, "Models");
        string outputDir = options.OutputDir ?? Path.Combine(repoRoot, "Output");

        ModelPaths modelPaths = ModelPaths.FromComfyLayout(modelsRoot, "StabilityAI/sd-v1-5");
        OutputSettings outputSettings = new OutputSettings { OutputDir = outputDir };

        Console.WriteLine("HartsyInference CLI - SD1.5 image generation");
        Console.WriteLine($"  Models root:    {modelsRoot}");
        Console.WriteLine($"  Output dir:     {outputDir}");
        Console.WriteLine($"  Backend:        {options.Backend}");
        Console.WriteLine($"  Prompt:         \"{options.Prompt}\"");
        Console.WriteLine($"  Size:           {options.Width}x{options.Height}");
        Console.WriteLine($"  Steps/CFG/Seed: {options.Steps} / {options.CfgScale} / {options.Seed}");
        Console.WriteLine();

        modelPaths.Validate();

        Stopwatch totalSw = Stopwatch.StartNew();

        using IBackend backend = CreateBackend(options.Backend);
        Console.WriteLine($"[1/5] Backend ready: {backend.Capabilities.Name}");

        Console.WriteLine("[2/5] Loading CLIP tokenizer...");
        using ClipTokenizer tokenizer = new ClipTokenizer(modelPaths.VocabPath, modelPaths.MergesPath);
        int[] promptTokens = tokenizer.Encode(options.Prompt);
        int[] negativeTokens = tokenizer.Encode(options.NegativePrompt);

        Console.WriteLine("[3/5] Loading CLIP text encoder...");
        Stopwatch sw = Stopwatch.StartNew();
        ClipTextEncoder textEncoder = new ClipTextEncoder(ClipTextEncoderConfig.Sd15);
        using SafeTensorsLoader textEncoderLoader = new SafeTensorsLoader();
        textEncoderLoader.Load(modelPaths.TextEncoderPath);
        Dictionary<string, Tensor> textEncoderWeights = CastWeightsToF32(textEncoderLoader.GetAllTensors());
        textEncoder.LoadWeights(textEncoderWeights, "text_model");
        sw.Stop();
        Console.WriteLine($"      Loaded in {sw.ElapsedMilliseconds}ms ({textEncoderWeights.Count} tensors)");

        Console.WriteLine("[4/5] Loading UNet...");
        sw.Restart();
        UNet unet = new UNet(UNetConfig.Sd15);
        using SafeTensorsLoader unetLoader = new SafeTensorsLoader();
        unetLoader.Load(modelPaths.UNetPath);
        Dictionary<string, Tensor> unetWeights = CastWeightsToF32(unetLoader.GetAllTensors());
        unet.LoadWeights(unetWeights);
        sw.Stop();
        Console.WriteLine($"      Loaded in {sw.ElapsedMilliseconds}ms ({unetWeights.Count} tensors)");

        Console.WriteLine("[5/5] Loading VAE decoder...");
        sw.Restart();
        VaeDecoder vaeDecoder = new VaeDecoder(VaeConfig.Sd15);
        using SafeTensorsLoader vaeLoader = new SafeTensorsLoader();
        vaeLoader.Load(modelPaths.VaePath);
        Dictionary<string, Tensor> vaeWeights = CastWeightsToF32(vaeLoader.GetAllTensors());
        vaeDecoder.LoadWeights(vaeWeights);
        sw.Stop();
        Console.WriteLine($"      Loaded in {sw.ElapsedMilliseconds}ms ({vaeWeights.Count} tensors)");

        Console.WriteLine();
        Console.WriteLine("Running inference...");

        using StableDiffusion15Pipeline pipeline = new StableDiffusion15Pipeline(backend, textEncoder, unet, vaeDecoder);
        TextToImageRequest request = new TextToImageRequest
        {
            Prompt = options.Prompt,
            NegativePrompt = options.NegativePrompt,
            Width = options.Width,
            Height = options.Height,
            Steps = options.Steps,
            CfgScale = options.CfgScale,
            Seed = options.Seed,
        };

        (byte[] rgbData, int outW, int outH, int usedSeed) = pipeline.GenerateFromTokens(
            promptTokens, negativeTokens, request,
            progress => Console.WriteLine($"  Step {progress.Step}/{progress.TotalSteps} ({progress.ElapsedMs:F0}ms)"));

        string promptSlug = options.Prompt.Length > 30 ? options.Prompt[..30] : options.Prompt;
        string outputPath = outputSettings.GetNextOutputPath(promptSlug);
        ImagePostProcessor.SaveBmp(outputPath, rgbData, outW, outH);

        totalSw.Stop();
        Console.WriteLine();
        Console.WriteLine($"Generation complete in {totalSw.Elapsed.TotalSeconds:F1}s (seed={usedSeed})");
        Console.WriteLine($"Saved: {outputPath}");
        return 0;
    }

    private static IBackend CreateBackend(string backendName) => backendName switch
    {
        "vulkan" => new VulkanBackend(),
        "cuda"   => new CudaBackend(deviceOrdinal: 0, ptxDir: Path.Combine(AppContext.BaseDirectory, "Ptx")),
        "cpu"    => new CpuBackend(),
        _ => throw new ArgumentException($"Unknown backend '{backendName}'. Valid: cpu, vulkan, cuda."),
    };

    private static Dictionary<string, Tensor> CastWeightsToF32(Dictionary<string, Tensor> weights)
    {
        Dictionary<string, Tensor> f32 = new Dictionary<string, Tensor>(weights.Count);
        foreach (KeyValuePair<string, Tensor> kvp in weights)
        {
            f32[kvp.Key] = (kvp.Value.DType == DType.F16 || kvp.Value.DType == DType.BF16)
                ? kvp.Value.CastTo(DType.F32)
                : kvp.Value;
        }
        return f32;
    }

    private static string ResolveRepoRoot()
    {
        string? overridden = Environment.GetEnvironmentVariable("HARTSYINFERENCE_REPO_ROOT");
        if (!string.IsNullOrEmpty(overridden) && Directory.Exists(overridden))
            return Path.GetFullPath(overridden);

        string current = AppContext.BaseDirectory;
        for (int depth = 0; depth < 12; depth++)
        {
            if (File.Exists(Path.Combine(current, "HartsyInference.sln")))
                return current;
            string? parent = Path.GetDirectoryName(current.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar));
            if (string.IsNullOrEmpty(parent) || parent == current) break;
            current = parent;
        }
        return Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", ".."));
    }

    private static void PrintHelp()
    {
        Console.WriteLine("Usage: HartsyInference.Cli [options]");
        Console.WriteLine();
        Console.WriteLine("Options:");
        Console.WriteLine("  --backend cpu|vulkan|cuda   Backend to run on (default: cpu)");
        Console.WriteLine("  --prompt \"...\"              Positive prompt");
        Console.WriteLine("  --negative \"...\"            Negative prompt");
        Console.WriteLine("  --width N                   Image width  (default: 256)");
        Console.WriteLine("  --height N                  Image height (default: 256)");
        Console.WriteLine("  --steps N                   Diffusion steps (default: 20)");
        Console.WriteLine("  --cfg N.N                   CFG scale (default: 7.5)");
        Console.WriteLine("  --seed N                    RNG seed (default: 42)");
        Console.WriteLine("  --models <dir>              Override Models root (default: <repo>/Models)");
        Console.WriteLine("  --output <dir>              Override Output dir   (default: <repo>/Output)");
        Console.WriteLine("  -h, --help                  Show this help");
    }

    private sealed class CliOptions
    {
        public string Backend { get; set; } = "cpu";
        public string Prompt { get; set; } = DefaultPrompt;
        public string NegativePrompt { get; set; } = DefaultNegativePrompt;
        public int Width { get; set; } = DefaultImageWidth;
        public int Height { get; set; } = DefaultImageHeight;
        public int Steps { get; set; } = DefaultSteps;
        public float CfgScale { get; set; } = DefaultCfgScale;
        public int Seed { get; set; } = DefaultSeed;
        public string? ModelsRoot { get; set; }
        public string? OutputDir { get; set; }
        public bool ShowHelp { get; set; }

        public static CliOptions Parse(string[] args)
        {
            CliOptions o = new CliOptions();
            for (int i = 0; i < args.Length; i++)
            {
                string a = args[i];
                switch (a)
                {
                    case "-h" or "--help":  o.ShowHelp = true; break;
                    case "--backend":       o.Backend = NextArg(args, ref i).ToLowerInvariant(); break;
                    case "--prompt":        o.Prompt = NextArg(args, ref i); break;
                    case "--negative":      o.NegativePrompt = NextArg(args, ref i); break;
                    case "--width":         o.Width = int.Parse(NextArg(args, ref i)); break;
                    case "--height":        o.Height = int.Parse(NextArg(args, ref i)); break;
                    case "--steps":         o.Steps = int.Parse(NextArg(args, ref i)); break;
                    case "--cfg":           o.CfgScale = float.Parse(NextArg(args, ref i), System.Globalization.CultureInfo.InvariantCulture); break;
                    case "--seed":          o.Seed = int.Parse(NextArg(args, ref i)); break;
                    case "--models":        o.ModelsRoot = NextArg(args, ref i); break;
                    case "--output":        o.OutputDir = NextArg(args, ref i); break;
                    default:
                        throw new ArgumentException($"Unknown argument: {a}. Use --help for usage.");
                }
            }
            return o;
        }

        private static string NextArg(string[] args, ref int i)
        {
            if (i + 1 >= args.Length)
                throw new ArgumentException($"Missing value after {args[i]}");
            return args[++i];
        }
    }
}
