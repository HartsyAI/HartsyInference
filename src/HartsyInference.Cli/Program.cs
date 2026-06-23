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
using HartsyInference.LLM.ChatTemplates;
using HartsyInference.LLM.Generation;
using HartsyInference.LLM.Sampling;
using HartsyInference.LLM.Transformer;
using HartsyInference.ModelHandler.SafeTensors;
using HartsyInference.Tokenizers;
using HartsyInference.Vulkan;

namespace HartsyInference.Cli;

/// <summary>
/// HartsyInference unified CLI dispatcher supporting multiple modalities:
/// image, text, music, vision, video, 3d, interactive.
/// </summary>
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

        return options.Task switch
        {
            "image" => GenerateImage(options, modelsRoot, outputDir),
            "text" => GenerateText(options, modelsRoot, outputDir),
            "music" => GenerateMusic(options, modelsRoot, outputDir),
            "vision" => RunVision(options, modelsRoot, outputDir),
            "video" => GenerateVideo(options, modelsRoot, outputDir),
            "3d" => Generate3d(options, modelsRoot, outputDir),
            "interactive" => RunInteractive(options, modelsRoot, outputDir),
            _ => throw new ArgumentException($"Unknown task '{options.Task}'. Valid: image, text, music, vision, video, 3d, interactive."),
        };
    }

    // ──── Image Generation (SD1.5, SDXL, Flux, SD3, etc.) ────
    private static int GenerateImage(CliOptions options, string modelsRoot, string outputDir)
    {
        string modelLayout = options.ResolveModel(options.ImageModel, "StabilityAI/sd-v1-5")!;
        ModelPaths modelPaths = ModelPaths.FromComfyLayout(modelsRoot, modelLayout);
        OutputSettings outputSettings = new OutputSettings { OutputDir = outputDir };

        Console.WriteLine($"HartsyInference CLI - Image generation ({modelLayout})");
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

    // ──── Text Generation (LLM) ────
    private static int GenerateText(CliOptions options, string modelsRoot, string outputDir)
    {
        string arch = (options.ResolveModel(options.TextModel, "qwen2") ?? "qwen2").ToLowerInvariant();
        GenericTransformer transformer;
        GgufLanguageModel? ggufModel = null;
        SafeTensorsShardLoader? safeLoader = null;
        ILlmTokenizer tokenizer;
        IChatTemplate? template = null;

        using IBackend backend = CreateBackend(options.Backend);
        CudaBackend? cuda = backend as CudaBackend;

        if (arch == "gguf")
        {
            string ggufPath = options.ResolveModelPath(options.TextModelPath) ?? throw new InvalidOperationException("Specify --model-path (or --text-model-path) for GGUF arch.");
            if (!File.Exists(ggufPath))
            {
                Console.Error.WriteLine($"GGUF file not found: {ggufPath}");
                return 1;
            }
            Console.WriteLine($"Loading GGUF {ggufPath} ...");
            bool lowVram = Environment.GetEnvironmentVariable("HARTSY_LOWVRAM") == "1";
            ggufModel = GgufLanguageModel.Load(ggufPath, lowVram);
            transformer = ggufModel.Transformer;
            TransformerConfig c = ggufModel.Config;
            Console.WriteLine($"  arch={ggufModel.Architecture} hidden={c.HiddenSize} layers={c.NumLayers}");
            tokenizer = ggufModel.Tokenizer;
            template = ggufModel.Template;
        }
        else
        {
            string modelDir = options.ResolveModelPath(options.TextModelPath) ?? Path.Combine(modelsRoot, "LLM", arch);
            if (!Directory.Exists(modelDir) || Directory.GetFiles(modelDir, "*.safetensors").Length == 0)
            {
                Console.Error.WriteLine($"No safetensors weights found under {modelDir}.");
                return 1;
            }
            Console.WriteLine($"Loading {modelDir} ...");
            safeLoader = new SafeTensorsShardLoader();
            safeLoader.LoadDirectory(modelDir);
            Dictionary<string, Tensor> weights = safeLoader.GetAllTensors();

            // Checkpoints ship bf16/f16 weights; the CUDA backend dequantizes on-device, but the CPU backend is
            // F32-only and its kernels would read past the half-width buffers. Cast up front for cpu.
            if (cuda is null)
            {
                foreach (string key in weights.Keys.ToList())
                    if (weights[key].DType != DType.F32)
                        weights[key] = weights[key].CastTo(DType.F32);
            }

            TransformerConfig cfg = arch switch
            {
                "qwen2" => TransformerConfig.Qwen2_5_0_5B,
                "qwen3" => TransformerConfig.Qwen3_0_6B,
                _ => throw new ArgumentException($"Unknown LLM arch: {arch}"),
            };

            GenericTransformer built = new GenericTransformer(cfg);
            built.LoadWeights(weights, "model");
            transformer = built;

            string vocabPath = Path.Combine(modelDir, "vocab.json");
            tokenizer = File.Exists(vocabPath) ? new Qwen2Tokenizer(vocabPath, Path.Combine(modelDir, "merges.txt")) : new Qwen2Tokenizer();
        }

        if (cuda is not null)
        {
            Console.WriteLine("Preloading weights to GPU...");
            backend.PreloadWeights(transformer.EnumerateWeights());
        }

        TextGenerationPipeline pipeline = new TextGenerationPipeline(transformer, tokenizer, backend, template);
        GenerationRequest request = new GenerationRequest
        {
            Prompt = options.Prompt,
            MaxTokens = options.TextMaxTokens,
            Sampling = SamplingOptions.Default,
        };

        Stopwatch sw = Stopwatch.StartNew();
        GenerationResult result = pipeline.Generate(request);
        backend.Sync();
        sw.Stop();

        double sec = sw.Elapsed.TotalSeconds;
        int n = Math.Max(1, result.TokenIds.Count);
        Console.WriteLine();
        Console.WriteLine($"--- {arch} / {options.Backend} ---");
        Console.WriteLine($"  prompt tokens : {result.PromptTokens}");
        Console.WriteLine($"  generated     : {result.TokenIds.Count} tokens in {sec:F2}s ({n / sec:F2} tok/s)");
        Console.WriteLine();
        Console.WriteLine("=== text ===");
        Console.WriteLine(result.Text);
        return 0;
    }

    // ──── Music Generation (placeholder) ────
    private static int GenerateMusic(CliOptions options, string modelsRoot, string outputDir)
    {
        Console.WriteLine($"HartsyInference CLI - Music generation ({options.ResolveModel(options.MusicModel, "ace-step")})");
        Console.WriteLine($"  Backend:  {options.Backend}");
        Console.WriteLine($"  Prompt:   \"{options.Prompt}\"");
        Console.WriteLine($"  Duration: {options.MusicDuration}s");
        Console.WriteLine($"  Seed:     {options.Seed}");
        Console.WriteLine();
        Console.WriteLine("Music generation CLI coming soon! See samples/HartsyInference.Music.Cli");
        Console.WriteLine("Supported models: ace-step, musicgen, yue");
        return 0;
    }

    // ──── Vision (embeddings, detection, segmentation) ────
    private static int RunVision(CliOptions options, string modelsRoot, string outputDir)
    {
        Console.WriteLine($"HartsyInference CLI - Vision ({options.VisionMode ?? "embed"})");
        Console.WriteLine($"  Mode:    {options.VisionMode ?? "embed"}");
        Console.WriteLine($"  Model:   {options.ResolveModel(options.VisionModel, "clip-vit-l-14")}");
        Console.WriteLine($"  Backend: {options.Backend}");
        Console.WriteLine();
        Console.WriteLine("Vision CLI coming soon! See samples/HartsyInference.Vision.Cli");
        Console.WriteLine("Modes: embed (CLIP/SigLIP), detect (YOLO), segment (SAM), face");
        return 0;
    }

    // ──── Video Generation (placeholder) ────
    private static int GenerateVideo(CliOptions options, string modelsRoot, string outputDir)
    {
        Console.WriteLine($"HartsyInference CLI - Video generation ({options.ResolveModel(options.VideoModel, "ltx-video")})");
        Console.WriteLine($"  Backend:  {options.Backend}");
        Console.WriteLine($"  Prompt:   \"{options.Prompt}\"");
        Console.WriteLine($"  Duration: {options.VideoDuration}s");
        Console.WriteLine($"  Seed:     {options.Seed}");
        Console.WriteLine();
        Console.WriteLine("Video generation CLI coming soon! See samples/HartsyInference.Video.Cli");
        Console.WriteLine("Supported models: ltx-video, wan, lance, kandinsky");
        return 0;
    }

    // ──── 3D Generation (placeholder) ────
    private static int Generate3d(CliOptions options, string modelsRoot, string outputDir)
    {
        Console.WriteLine($"HartsyInference CLI - 3D generation ({options.ResolveModel(options.ThreeDModel, "triposr")})");
        Console.WriteLine($"  Backend: {options.Backend}");
        Console.WriteLine("  See samples/HartsyInference.ThreeD.Cli for full 3D support");
        return 0;
    }

    // ──── Interactive / World Models (placeholder) ────
    private static int RunInteractive(CliOptions options, string modelsRoot, string outputDir)
    {
        Console.WriteLine($"HartsyInference CLI - Interactive world models ({options.ResolveModel(options.InteractiveModel, "matrix-game-2")})");
        Console.WriteLine($"  Backend: {options.Backend}");
        Console.WriteLine("  Supported models: matrix-game-2, matrix-game-3, hunyuan-gamecraft, oasis");
        Console.WriteLine();
        Console.WriteLine("Interactive CLI coming soon!");
        return 0;
    }

    // ──── Shared utilities ────
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
        Console.WriteLine("HartsyInference CLI — unified dispatcher for AI inference");
        Console.WriteLine();
        Console.WriteLine("Usage: HartsyInference.Cli --task <task> [options]");
        Console.WriteLine();
        Console.WriteLine("Tasks (modalities):");
        Console.WriteLine("  --task image       Image generation (SD1.5, SDXL, Flux, SD3, etc.) [default]");
        Console.WriteLine("  --task text        LLM text generation (Qwen2, Qwen3, Llama, GGUF)");
        Console.WriteLine("  --task music       Music generation (ACE-Step, MusicGen, YuE)");
        Console.WriteLine("  --task vision      Vision embeddings, detection, segmentation, face");
        Console.WriteLine("  --task video       Video generation (LTX-Video, Wan, Lance, Kandinsky)");
        Console.WriteLine("  --task 3d          3D mesh generation (TripoSR, Hunyuan3D)");
        Console.WriteLine("  --task interactive Interactive world models (Matrix-Game, Hunyuan-GameCraft, Oasis)");
        Console.WriteLine();
        Console.WriteLine("Global options:");
        Console.WriteLine("  --model <name>              Model to use for ANY task (unified selector; overrides the");
        Console.WriteLine("                             per-task --*-model flags below)");
        Console.WriteLine("  --model-path <path>         Path to a model checkpoint/dir (any task)");
        Console.WriteLine("  --backend cpu|cuda|vulkan   Backend (default: cpu)");
        Console.WriteLine("  --models <dir>              Override Models root");
        Console.WriteLine("  --output <dir>              Override Output dir");
        Console.WriteLine("  -h, --help                  Show this help");
        Console.WriteLine();
        Console.WriteLine("Image task options:");
        Console.WriteLine("  --image-model <model>       Model (default: sd-v1-5)");
        Console.WriteLine("  --prompt \"...\"              Positive prompt");
        Console.WriteLine("  --negative \"...\"            Negative prompt");
        Console.WriteLine("  --width N                   Image width (default: 256)");
        Console.WriteLine("  --height N                  Image height (default: 256)");
        Console.WriteLine("  --steps N                   Diffusion steps (default: 20)");
        Console.WriteLine("  --cfg N.N                   CFG scale (default: 7.5)");
        Console.WriteLine("  --seed N                    RNG seed (default: 42)");
        Console.WriteLine();
        Console.WriteLine("Text task options:");
        Console.WriteLine("  --text-model <arch>         LLM architecture: qwen2, qwen3, gguf (default: qwen2)");
        Console.WriteLine("  --text-model-path <path>    Path to model checkpoint");
        Console.WriteLine("  --prompt \"...\"              Text prompt");
        Console.WriteLine("  --text-max-tokens N         Max tokens to generate (default: 64)");
        Console.WriteLine();
        Console.WriteLine("Music task options:");
        Console.WriteLine("  --music-model <model>       Model: ace-step, musicgen, yue (default: ace-step)");
        Console.WriteLine("  --prompt \"...\"              Music description");
        Console.WriteLine("  --music-duration N          Duration in seconds (default: 30)");
        Console.WriteLine("  --seed N                    RNG seed");
        Console.WriteLine();
        Console.WriteLine("Vision task options:");
        Console.WriteLine("  --vision-mode <mode>        Mode: embed, detect, segment, face (default: embed)");
        Console.WriteLine("  --vision-model <model>      Model (default: clip-vit-l-14)");
        Console.WriteLine();
        Console.WriteLine("Video task options:");
        Console.WriteLine("  --video-model <model>       Model: ltx-video, wan, lance, kandinsky (default: ltx-video)");
        Console.WriteLine("  --prompt \"...\"              Video description");
        Console.WriteLine("  --video-duration N          Duration in seconds (default: 8)");
        Console.WriteLine();
        Console.WriteLine("3D task options:");
        Console.WriteLine("  --3d-model <model>          Model: triposr, hunyuan3d (default: triposr)");
        Console.WriteLine();
        Console.WriteLine("Interactive task options:");
        Console.WriteLine("  --interactive-model <model> Model: matrix-game-2, matrix-game-3, hunyuan-gamecraft, oasis");
        Console.WriteLine();
        Console.WriteLine("Note: every per-task --*-model flag above is an alias; --model selects the model for");
        Console.WriteLine("      whichever task you run, and --model-path points at its checkpoint.");
        Console.WriteLine();
        Console.WriteLine("Examples:");
        Console.WriteLine("  HartsyInference.Cli --task image --model StabilityAI/sd-v1-5 --prompt \"cat\"");
        Console.WriteLine("  HartsyInference.Cli --task text  --model qwen3 --model-path /models/qwen3 --prompt \"What is AI?\"");
        Console.WriteLine("  HartsyInference.Cli --task music --prompt \"upbeat electronic\" --music-duration 30");
    }

    private sealed class CliOptions
    {
        // Global
        public string Task { get; set; } = "image";
        public string Backend { get; set; } = "cpu";
        public string? ModelsRoot { get; set; }
        public string? OutputDir { get; set; }
        public bool ShowHelp { get; set; }

        // Unified model selection — works for every task. The per-task flags below (--image-model,
        // --text-model, …) remain as aliases; when both are given, --model wins.
        public string? Model { get; set; }
        public string? ModelPath { get; set; }

        /// <summary>The effective model name for the current task: the unified <c>--model</c> if set, else the
        /// task-specific flag, else <paramref name="fallback"/>.</summary>
        public string? ResolveModel(string? taskSpecific, string? fallback = null) => Model ?? taskSpecific ?? fallback;

        /// <summary>The effective checkpoint path: the unified <c>--model-path</c> if set, else the task-specific one.</summary>
        public string? ResolveModelPath(string? taskSpecific) => ModelPath ?? taskSpecific;

        // Image
        public string? ImageModel { get; set; }
        public string Prompt { get; set; } = DefaultPrompt;
        public string NegativePrompt { get; set; } = DefaultNegativePrompt;
        public int Width { get; set; } = DefaultImageWidth;
        public int Height { get; set; } = DefaultImageHeight;
        public int Steps { get; set; } = DefaultSteps;
        public float CfgScale { get; set; } = DefaultCfgScale;
        public int Seed { get; set; } = DefaultSeed;

        // Text
        public string? TextModel { get; set; }
        public string? TextModelPath { get; set; }
        public int TextMaxTokens { get; set; } = 64;

        // Music
        public string? MusicModel { get; set; }
        public int MusicDuration { get; set; } = 30;

        // Vision
        public string? VisionMode { get; set; }
        public string? VisionModel { get; set; }

        // Video
        public string? VideoModel { get; set; }
        public int VideoDuration { get; set; } = 8;

        // 3D
        public string? ThreeDModel { get; set; }

        // Interactive
        public string? InteractiveModel { get; set; }

        public static CliOptions Parse(string[] args)
        {
            CliOptions o = new CliOptions();
            for (int i = 0; i < args.Length; i++)
            {
                string a = args[i];
                switch (a)
                {
                    case "-h" or "--help":
                        o.ShowHelp = true;
                        break;
                    case "--task":
                        o.Task = NextArg(args, ref i).ToLowerInvariant();
                        break;
                    case "--backend":
                        o.Backend = NextArg(args, ref i).ToLowerInvariant();
                        break;
                    case "--models":
                        o.ModelsRoot = NextArg(args, ref i);
                        break;
                    case "--output":
                        o.OutputDir = NextArg(args, ref i);
                        break;
                    case "--model":
                        o.Model = NextArg(args, ref i);
                        break;
                    case "--model-path":
                        o.ModelPath = NextArg(args, ref i);
                        break;
                    case "--image-model":
                        o.ImageModel = NextArg(args, ref i);
                        break;
                    case "--prompt":
                        o.Prompt = NextArg(args, ref i);
                        break;
                    case "--negative":
                        o.NegativePrompt = NextArg(args, ref i);
                        break;
                    case "--width":
                        o.Width = int.Parse(NextArg(args, ref i));
                        break;
                    case "--height":
                        o.Height = int.Parse(NextArg(args, ref i));
                        break;
                    case "--steps":
                        o.Steps = int.Parse(NextArg(args, ref i));
                        break;
                    case "--cfg":
                        o.CfgScale = float.Parse(NextArg(args, ref i), System.Globalization.CultureInfo.InvariantCulture);
                        break;
                    case "--seed":
                        o.Seed = int.Parse(NextArg(args, ref i));
                        break;
                    case "--text-model":
                        o.TextModel = NextArg(args, ref i).ToLowerInvariant();
                        break;
                    case "--text-model-path":
                        o.TextModelPath = NextArg(args, ref i);
                        break;
                    case "--text-max-tokens":
                        o.TextMaxTokens = int.Parse(NextArg(args, ref i));
                        break;
                    case "--music-model":
                        o.MusicModel = NextArg(args, ref i).ToLowerInvariant();
                        break;
                    case "--music-duration":
                        o.MusicDuration = int.Parse(NextArg(args, ref i));
                        break;
                    case "--vision-mode":
                        o.VisionMode = NextArg(args, ref i).ToLowerInvariant();
                        break;
                    case "--vision-model":
                        o.VisionModel = NextArg(args, ref i).ToLowerInvariant();
                        break;
                    case "--video-model":
                        o.VideoModel = NextArg(args, ref i).ToLowerInvariant();
                        break;
                    case "--video-duration":
                        o.VideoDuration = int.Parse(NextArg(args, ref i));
                        break;
                    case "--3d-model":
                        o.ThreeDModel = NextArg(args, ref i).ToLowerInvariant();
                        break;
                    case "--interactive-model":
                        o.InteractiveModel = NextArg(args, ref i).ToLowerInvariant();
                        break;
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
