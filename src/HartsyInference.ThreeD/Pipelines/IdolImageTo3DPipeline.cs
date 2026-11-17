using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Models;
using HartsyInference.ThreeD.Models.Idol;
using HartsyInference.ThreeD.Pipelines.Stages;
using HartsyInference.ThreeD.Preprocessing;

namespace HartsyInference.ThreeD.Pipelines;

/// <summary>The stages of an Idol generation, in order; progress reports carry one so a consumer does not parse stage names.</summary>
public enum IdolStage
{
    /// <summary>Matting, framing and compositing of the picture.</summary>
    Preparing = 1,

    /// <summary>DINOv3 and the FLUX.2 VAE.</summary>
    Encoding = 2,

    /// <summary>The denoiser's steps.</summary>
    Denoising = 3,

    /// <summary>The octree and Gaussian decoders.</summary>
    Decoding = 4,
}

/// <summary>What one Idol generation is asked for.</summary>
public sealed record IdolOptions
{
    /// <summary>Euler steps; 0 takes Idol's 50.</summary>
    public int Steps { get; init; }

    /// <summary>Classifier-free guidance strength <c>w</c>; null takes Idol's 3, 0 runs the conditional pass alone.</summary>
    public float? Guidance { get; init; }

    /// <summary>Time shift of the schedule; Idol's 3.</summary>
    public double RescaleT { get; init; } = IdolSampler.DefaultRescaleT;

    /// <summary>Gaussians to write: 32,768 to 1,048,576, a multiple of 32; 0 takes 262,144.</summary>
    public int Splats { get; init; }

    /// <summary>How tightly the picture is cropped around the object.</summary>
    public IdolFraming Framing { get; init; }

    /// <summary>Seed of the initial noise and of the anchor draws.</summary>
    public int Seed { get; init; }

    /// <summary>Idol-turbo: the denoiser carries the slow and fast phase LoRAs, takes no guidance and one pass per step, and <see cref="Steps"/> 0 means 5.</summary>
    public bool Turbo { get; init; }
}

/// <summary>Idol, image to 3D Gaussian splats (Sperid Labs, Apache-2.0): the picture is prepared (matte, erode, crop), encoded by DINOv3 ViT-H+ and the
/// FLUX.2 VAE, a 370 M-parameter rectified-flow transformer denoises an 8192-token latent (50 steps with guidance), and two decoders place anchor
/// points by octree descent and turn each into 32 Gaussians. Every stage loads its weights, runs and releases them, so the whole pipeline fits a
/// 12 GB card; the weights are read from the model folder for each request. Requests run one at a time.</summary>
public sealed unsafe class IdolImageTo3DPipeline
{
    /// <summary>The files the folder holds, in the layout of the Comfy-Org/Idol repack.</summary>
    public const string DenoiserFile = "diffusion_models/idol_fp16.safetensors";
    public const string DinoFile = "clip_vision/dino_v3_vit_h_bf16.safetensors";
    public const string VaeFile = "vae/flux2-vae.safetensors";
    public const string DecoderFile = "vae/idol_decoder_fp16.safetensors";

    /// <summary>The two phase experts of idol-turbo (Comfy format, akhaliq/idol-turbo <c>loras/</c>), beside the base files.</summary>
    public const string TurboSlowFile = "loras/idol_turbo_slow_comfy.safetensors";
    public const string TurboFastFile = "loras/idol_turbo_fast_comfy.safetensors";

    /// <summary>Gaussians of a default request.</summary>
    public const int DefaultSplats = 262144;
    public const int MinSplats = 32768;
    public const int MaxSplats = 1048576;

    private readonly IBackend _backend;
    private readonly string _directory;
    private readonly IForegroundMatter? _matter;
    private readonly Func<string, Flux2VaeEncoderHandle> _loadVae;
    private readonly DType _blockDType;

    /// <param name="backend">Runs every stage.</param>
    /// <param name="directory">The model folder.</param>
    /// <param name="matter">Mattes an image without alpha; null refuses one.</param>
    /// <param name="loadVae">Opens the FLUX.2 VAE file as an encoder; the pipeline disposes the handle.</param>
    /// <param name="blockDType">What the denoiser's and the decoders' block matrices are held as: null takes F16 (idol-sdk's own precision for its blocks, half the memory and 16-bit GEMMs) or F32.</param>
    public IdolImageTo3DPipeline(IBackend backend, string directory, IForegroundMatter? matter, Func<string, Flux2VaeEncoderHandle> loadVae, DType? blockDType = null)
    {
        DType dtype = blockDType ?? DType.F16;
        if (dtype != DType.F16 && dtype != DType.F32)
            throw new ArgumentOutOfRangeException(nameof(blockDType), dtype, "Idol's blocks are held as F16 or F32.");
        _blockDType = dtype;
        _backend = backend ?? throw new ArgumentNullException(nameof(backend));
        _directory = directory ?? throw new ArgumentNullException(nameof(directory));
        _matter = matter;
        _loadVae = loadVae ?? throw new ArgumentNullException(nameof(loadVae));
    }

    /// <summary>The files a model folder holds.</summary>
    public static IReadOnlyList<string> RequiredFiles { get; } = [DenoiserFile, DinoFile, VaeFile, DecoderFile];

    /// <summary>The files of <paramref name="directory"/> that are missing, empty when it holds a complete model.</summary>
    public static IReadOnlyList<string> MissingFiles(string directory, bool turbo = false)
        => [.. (turbo ? [.. RequiredFiles, TurboSlowFile, TurboFastFile] : RequiredFiles).Where(f => !File.Exists(Path.Combine(directory, f)))];

    /// <summary>Throws when <paramref name="options"/> cannot drive a run.</summary>
    public static void ValidateOptions(IdolOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        if (options.Steps < 0)
            throw new ArgumentOutOfRangeException(nameof(options), options.Steps, "Steps must not be negative.");
        if (options.Guidance is { } g && (g < 0f || !float.IsFinite(g)))
            throw new ArgumentOutOfRangeException(nameof(options), g, "Guidance must be finite and not negative.");
        if (options.Turbo && options.Guidance is > 0f)
            throw new ArgumentOutOfRangeException(nameof(options), options.Guidance, "Idol-turbo takes no guidance: the distilled LoRAs carry it.");
        int splats = options.Splats == 0 ? DefaultSplats : options.Splats;
        if (splats < MinSplats || splats > MaxSplats || splats % 32 != 0)
            throw new ArgumentOutOfRangeException(nameof(options), splats, $"Idol writes {MinSplats} to {MaxSplats} Gaussians, a multiple of 32.");
    }

    /// <summary>Generates the splats of one picture (<paramref name="rgb"/> with an optional <paramref name="alpha"/> matte).</summary>
    /// <param name="report">The stage, its display name, the step and the step count; it may throw to cancel.</param>
    public GaussianSplatCloud Generate(byte[] rgb, byte[]? alpha, int width, int height, IdolOptions options,
        Action<IdolStage, string, int, int>? report = null, Action<string>? log = null, CancellationToken cancel = default)
    {
        ValidateOptions(options);
        IReadOnlyList<string> missing = MissingFiles(_directory, options.Turbo);
        if (missing.Count > 0)
            throw new FileNotFoundException($"'{_directory}' lacks Idol files: {string.Join(", ", missing)}.");
        int steps = options.Steps == 0 ? (options.Turbo ? IdolTurboAdapters.DefaultSteps : IdolSampler.DefaultSteps) : options.Steps;
        float guidance = options.Turbo ? 0f : options.Guidance ?? IdolSampler.DefaultGuidance;
        int splats = options.Splats == 0 ? DefaultSplats : options.Splats;
        Stopwatch clock = Stopwatch.StartNew();
        void Stage(IdolStage stage, string name, int step = 0, int total = 1)
        {
            cancel.ThrowIfCancellationRequested();
            report?.Invoke(stage, name, step, total);
        }

        Stage(IdolStage.Preparing, "Preparing the picture");
        PreparedImage prepared = IdolImageProcessor.Process(_backend, rgb, alpha, width, height, options.Framing, _matter);
        log?.Invoke($"prepared {prepared.Width}x{prepared.Height} ({clock.Elapsed.TotalSeconds:F1} s)");

        Stage(IdolStage.Encoding, "Encoding the picture");
        (Tensor feature1, Tensor feature2) = Condition(prepared.Rgb, log);
        using (feature1)
        using (feature2)
        {
            cancel.ThrowIfCancellationRequested();
            Stage(IdolStage.Denoising, "Denoising", 0, steps);
            Tensor latent = Denoise(feature1, feature2, options.Seed, steps, guidance, options.RescaleT, options.Turbo, (step) => Stage(IdolStage.Denoising, "Denoising", step, steps), log);
            using (latent)
            {
                Stage(IdolStage.Decoding, "Decoding the splats");
                return Decode(latent, splats, options.Seed, log, cancel);
            }
        }
    }

    private (Tensor Feature1, Tensor Feature2) Condition(byte[] prepared, Action<string>? log)
    {
        Stopwatch clock = Stopwatch.StartNew();
        string dinoPath = Path.Combine(_directory, DinoFile);
        Tensor feature1;
        using (Dinov3ImageConditioner dino = Dinov3ImageConditioner.Load(dinoPath, IdolImageProcessor.Size))
        {
            _backend.PreloadWeights(dino.EnumerateWeights());
            try
            {
                feature1 = IdolConditioner.DinoTokens(_backend, dino, prepared);
                _ = feature1.DataPointer;
            }
            finally
            {
                _backend.Sync();
                _backend.FreeWeights(dino.EnumerateWeights());
            }
        }

        log?.Invoke($"DINOv3 tokens {feature1.Shape} ({clock.Elapsed.TotalSeconds:F1} s)");
        try
        {
            using Flux2VaeEncoderHandle vae = _loadVae(Path.Combine(_directory, VaeFile));
            // The encoder's weights stay where the encoder put them; a request releases them with the handle.
            Tensor feature2 = IdolConditioner.VaeTokens(_backend, vae.Encoder, vae.BnMean, vae.BnVar, prepared);
            _ = feature2.DataPointer;
            _backend.Sync();
            log?.Invoke($"VAE tokens {feature2.Shape} ({clock.Elapsed.TotalSeconds:F1} s)");
            return (feature1, feature2);
        }
        catch
        {
            feature1.Dispose();
            throw;
        }
    }

    /// <summary>The camera token's noise stream: a different one from the latent's for every seed, with no int overflow at the top of the seed range.</summary>
    private static int CameraSeed(int seed) => seed ^ 0x2545F491;

    private Tensor Denoise(Tensor feature1, Tensor feature2, int seed, int steps, float guidance, double rescaleT, bool turbo, Action<int> onStep, Action<string>? log)
    {
        Stopwatch clock = Stopwatch.StartNew();
        using IdolWeights weights = IdolWeights.Load(Path.Combine(_directory, DenoiserFile), linearDType: _blockDType);
        using IdolTurboAdapters? adapters = turbo
            ? IdolTurboAdapters.Attach(_backend, weights, Path.Combine(_directory, TurboSlowFile), Path.Combine(_directory, TurboFastFile))
            : null;
        using IdolDenoiser model = new(adapters?.Weights ?? weights);
        IdolConfig config = weights.Config;
        _backend.PreloadWeights(model.EnumerateWeights());
        try
        {
            using Tensor positive = model.EmbedContext(_backend, feature1, feature2);
            // Without guidance there is no blank pass, so the blank image streams are not embedded.
            using Tensor zero1 = new(feature1.Shape, DType.F32), zero2 = new(feature2.Shape, DType.F32);
            using Tensor? blank = adapters is null ? model.EmbedContext(_backend, zero1, zero2) : null;
            using Tensor noise = SeedGenerator.CreateNoise(new TensorShape(1, config.Tokens, config.LatentChannels), seed);
            using Tensor cameraNoise = SeedGenerator.CreateNoise(new TensorShape(1, 1, config.CameraChannels), CameraSeed(seed));
            log?.Invoke($"denoiser loaded, image streams embedded ({clock.Elapsed.TotalSeconds:F1} s)");
            (Tensor latent, Tensor camera) = IdolSampler.Sample(_backend, model, positive, blank ?? positive, noise, cameraNoise, steps, guidance, rescaleT, onStep, adapters is null ? null : adapters.Phase);
            camera.Dispose();
            _ = latent.DataPointer;
            log?.Invoke($"{steps} steps ({clock.Elapsed.TotalSeconds:F1} s)");
            return latent;
        }
        finally
        {
            _backend.Sync();
            _backend.FreeWeights(model.EnumerateWeights());
        }
    }

    private GaussianSplatCloud Decode(Tensor latent, int splats, int seed, Action<string>? log, CancellationToken cancel)
    {
        Stopwatch clock = Stopwatch.StartNew();
        string path = Path.Combine(_directory, DecoderFile);
        using IdolDecoderWeights weights = IdolDecoderWeights.Load(path, linearDType: _blockDType);
        IdolDecoderConfig config = weights.Config;
        int anchors = splats / config.GaussiansPerAnchor;
        float[] points;
        using (IdolOctreeDecoder octree = new(weights))
        {
            points = IdolOctreeSampler.Sample((centres, resolution) =>
            {
                cancel.ThrowIfCancellationRequested();
                return octree.Logits(_backend, centres, resolution, latent);
            }, anchors, config.MaxVoxelLevel, new Random(seed));
        }

        log?.Invoke($"{anchors} anchors placed ({clock.Elapsed.TotalSeconds:F1} s)");
        using IdolGaussianDecoder gaussian = new(weights);
        using Tensor features = gaussian.Features(_backend, points, latent);
        _ = features.DataPointer;
        _backend.Sync();
        GaussianSplatCloud cloud = gaussian.ToCloud(points, features);
        log?.Invoke($"{cloud.Count} splats ({clock.Elapsed.TotalSeconds:F1} s)");
        return cloud;
    }
}
