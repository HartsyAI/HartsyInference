using System.Runtime.CompilerServices;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Video.Encoding;
using HartsyInference.Video.Pipelines;

namespace HartsyInference.Engine.Services;

/// <summary>SeedVR2 restoration service: decodes the input (ffmpeg subprocess for containers, passthrough
/// for frames/stills), constructs and caches the pipeline per checkpoint, and streams restored frames.
/// The three weight files (DiT / VAE / frozen embeddings) resolve from catalog asset roles when the spec
/// carries a catalog entry, else from safetensors siblings of the explicit --model-path file.</summary>
public sealed class RestoreService : IRestoreService
{
    private readonly InferenceEngine _engine;
    private readonly object _lock = new();
    private string? _loadedPath;
    private SeedVr2RestorePipeline? _pipeline;
    private SeedVr2Dit? _dit;
    private readonly List<SafeTensorsLoader> _loaders = new();
    private Tensor? _posEmb;

    /// <summary>Creates the service bound to its owning engine.</summary>
    internal RestoreService(InferenceEngine engine) => _engine = engine;

    /// <inheritdoc/>
    public async IAsyncEnumerable<VideoFrame> RestoreAsync(ModelSpec spec, RestoreRequest request,
        IProgress<StepPreview>? progress = null, [EnumeratorCancellation] CancellationToken cancel = default)
    {
        ArgumentNullException.ThrowIfNull(request);
        int inputs = (request.Video is null ? 0 : 1) + (request.Frames is null ? 0 : 1) + (request.Image is null ? 0 : 1);
        if (inputs != 1)
            throw new ArgumentException("Exactly one of Video, Frames, or Image must be set.", nameof(request));

        (List<byte[]> frames, int width, int height) = await DecodeInputAsync(request, cancel).ConfigureAwait(false);

        SeedVr2RestoreOptions options = new()
        {
            TargetArea = (long)(request.TargetWidth ?? 1280) * (request.TargetHeight ?? 720),
            ClipFrames = request.ClipFrames ?? 25,
            Overlap = request.Overlap ?? 4,
            Strength = request.Strength ?? 1.0f,
            Seed = request.Seed ?? 666,
        };

        (List<byte[]> restored, int outW, int outH) = await Task.Run(() =>
        {
            SeedVr2RestorePipeline pipeline = GetOrLoadPipeline(spec);
            return pipeline.Restore(frames, width, height, options, p => progress?.Report(new StepPreview
            {
                Step = p.Step,
                TotalSteps = p.TotalSteps,
            }));
        }, cancel).ConfigureAwait(false);

        for (int i = 0; i < restored.Count; i++)
        {
            cancel.ThrowIfCancellationRequested();
            yield return new VideoFrame { Rgb = restored[i], Width = outW, Height = outH, Index = i };
        }
    }

    private async Task<(List<byte[]> Frames, int Width, int Height)> DecodeInputAsync(
        RestoreRequest request, CancellationToken cancel)
    {
        if (request.Video is not null)
        {
            FfmpegProcessDecoder decoder = new();
            FfmpegProcessDecoder.Result decoded =
                await decoder.DecodeAsync(request.Video.Data, request.Video.Format, cancel).ConfigureAwait(false);
            return (decoded.Frames, decoded.Width, decoded.Height);
        }
        if (request.Image is not null)
            return ([request.Image.Rgb], request.Image.Width, request.Image.Height);

        IReadOnlyList<ImageData> frames = request.Frames!;
        if (frames.Count == 0)
            throw new ArgumentException("Frames must be non-empty.", nameof(request));
        return ([.. frames.Select(f => f.Rgb)], frames[0].Width, frames[0].Height);
    }

    private SeedVr2RestorePipeline GetOrLoadPipeline(ModelSpec spec)
    {
        string ditPath = spec.LocalPath
            ?? throw new InvalidOperationException("SeedVR2 checkpoint path did not resolve.");
        lock (_lock)
        {
            if (_pipeline is not null && _loadedPath == ditPath)
                return _pipeline;
            ReleasePipeline();
            // A resident video DiT and SeedVR2's VAE peak cannot share 24 GB.
            _engine.FreeMemory();

            (string vaePath, string embPath) = ResolveSideAssets(spec, ditPath);
            Logs.Info($"Loading SeedVR2: dit={ditPath} vae={vaePath} emb={embPath}");

            (Dictionary<string, Tensor> ditWeights, SafeTensorsLoader ditLoader) =
                SeedVr2CheckpointConverter.LoadAndConvert(ditPath);
            _loaders.Add(ditLoader);
            SafeTensorsLoader vaeLoader = new();
            vaeLoader.Load(vaePath);
            _loaders.Add(vaeLoader);
            SafeTensorsLoader embLoader = new();
            embLoader.Load(embPath);
            _loaders.Add(embLoader);

            SeedVr2Config config = SeedVr2Config.Detect(ditWeights);
            SeedVr2Dit dit = new(config);
            dit.LoadWeights(ditWeights);
            _dit = dit;
            Dictionary<string, Tensor> vaeWeights = vaeLoader.GetAllTensors();
            // BF16 VAE activations on CUDA (reference precision): halves the fp32 activation peak that
            // OOMs 24 GB at 720p-area. HARTSY_SEEDVR2_VAE_F32=1 is the kill-switch back to full precision.
            bool vaeBf16 = _engine.Backend.Device.IsCuda
                && Environment.GetEnvironmentVariable("HARTSY_SEEDVR2_VAE_F32") != "1";
            SeedVr2VaeConfig vaeConfig = SeedVr2VaeConfig.Default with
            {
                ActivationDType = vaeBf16 ? DType.BF16 : DType.F32,
            };
            Logs.Info($"SeedVR2 VAE activation dtype: {vaeConfig.ActivationDType}");
            SeedVr2VaeEncoder encoder = new(vaeConfig);
            encoder.LoadWeights(vaeWeights);
            SeedVr2VaeDecoder decoder = new(vaeConfig);
            decoder.LoadWeights(vaeWeights);
            _posEmb = embLoader.GetTensor("pos_emb").CastTo(DType.F32);

            _pipeline = new SeedVr2RestorePipeline(_engine.Backend, dit, encoder, decoder, _posEmb);
            _loadedPath = ditPath;
            return _pipeline;
        }
    }

    private static (string Vae, string Emb) ResolveSideAssets(ModelSpec spec, string ditPath)
    {
        if (spec.Catalog is not null)
        {
            ModelAsset? vae = spec.Catalog.Assets.FirstOrDefault(a => a.Role == "vae");
            ModelAsset? emb = spec.Catalog.Assets.FirstOrDefault(a => a.Role == "embeddings");
            if (vae is not null && emb is not null)
                return (ModelDownloader.TargetPath(vae), ModelDownloader.TargetPath(emb));
        }
        string dir = Path.GetDirectoryName(ditPath)!;
        // Ordered: EnumerateFiles order is filesystem-dependent.
        string? vaeSibling = Directory.EnumerateFiles(dir, "*.safetensors").Order()
            .FirstOrDefault(f => Path.GetFileName(f).Contains("vae", StringComparison.OrdinalIgnoreCase));
        string? embSibling = Directory.EnumerateFiles(dir, "*.safetensors").Order()
            .FirstOrDefault(f => Path.GetFileName(f).Contains("emb", StringComparison.OrdinalIgnoreCase));
        if (vaeSibling is null || embSibling is null)
            throw new InvalidOperationException(
                $"SeedVR2 needs VAE and embeddings safetensors beside the DiT checkpoint in '{dir}' " +
                "(names containing 'vae' and 'emb'), or a catalog entry with vae/embeddings asset roles.");
        return (vaeSibling, embSibling);
    }

    /// <summary>Releases the cached pipeline and its mmap-backed weights (called from ReleaseLoaded).</summary>
    internal void ReleasePipeline()
    {
        lock (_lock)
        {
            _pipeline?.Dispose();
            _pipeline = null;
            _dit?.Dispose();
            _dit = null;
            _posEmb?.Dispose();
            _posEmb = null;
            foreach (SafeTensorsLoader loader in _loaders)
                loader.Dispose();
            _loaders.Clear();
            _loadedPath = null;
        }
    }
}
