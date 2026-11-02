using System.Runtime.CompilerServices;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Core.MemoryManagement;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Requests;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.PyTorch;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Video.Encoding;
using HartsyInference.Video.Pipelines;

namespace HartsyInference.Engine.Services;

/// <summary>SeedVR2 restoration service: decodes the input (ffmpeg subprocess for containers, passthrough for frames/stills), constructs and caches the pipeline per checkpoint, and streams restored frames. The three weight files (DiT / VAE / frozen embeddings) resolve from catalog asset roles when the spec carries a catalog entry, else from safetensors siblings of the explicit --model-path file.</summary>
public sealed class RestoreService : IRestoreService
{
    private readonly InferenceEngine _engine;
    private readonly object _lock = new();
    private string? _loadedPath;
    private DType? _loadedPacking;
    private IBackend? _loadedBackend;
    // Backends for restores placed on another device, created on first use and kept for the process: a device context is expensive to rebuild.
    private readonly Dictionary<string, IBackend> _deviceBackends = new(StringComparer.OrdinalIgnoreCase);
    private SeedVr2RestorePipeline? _pipeline;
    private SeedVr2Dit? _dit;
    private readonly List<SafeTensorsLoader> _loaders = new();
    // F32 copies of half-precision checkpoint tensors, made only for the CPU backend (see GetOrLoadPipeline).
    private readonly List<Tensor> _castCopies = new();
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

        // The DiT streams its blocks through the shared scope and packs to Q8_0 on the policy's QuantizedCompute lever, so both
        // levers are honoured here. Any other lever a posture asks for is still reported, so a tier never quietly does less than
        // it says (`video --vram-mode ... --restore` carries the video phase's override into this one).
        IBackend backend = BackendFor(request);
        MemorySupportReport.ReportService(
            "seedvr2-restore", backend, VramPolicyRegistry.Resolve(backend, request.Vram),
            MemoryCapabilities.BlockStreaming | MemoryCapabilities.QuantizedCompute);

        (List<byte[]> frames, int width, int height) = await DecodeInputAsync(request, cancel).ConfigureAwait(false);

        SeedVr2RestoreOptions options = new()
        {
            TargetArea = (long)(request.TargetWidth ?? 1280) * (request.TargetHeight ?? 720),
            ClipFrames = request.ClipFrames ?? 25,
            Overlap = request.Overlap ?? 4,
            Strength = request.Strength ?? 1.0f,
            ColorTransfer = request.ColorTransfer ?? SeedVr2ColorTransfer.None,
            Seed = request.Seed ?? 666,
        };

        (List<byte[]> restored, int outW, int outH) = await Task.Run(() =>
        {
            FreeEngineBeforeLoad(spec, backend);
            // Under the device gate like every other job, so an engine-wide release waits for the restore instead of
            // emptying the device under it. Taken after the free above, which takes the gate itself.
            using IDisposable gate = DeviceGate.AcquireAll([backend], cancel);
            SeedVr2RestorePipeline pipeline = GetOrLoadPipeline(spec, request, backend);
            if (progress is null)
            {
                return pipeline.Restore(frames, width, height, options);
            }
            return pipeline.RestoreWithPreviews(frames, width, height, options, (p, previewFrames) =>
            {
                byte[][] rgbFrames = [.. previewFrames.Select(frame => frame.Rgb)];
                progress.Report(new StepPreview
                {
                    Step = p.Step,
                    TotalSteps = p.TotalSteps,
                    PreviewRgb = rgbFrames.Length > 0 ? rgbFrames[rgbFrames.Length / 2] : null,
                    PreviewFramesRgb = rgbFrames.Length > 1 ? rgbFrames : null,
                    PreviewWidth = previewFrames.Count > 0 ? previewFrames[0].Width : 0,
                    PreviewHeight = previewFrames.Count > 0 ? previewFrames[0].Height : 0,
                });
            });
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

    /// <summary>Frees the engine's other models when <paramref name="spec"/> is not the loaded restore model: a resident
    /// video DiT and SeedVR2's VAE peak cannot share 24 GB.</summary>
    /// <remarks>Outside the device gate and outside <c>_lock</c>: the free takes the gate, and it releases this
    /// service's pipeline under <c>_lock</c>.</remarks>
    private void FreeEngineBeforeLoad(ModelSpec spec, IBackend backend)
    {
        bool loaded;
        lock (_lock)
        {
            loaded = _pipeline is not null && _loadedPath == spec.LocalPath && ReferenceEquals(_loadedBackend, backend);
        }
        // A restore on another device leaves the engine's own card alone: its resident model is not competing for that memory.
        if (!loaded && ReferenceEquals(backend, _engine.Backend))
        {
            _engine.FreeMemory();
        }
    }

    /// <summary>The backend this restore runs on: the engine's own, or the device the request names (created once, then reused).</summary>
    private IBackend BackendFor(RestoreRequest request)
    {
        string? selector = request.Device?.Trim();
        if (string.IsNullOrEmpty(selector) || selector.Equals(_engine.Backend.Device.ToString(), StringComparison.OrdinalIgnoreCase))
            return _engine.Backend;
        lock (_lock)
        {
            if (_deviceBackends.TryGetValue(selector, out IBackend? existing))
                return existing;
            IBackend created = BackendFactory.Create(selector);
            // The restore follows the posture its request asked for, the way a restore on the engine's own card does.
            VramPolicyRegistry.Set(created, VramPolicyRegistry.Resolve(_engine.Backend, request.Vram));
            Logs.Info($"SeedVR2 restore placed on {created.Device} (request Device '{selector}').");
            _deviceBackends[selector] = created;
            return created;
        }
    }

    private SeedVr2RestorePipeline GetOrLoadPipeline(ModelSpec spec, RestoreRequest request, IBackend backend)
    {
        string ditPath = spec.LocalPath
            ?? throw new FileNotFoundException(
                $"No checkpoint found for model '{spec.Requested}'. Supply its local path, or select a SeedVR2 "
                + "catalog id so its assets download first.");
        // The packing is decided from the checkpoint's size and the free memory now, before any weight is read, so a
        // cached pipeline is only reused when it was packed the same way.
        DType? packing = ChoosePacking(request, backend);
        lock (_lock)
        {
            if (_pipeline is not null && _loadedPath == ditPath && _loadedPacking == packing && ReferenceEquals(_loadedBackend, backend))
                return _pipeline;
            ReleasePipeline();

            (string vaePath, string embPath) = ResolveSideAssets(spec, ditPath);
            Logs.Info($"Loading SeedVR2: dit={ditPath} vae={vaePath} emb={embPath}");

            (Dictionary<string, Tensor> ditWeights, SafeTensorsLoader ditLoader) =
                SeedVr2CheckpointConverter.LoadAndConvert(ditPath);
            _loaders.Add(ditLoader);
            SafeTensorsLoader vaeLoader = new();
            vaeLoader.Load(vaePath);
            _loaders.Add(vaeLoader);

            Dictionary<string, Tensor> vaeWeights = vaeLoader.GetAllTensors();
            // The CPU backend's kernels are F32-only: its Linear casts a half-precision weight on the fly but not the
            // bias, and the VAE's 3-D convs do not cast at all, so the fp16 checkpoints stop at the first GEMM. On CPU
            // the whole set becomes F32 once at load instead (about 13.5 GB for the 3B DiT); CUDA keeps the half
            // weights its kernels consume directly.
            if (backend.Device.IsCpu)
            {
                ditWeights = CastAllToF32(ditWeights, _castCopies);
                vaeWeights = CastAllToF32(vaeWeights, _castCopies);
                Logs.Info("SeedVR2 on the CPU backend: checkpoint tensors cast to F32 at load.");
            }
            if (packing is { } target)
            {
                SeedVr2WeightPacking.PackLinears(ditWeights, target, _castCopies);
                Logs.Info($"SeedVR2 DiT packed to {target.Name} (VRAM policy).");
            }
            SeedVr2Config config = SeedVr2Config.Detect(ditWeights);
            SeedVr2Dit dit = new(config);
            dit.LoadWeights(ditWeights);
            _dit = dit;
            // BF16 VAE activations on CUDA (reference precision): halves the fp32 activation peak that
            // OOMs 24 GB at 720p-area. numerics.seedvr2VaeF32=true is the kill-switch back to full precision.
            // The question is whether the backend computes in BF16, which is what the capability says. Keying on
            // the device kind meant a backend that gained BF16 would keep paying the fp32 activation peak.
            bool vaeBf16 = backend.Capabilities.SupportsBF16
                && !EngineKnobs.Seedvr2VaeF32.Value;
            SeedVr2VaeConfig vaeConfig = SeedVr2VaeConfig.Default with
            {
                ActivationDType = vaeBf16 ? DType.BF16 : DType.F32,
            };
            Logs.Info($"SeedVR2 VAE activation dtype: {vaeConfig.ActivationDType}");
            SeedVr2VaeEncoder encoder = new(vaeConfig);
            encoder.LoadWeights(vaeWeights);
            SeedVr2VaeDecoder decoder = new(vaeConfig);
            decoder.LoadWeights(vaeWeights);
            _posEmb = LoadPositiveEmbedding(embPath);

            _pipeline = new SeedVr2RestorePipeline(backend, dit, encoder, decoder, _posEmb);
            _loadedPath = ditPath;
            _loadedPacking = packing;
            _loadedBackend = backend;
            return _pipeline;
        }
    }

    /// <summary>Replaces every non-F32 tensor with an owned F32 copy, appended to <paramref name="owned"/> so the caller
    /// can release it; F32 entries pass through as the loader-owned views they are.</summary>
    internal static Dictionary<string, Tensor> CastAllToF32(IReadOnlyDictionary<string, Tensor> weights, List<Tensor> owned)
    {
        Dictionary<string, Tensor> result = new(weights.Count, StringComparer.Ordinal);
        foreach (KeyValuePair<string, Tensor> kvp in weights)
        {
            Tensor t = kvp.Value;
            if (t.DType != DType.F32)
            {
                t = t.CastTo(DType.F32);
                owned.Add(t);
            }
            result[kvp.Key] = t;
        }
        return result;
    }

    /// <summary>The frozen positive text embedding as an owned F32 tensor, from either form it ships in: the upstream
    /// <c>pos_emb.pt</c> (a bare BF16 tensor pickled by <c>torch.save</c>, which the pickle loader surfaces under the
    /// key <c>data</c>) or a safetensors export carrying <c>pos_emb</c>. The pickle loader is disposed once the cast
    /// copy exists; a safetensors loader stays open for the pipeline's lifetime like the others.</summary>
    internal Tensor LoadPositiveEmbedding(string embPath)
    {
        if (embPath.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase))
        {
            SafeTensorsLoader embLoader = new();
            embLoader.Load(embPath);
            _loaders.Add(embLoader);
            return embLoader.GetTensor("pos_emb").CastTo(DType.F32);
        }
        using PytorchPickleLoader pickle = new();
        pickle.Load(embPath);
        Dictionary<string, Tensor> tensors = pickle.GetAllTensors();
        Tensor raw = tensors.TryGetValue("pos_emb", out Tensor? named) ? named
            : tensors.TryGetValue("data", out Tensor? bare) ? bare
            : throw new InvalidOperationException(
                $"'{embPath}' holds no positive embedding (expected a bare tensor or a 'pos_emb' entry; found: {string.Join(", ", tensors.Keys)}).");
        if (raw.Shape.Rank != 2)
            throw new InvalidOperationException($"SeedVR2 positive embedding must be [tokens, dim]; '{embPath}' is rank {raw.Shape.Rank}.");
        return raw.CastTo(DType.F32);
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
        string? embSibling = Directory.EnumerateFiles(dir).Order()
            .FirstOrDefault(f => (f.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase) || f.EndsWith(".pt", StringComparison.OrdinalIgnoreCase))
                && Path.GetFileName(f).Contains("emb", StringComparison.OrdinalIgnoreCase));
        if (vaeSibling is null || embSibling is null)
            throw new InvalidOperationException(
                $"SeedVR2 needs a VAE safetensors and a positive embedding (pos_emb.pt or *emb*.safetensors) beside the DiT " +
                $"checkpoint in '{dir}', or a catalog entry with vae/embeddings asset roles.");
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
            foreach (Tensor copy in _castCopies)
                copy.Dispose();
            _castCopies.Clear();
            foreach (SafeTensorsLoader loader in _loaders)
                loader.Dispose();
            _loaders.Clear();
            _loadedPath = null;
            _loadedPacking = null;
            _loadedBackend = null;
        }
    }

    /// <summary>The DiT packing this request runs with: the knob if set, else the policy's <c>QuantizedCompute</c> lever.</summary>
    private DType? ChoosePacking(RestoreRequest request, IBackend backend)
    {
        string? forced = EngineKnobs.Seedvr2Packing.Value?.Trim().ToLowerInvariant();
        if (forced is "off")
            return null;
        if (forced is "q8")
            return DType.Q8_0;
        VramPolicy policy = VramPolicyRegistry.Resolve(backend, request.Vram);
        return SeedVr2WeightPacking.Choose(policy.QuantizedCompute);
    }
}
