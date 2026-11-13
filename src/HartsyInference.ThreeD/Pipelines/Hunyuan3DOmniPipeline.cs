using HartsyInference.Core.Backends;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Requests;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ThreeD.Models.Hunyuan3D;
using HartsyInference.ThreeD.Models.Hunyuan3D.Omni;
using HartsyInference.ThreeD.Pipelines.Requests;
using HartsyInference.Vision.Dinov2;

namespace HartsyInference.ThreeD.Pipelines;

/// <summary>A control signal for Hunyuan3D-Omni: its kind and its rows of 6 values (build them with the static <c>PoseRows</c>, <c>BoundingBoxRow</c>, <c>VoxelRows</c> and <c>PointRows</c> of <see cref="OmniControlEncoder"/>).</summary>
public sealed record OmniControl(OmniControlKind Kind, float[] Rows);

/// <summary>Hunyuan3D-Omni image plus control to mesh: the Hunyuan3D-2.1 shape stack (DINOv2-large, plain DiT, ShapeVAE) with a longer context, the DINO tokens followed by the control tokens and 10 signal tokens. Classifier-free guidance
/// replaces only the DINO tokens by zeros and keeps the control and signal tokens in both branches (<c>Hunyuan3DOmniSiTFlowMatchingPipeline</c>).</summary>
public sealed class Hunyuan3DOmniPipeline : ThreeDPipelineBase
{
    /// <summary>The files <see cref="Load"/> reads from the Omni folder (the DINOv2-large weights are a separate repository, placed in <c>dinov2-large/</c>).</summary>
    public static readonly IReadOnlyList<string> Files =
    [
        "config.json", "model/config.json", "model/pytorch_model.bin", "vae/config.json", "vae/pytorch_model.bin", "cond_encoder/pytorch_model.bin", DinoFile,
    ];

    /// <summary>The DINOv2-large weights beside the Omni files (<c>facebook/dinov2-large</c>).</summary>
    public const string DinoFile = "dinov2-large/model.safetensors";

    private readonly Hunyuan3DShapePipeline _shape;
    private readonly OmniControlEncoder _control;
    private readonly Hunyuan3DConfig _config;
    private readonly OmniCheckpoint _checkpoint;
    private readonly List<IDisposable> _loaders;

    private Hunyuan3DOmniPipeline(IBackend backend, Hunyuan3DShapePipeline shape, OmniControlEncoder control, Hunyuan3DConfig config, OmniCheckpoint checkpoint, List<IDisposable> loaders)
        : base(backend)
    {
        _shape = shape;
        _control = control;
        _config = config;
        _checkpoint = checkpoint;
        _loaders = loaders;
    }

    /// <summary>Loads the pipeline from a <c>tencent/Hunyuan3D-Omni</c> folder. The DINOv2-large weights are not in that repository (upstream fetches <c>facebook/dinov2-large</c>), so they come from <paramref name="dinoWeights"/>:
    /// a safetensors file with the HF layout (<c>facebook/dinov2-large</c>) or a safetensors repack of the Hunyuan3D-2.1 shape checkpoint (<c>conditioner.main_image_encoder.model.*</c>; the released pickle is not read).</summary>
    /// <param name="log">Receives a line per loading stage.</param>
    public static Hunyuan3DOmniPipeline Load(IBackend backend, string folder, string dinoWeights, Action<string>? log = null)
    {
        ArgumentNullException.ThrowIfNull(backend);
        if (!File.Exists(dinoWeights))
            throw new FileNotFoundException($"Hunyuan3D-Omni needs DINOv2-large weights; '{dinoWeights}' does not exist.", dinoWeights);
        bool half = backend.Device != DeviceKind.Cpu;
        List<IDisposable> loaders = [];
        OmniCheckpoint? checkpoint = null;
        try
        {
            log?.Invoke("reading the DiT, VAE and control encoder from the pickles");
            checkpoint = OmniCheckpoint.Load(folder, half, (done, total) =>
            {
                if (done % 100 == 0 || done == total)
                    log?.Invoke($"DiT tensors {done}/{total}");
            });
            Hunyuan3DConfig config = checkpoint.Config;

            SafeTensorsLoader dinoLoader = new();
            loaders.Add(dinoLoader);
            dinoLoader.Load(dinoWeights);
            Dictionary<string, Tensor> all = [];
            foreach ((string key, Tensor tensor) in dinoLoader.GetAllTensors())
                all[key] = tensor;
            Dictionary<string, Tensor> dinoTensors = Hunyuan3DCheckpointConverter.Convert(all).Dinov2;
            if (dinoTensors.Count == 0)
                dinoTensors = all;
            Dinov2VisionEncoder dino = new(Dinov2Preset.Large);
            dino.LoadWeights(dinoTensors);

            Hunyuan3DPlainDit dit = new(config);
            dit.LoadWeights(checkpoint.Dit, halfWeights: half);
            Hunyuan3DShapeVae vae = new(config);
            vae.LoadWeights(checkpoint.Vae, OmniCheckpoint.VaePrefix);
            OmniControlEncoder control = new(backend, checkpoint.Control);
            Hunyuan3DShapePipeline shape = new(backend, dino, dit, vae, config);
            log?.Invoke("loaded");
            return new Hunyuan3DOmniPipeline(backend, shape, control, config, checkpoint, loaders);
        }
        catch
        {
            foreach (IDisposable loader in loaders)
                loader.Dispose();
            checkpoint?.Dispose();
            throw;
        }
    }

    /// <summary>Generates a mesh from one image and one control signal.</summary>
    public ThreeDResult Generate(ImageTo3DRequest request, OmniControl control, Action<GenerationProgress>? onProgress = null, CancellationToken cancel = default)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(request);
        ArgumentNullException.ThrowIfNull(control);
        cancel.ThrowIfCancellationRequested();
        return _shape.SampleConditioned(() => Condition(request, control), request.Seed, request.Steps, request.CfgScale, request.GridResolution, request.IsoLevel, onProgress, cancel);
    }

    private Hunyuan3DShapePipeline.ConditionPair Condition(ImageTo3DRequest request, OmniControl control)
    {
        using Tensor controlTokens = _control.Encode(control.Kind, control.Rows);
        using Tensor image = _shape.EncodeImage(request.ImageRgb, request.Width, request.Height);
        (Tensor cond, Tensor uncond) = OmniContext.Build(Backend, image, controlTokens);
        return new Hunyuan3DShapePipeline.ConditionPair(cond, uncond);
    }

    /// <inheritdoc/>
    protected override void DisposeCore()
    {
        _shape.Dispose();
        _control.Dispose();
        foreach (IDisposable loader in _loaders)
            loader.Dispose();
        _loaders.Clear();
        _checkpoint.Dispose();
    }
}
