using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.ModelHandler.CheckpointConverters;
using HartsyInference.ModelHandler.SafeTensors;
using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Geometry.Ops;
using HartsyInference.ThreeD.Models.Hunyuan3D;
using HartsyInference.ThreeD.Pipelines.Requests;
using HartsyInference.Vision.Dinov2;

namespace HartsyInference.ThreeD.Pipelines;

/// <summary>Hunyuan3D-2 single-image → mesh pipeline: DINOv2 encodes the conditioning image, a flow-match
/// DiT denoises a VecSet shape latent (2-way CFG), the ShapeVAE decodes a dense occupancy field, and
/// marching cubes extracts a watertight mesh. Reuses the diffusion flow-match helpers
/// (<see cref="LancePipelineCommon"/>, <see cref="SeedGenerator"/>) and the 3D foundation
/// (<see cref="MarchingCubes"/>, <see cref="MeshOps"/>).
/// <para><b>Numerics validation-pending</b> — produces a real, watertight mesh structurally; per-checkpoint
/// fidelity awaits the reference-diff pass (see the research doc).</para></summary>
public sealed unsafe class Hunyuan3DShapePipeline : ThreeDPipelineBase
{
    private readonly Dinov2VisionEncoder _dino;
    private readonly Dinov2ImagePreprocessor _preprocessor;
    private readonly Hunyuan3DDit _dit;
    private readonly Hunyuan3DShapeVae _vae;
    private readonly Hunyuan3DConfig _cfg;
    private readonly List<IDisposable> _ownedLoaders = [];

    public Hunyuan3DShapePipeline(IBackend backend, Dinov2VisionEncoder dino, Hunyuan3DDit dit, Hunyuan3DShapeVae vae, Hunyuan3DConfig cfg)
        : base(backend)
    {
        _dino = dino;
        _dit = dit;
        _vae = vae;
        _cfg = cfg;
        _preprocessor = new Dinov2ImagePreprocessor(dino.Preset.ImageSize);
    }

    /// <summary>Loads a Hunyuan3D-2 shape pipeline from a local checkpoint path (a directory of
    /// <c>.safetensors</c> shards, or a single file). Merges all shards, splits them with
    /// <see cref="Hunyuan3DCheckpointConverter"/>, and builds the DINOv2 encoder + DiT + ShapeVAE. The
    /// memory-mapped weights stay alive for the pipeline's lifetime and are released on
    /// <see cref="Dispose"/>.
    /// <para>Defaults to <see cref="Hunyuan3DConfig.Hunyuan3D2"/> and <see cref="Dinov2Preset.Large"/>;
    /// pass explicit values for other variants. <b>Numerics validation-pending</b> — the converter key
    /// tables and these dims are validation-gated.</para></summary>
    public static Hunyuan3DShapePipeline LoadFromPath(
        IBackend backend, string modelPath, Hunyuan3DConfig? cfg = null, Dinov2Preset? dinoPreset = null)
    {
        ArgumentNullException.ThrowIfNull(backend);
        Hunyuan3DConfig config = cfg ?? Hunyuan3DConfig.Hunyuan3D2;
        Dinov2Preset dp = dinoPreset ?? Dinov2Preset.Giant;   // Hunyuan3D-2 conditions on DINOv2-giant (1536-dim)

        string[] files = Directory.Exists(modelPath)
            ? Directory.GetFiles(modelPath, "*.safetensors", SearchOption.AllDirectories)
            : [modelPath];
        if (files.Length == 0)
            throw new FileNotFoundException($"No .safetensors found under '{modelPath}'.");

        List<IDisposable> loaders = [];
        Dictionary<string, Tensor> all = [];
        foreach (string f in files.OrderBy(s => s, StringComparer.Ordinal))
        {
            SafeTensorsLoader loader = new();
            loader.Load(f);
            loaders.Add(loader);
            foreach ((string k, Tensor t) in loader.GetAllTensors()) all[k] = t;
        }

        Hunyuan3DCheckpointConverter.ConvertedWeights w = Hunyuan3DCheckpointConverter.Convert(all);
        Dinov2VisionEncoder dino = new(dp); dino.LoadWeights(w.Dinov2);
        Hunyuan3DDit dit = new(config); dit.LoadWeights(w.Dit);
        Hunyuan3DShapeVae vae = new(config); vae.LoadWeights(w.ShapeVae);

        Hunyuan3DShapePipeline pipeline = new(backend, dino, dit, vae, config);
        pipeline._ownedLoaders.AddRange(loaders);
        return pipeline;
    }

    /// <inheritdoc/>
    protected override void DisposeCore()
    {
        foreach (IDisposable d in _ownedLoaders) d.Dispose();
        _ownedLoaders.Clear();
    }

    /// <summary>Generates a mesh from a single conditioning image.</summary>
    public ThreeDResult Generate(ImageTo3DRequest request, Action<GenerationProgress>? onProgress = null)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(request);

        int seed = request.Seed ?? SeedGenerator.RandomSeed();
        int steps = request.Steps > 0 ? request.Steps : _cfg.NumInferenceSteps;
        float guidance = request.CfgScale > 0 ? request.CfgScale : _cfg.GuidanceScale;
        int gridRes = request.GridResolution > 0 ? request.GridResolution : _cfg.GridResolution;

        // 1. Image → DINOv2 conditioning tokens [1, seq, condDim].
        Tensor pixels = _preprocessor.Preprocess(request.ImageRgb, request.Width, request.Height);
        Backend.PreloadWeights(_dino.EnumerateWeights());
        Tensor cond = _dino.Encode(Backend, pixels);
        pixels.Dispose();
        Backend.FreeWeights(_dino.EnumerateWeights());
        Tensor uncond = new(cond.Shape, DType.F32); // zero (null) conditioning for CFG

        // 2. Flow-match denoise the VecSet latent. hy3dgen FlowMatchEulerDiscreteScheduler (shift 1): sigmas
        //    ASCEND [~0→1] (linspace(1,1000,steps)/1000, + a trailing 1.0); the DiT is fed t = sigma·1000, and the
        //    Euler update is prev = x + (sigma_next − sigma)·noise_pred. init latent = pure noise (init_noise_sigma 1).
        float[] sigmas = new float[steps + 1];
        for (int i = 0; i < steps; i++) sigmas[i] = (steps > 1 ? (1f + i * (999f / (steps - 1))) : 1f) / 1000f;
        sigmas[steps] = 1f;

        Tensor latents = SeedGenerator.CreateNoise(new TensorShape(1, _cfg.LatentTokens, _cfg.LatentChannels), seed);
        Backend.PreloadWeights(_dit.EnumerateWeights());
        for (int k = 0; k < steps; k++)
        {
            float t = sigmas[k] * 1000f, dt = sigmas[k + 1] - sigmas[k];
            Tensor vCond = _dit.Forward(Backend, latents, cond, t);
            Tensor vUncond = _dit.Forward(Backend, latents, uncond, t);
            FlowStepAscending(latents, vCond, vUncond, guidance, dt);
            vCond.Dispose();
            vUncond.Dispose();
            onProgress?.Invoke(new GenerationProgress(k + 1, steps, 0));
        }
        Backend.Sync();
        Backend.FreeWeights(_dit.EnumerateWeights());
        cond.Dispose();
        uncond.Dispose();

        // 3. Scale the denoised latent (latents /= scale_factor), then ShapeVAE decode → occupancy → mesh.
        float inv = 1f / _cfg.VaeScaleFactor;
        float* lp = (float*)latents.DataPointer;
        for (long i = 0; i < latents.Shape.ElementCount; i++) lp[i] *= inv;

        Backend.PreloadWeights(_vae.EnumerateWeights());
        ScalarField3D field = _vae.Decode(Backend, latents, gridRes, _cfg.BoundingBox);
        Backend.FreeWeights(_vae.EnumerateWeights());
        latents.Dispose();

        Mesh mesh = MeshOps.ComputeVertexNormals(MarchingCubes.Extract(field, request.IsoLevel != 0f ? request.IsoLevel : _cfg.IsoLevel));
        return new ThreeDResult { Mesh = mesh, Seed = seed };
    }

    /// <summary>hy3dgen flow-match Euler step with 2-way CFG: <c>noise_pred = uncond + cfg·(cond − uncond)</c>,
    /// then ascending update <c>latents += dt·noise_pred</c> (dt = σ_next − σ &gt; 0). vCond/vUncond are GPU
    /// activations; reading their <c>DataPointer</c> lazily syncs. latents is a host tensor (re-uploaded next step).</summary>
    private static void FlowStepAscending(Tensor latents, Tensor vCond, Tensor vUncond, float cfg, float dt)
    {
        long n = latents.Shape.ElementCount;
        float* z = (float*)latents.DataPointer, c = (float*)vCond.DataPointer, u = (float*)vUncond.DataPointer;
        for (long i = 0; i < n; i++) z[i] += dt * (u[i] + cfg * (c[i] - u[i]));
    }
}
