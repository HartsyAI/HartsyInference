using HartsyInference.Core.Backends;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Geometry.Ops;
using HartsyInference.ThreeD.Models.Hunyuan3D;
using HartsyInference.ThreeD.Pipelines.Requests;
using HartsyInference.Vision.Dinov2;

namespace HartsyInference.ThreeD.Pipelines;

/// <summary>Hunyuan3D-2 single-image → mesh pipeline: DINOv2 encodes the image, a flow-match DiT denoises a VecSet shape latent (2-way CFG), the ShapeVAE decodes an occupancy field, and marching cubes extracts a watertight mesh.</summary>
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

    /// <summary>Loads a Hunyuan3D-2 shape pipeline from a local checkpoint path, merging all shards and keeping the memory-mapped weights alive for the pipeline's lifetime until <see cref="Dispose"/>.</summary>
    public static Hunyuan3DShapePipeline LoadFromPath(
        IBackend backend, string modelPath, Hunyuan3DConfig? cfg = null, Dinov2Preset? dinoPreset = null)
    {
        ArgumentNullException.ThrowIfNull(backend);
        Hunyuan3DConfig config = cfg ?? Hunyuan3DConfig.Hunyuan3D2;
        Dinov2Preset dp = dinoPreset ?? Dinov2Preset.Giant;   // Hunyuan3D-2 conditions on DINOv2-giant (1536-dim)

        string[] files = Directory.Exists(modelPath)
            ? Directory.GetFiles(modelPath, "*.safetensors", SearchOption.AllDirectories) : [modelPath];
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
        bool phase = EngineKnobs.ThreeDPhase.Value;
        System.Diagnostics.Stopwatch pw = System.Diagnostics.Stopwatch.StartNew();
        void Probe(string tag) { if (phase) { Backend.Sync(); Console.WriteLine($"[hy3d-phase] {tag}: {pw.ElapsedMilliseconds} ms"); pw.Restart(); } }

        // 1. Image → DINOv2 conditioning tokens [1, seq, condDim].
        Tensor pixels = _preprocessor.Preprocess(request.ImageRgb, request.Width, request.Height);
        Backend.PreloadWeights(_dino.EnumerateWeights());
        Tensor cond = _dino.Encode(Backend, pixels);
        pixels.Dispose();
        Backend.FreeWeights(_dino.EnumerateWeights());
        Probe("dinov2-cond");
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
            // Device-resident CFG + ascending Euler (matches FlowStepAscending: z += dt·(uncond + cfg·(cond−uncond))).
            // Keeps `latents` on the GPU across the whole loop — no per-step D2H drain of the two velocities (the
            // graph replays stay async; the stream drains only once, at the post-loop scale-factor readback).
            Backend.CfgEulerStep(latents, vCond, vUncond, guidance, dt);
            vCond.Dispose();
            vUncond.Dispose();
            onProgress?.Invoke(new GenerationProgress(k + 1, steps, 0));
        }
        Backend.Sync();
        Backend.FreeWeights(_dit.EnumerateWeights());
        cond.Dispose();
        uncond.Dispose();
        Probe($"dit-loop ({steps} steps x2 CFG)");

        // 3. Scale the denoised latent (latents /= scale_factor), then ShapeVAE decode → occupancy → mesh.
        float inv = 1f / _cfg.VaeScaleFactor;
        float* lp = (float*)latents.DataPointer;
        for (long i = 0; i < latents.Shape.ElementCount; i++) lp[i] *= inv;

        Backend.PreloadWeights(_vae.EnumerateWeights());
        ScalarField3D field = _vae.Decode(Backend, latents, gridRes, _cfg.BoundingBox);
        Backend.FreeWeights(_vae.EnumerateWeights());
        latents.Dispose();
        Probe($"vae-decode ({gridRes}^3)");

        Mesh mesh = MeshOps.ComputeVertexNormals(MarchingCubes.Extract(field, request.IsoLevel != 0f ? request.IsoLevel : _cfg.IsoLevel));
        Probe("marching-cubes");
        return new ThreeDResult { Mesh = mesh, Seed = seed };
    }

}
