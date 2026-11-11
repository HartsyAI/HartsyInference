using HartsyInference.Core.Backends.Guidance;
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
    private readonly IHunyuan3DDenoiser _dit;
    private readonly Hunyuan3DShapeVae _vae;
    private readonly Hunyuan3DConfig _cfg;
    private readonly bool _flashVdm;
    private readonly List<IDisposable> _ownedLoaders = [];

    /// <param name="flashVdm">Decode with FlashVDM hierarchical volume decoding; <paramref name="vae"/> must then be the turbo VAE.</param>
    public Hunyuan3DShapePipeline(
        IBackend backend, Dinov2VisionEncoder dino, Hunyuan3DDit dit, Hunyuan3DShapeVae vae, Hunyuan3DConfig cfg, bool flashVdm = false)
        : this(backend, dino, (IHunyuan3DDenoiser)dit, vae, cfg, flashVdm)
    {
    }

    private Hunyuan3DShapePipeline(
        IBackend backend, Dinov2VisionEncoder dino, IHunyuan3DDenoiser dit, Hunyuan3DShapeVae vae, Hunyuan3DConfig cfg, bool flashVdm)
        : base(backend)
    {
        _dino = dino;
        _dit = dit;
        _vae = vae;
        _cfg = cfg;
        _flashVdm = flashVdm;
        _preprocessor = new Dinov2ImagePreprocessor(dino.Preset.ImageSize);
    }

    /// <summary>Loads a Hunyuan3D-2 shape pipeline from a local checkpoint path, merging all shards and keeping the memory-mapped weights alive for the pipeline's lifetime until <see cref="Dispose"/>.</summary>
    public static Hunyuan3DShapePipeline LoadFromPath(
        IBackend backend, string modelPath, Hunyuan3DConfig? cfg = null, Dinov2Preset? dinoPreset = null, string? flashVdmVaeDir = null)
    {
        ArgumentNullException.ThrowIfNull(backend);

        string[] files = Directory.Exists(modelPath)
            ? Directory.GetFiles(modelPath, "*.safetensors", SearchOption.AllDirectories) : [modelPath];
        if (files.Length == 0)
            throw new FileNotFoundException($"No .safetensors found under '{modelPath}'.");
        // The checkpoint's own config.yaml decides the architecture; a bare checkpoint without one is Hunyuan3D-2.
        string weightsDir = Path.GetDirectoryName(Path.GetFullPath(files.OrderBy(f => f, StringComparer.Ordinal).First()))!;
        string configPath = Path.Combine(weightsDir, "config.yaml");
        Hunyuan3DConfig config = cfg ?? (File.Exists(configPath)
            ? Hunyuan3DConfigFile.Parse(File.ReadAllText(configPath)) : Hunyuan3DConfig.Hunyuan3D2);

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
        // Hunyuan3D-2 conditions on DINOv2-giant (1536-dim); 2.1 on DINOv2-large (1024-dim).
        Dinov2Preset dp = dinoPreset ?? (config.CondDim == 1024 ? Dinov2Preset.Large : Dinov2Preset.Giant);
        Dinov2VisionEncoder dino = new(dp); dino.LoadWeights(w.Dinov2);
        IHunyuan3DDenoiser dit;
        if (config.PlainDit is not null)
        {
            Hunyuan3DPlainDit plain = new(config);
            plain.LoadWeights(w.Dit, halfWeights: backend.Device != DeviceKind.Cpu);
            dit = plain;
        }
        else
        {
            Hunyuan3DDit flux = new(config);
            flux.LoadWeights(w.Dit);
            dit = flux;
        }

        Hunyuan3DShapeVae vae = new(config); vae.LoadWeights(w.ShapeVae);
        if (flashVdmVaeDir is not null)
        {
            // FlashVDM decodes with its own VAE checkpoint (a narrower geo decoder), not the one bundled with the DiT.
            string vaeYaml = Path.Combine(flashVdmVaeDir, "config.yaml"), vaeWeights = Path.Combine(flashVdmVaeDir, "model.fp16.safetensors");
            if (!File.Exists(vaeYaml) || !File.Exists(vaeWeights))
                throw new FileNotFoundException($"FlashVDM needs the turbo VAE (config.yaml and model.fp16.safetensors) in '{flashVdmVaeDir}'.");
            SafeTensorsLoader vaeLoader = new();
            vaeLoader.Load(vaeWeights);
            loaders.Add(vaeLoader);
            Dictionary<string, Tensor> vaeTensors = [];
            foreach ((string k, Tensor t) in vaeLoader.GetAllTensors()) vaeTensors[k] = t;
            vae = new Hunyuan3DShapeVae(Hunyuan3DConfigFile.ParseVae(File.ReadAllText(vaeYaml), config));
            vae.LoadWeights(vaeTensors);
        }

        Hunyuan3DShapePipeline pipeline = new(backend, dino, dit, vae, config, flashVdm: flashVdmVaeDir is not null);
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
    /// <exception cref="NotSupportedException">The checkpoint is a multiview build; use <see cref="GenerateMultiView"/>.</exception>
    public ThreeDResult Generate(ImageTo3DRequest request, Action<GenerationProgress>? onProgress = null)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(request);
        if (_cfg.ViewCount != 1)
            throw new NotSupportedException("This is a multiview checkpoint; give it views with GenerateMultiView.");
        return Sample(
            () => EncodeViews([new Hunyuan3DViewImage(Hunyuan3DView.Front, request.ImageRgb, request.Width, request.Height)]),
            request.Seed, request.Steps, request.CfgScale, request.GridResolution, request.IsoLevel, onProgress);
    }

    /// <summary>Generates a mesh from one to four views of an object with a multiview (<c>2mv</c>) checkpoint.</summary>
    /// <exception cref="NotSupportedException">The checkpoint conditions on a single image.</exception>
    public ThreeDResult GenerateMultiView(Hunyuan3DMultiViewRequest request, Action<GenerationProgress>? onProgress = null)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(request);
        if (_cfg.ViewCount == 1)
            throw new NotSupportedException("This checkpoint conditions on a single image; use Generate.");
        if (request.Views.Count == 0 || request.Views.Count > _cfg.ViewCount || request.Views.Select(v => v.View).Distinct().Count() != request.Views.Count)
            throw new ArgumentException($"Give 1 to {_cfg.ViewCount} views, each at most once.", nameof(request));
        return Sample(() => EncodeViews(request.Views), request.Seed, request.Steps, request.CfgScale, request.GridResolution,
            request.IsoLevel, onProgress);
    }

    /// <summary>DINOv2 tokens of each view (plus the view embedding when the checkpoint is multiview), concatenated in view order into <c>[1, views * tokens, condDim]</c>.</summary>
    private Tensor EncodeViews(IReadOnlyList<Hunyuan3DViewImage> views)
    {
        Backend.PreloadWeights(_dino.EnumerateWeights());
        List<Tensor> tokens = [];
        foreach (Hunyuan3DViewImage view in views.OrderBy(v => (int)v.View))
        {
            Tensor pixels = _preprocessor.Preprocess(view.Rgb, view.Width, view.Height);
            Tensor encoded = _dino.Encode(Backend, pixels);
            pixels.Dispose();
            if (_cfg.ViewCount > 1)
            {
                int count = (int)encoded.Shape[1], width = (int)encoded.Shape[2];
                float[] embedding = Hunyuan3DViewEmbedding.Create((int)view.View, width);
                Tensor full = new(encoded.Shape, DType.F32);
                float* fp = (float*)full.DataPointer;
                for (int t = 0; t < count; t++)
                    embedding.AsSpan().CopyTo(new Span<float>(fp + (long)t * width, width));
                Tensor sum = new(encoded.Shape, DType.F32);
                Backend.Add(sum, encoded, full);
                encoded.Dispose(); full.Dispose();
                encoded = sum;
            }

            tokens.Add(encoded);
        }

        Backend.FreeWeights(_dino.EnumerateWeights());
        if (tokens.Count == 1)
            return tokens[0];
        int condDim = (int)tokens[0].Shape[2];
        long total = tokens.Sum(t => t.Shape[1]);
        Tensor joined = new(new TensorShape(1, total, condDim), DType.F32);
        Backend.Concat(joined, [.. tokens], 1);
        foreach (Tensor t in tokens) t.Dispose();
        return joined;
    }

    private ThreeDResult Sample(Func<Tensor> encode, int? requestSeed, int requestSteps, float requestCfg, int requestGrid, float requestIso,
        Action<GenerationProgress>? onProgress)
    {
        int seed = requestSeed ?? SeedGenerator.RandomSeed();
        int steps = requestSteps > 0 ? requestSteps : _cfg.NumInferenceSteps;
        float guidance = requestCfg > 0 ? requestCfg : _cfg.GuidanceScale;
        int gridRes = requestGrid > 0 ? requestGrid : _cfg.GridResolution;
        bool phase = EngineKnobs.ThreeDPhase.Value;
        System.Diagnostics.Stopwatch pw = System.Diagnostics.Stopwatch.StartNew();
        void Probe(string tag) { if (phase) { Backend.Sync(); Console.WriteLine($"[hy3d-phase] {tag}: {pw.ElapsedMilliseconds} ms"); pw.Restart(); } }

        // 1. Image(s) → DINOv2 conditioning tokens [1, seq, condDim].
        Tensor cond = encode();
        Hunyuan3DDebugDump.Dump("cond", cond);
        Probe("dinov2-cond");
        // Zero (null) conditioning for CFG; guidance-distilled checkpoints take the guidance as an embedding instead.
        Tensor? uncond = _cfg.GuidanceEmbed ? null : new(cond.Shape, DType.F32);

        // 2. Flow-match denoise the VecSet latent (hy3dgen FlowMatchEulerDiscreteScheduler, shift 1): ascending sigmas,
        //    prev = x + (sigma_next − sigma)·v, initial latent = pure noise.
        float[] sigmas = Hunyuan3DFlowSchedule.Sigmas(steps, _cfg.Scheduler, _cfg.PcmTimesteps);

        Tensor latents = SeedGenerator.CreateNoise(new TensorShape(1, _cfg.LatentTokens, _cfg.LatentChannels), seed);
        Backend.PreloadWeights(_dit.EnumerateWeights());
        for (int k = 0; k < steps; k++)
        {
            float dt = sigmas[k + 1] - sigmas[k];
            if (dt == 0f)
            {
                // The last step has zero width, so its velocity cannot change the latent.
                onProgress?.Invoke(new GenerationProgress(k + 1, steps, 0));
                continue;
            }
            float t = Hunyuan3DFlowSchedule.DitTimestep(sigmas[k]);
            if (uncond is null)
            {
                Tensor velocity = _dit.Forward(Backend, latents, cond, t, guidance);
                Backend.GuidedEulerStep(latents, velocity, velocity, GuidedEulerSpec.Lerp(1f, dt));
                velocity.Dispose();
                onProgress?.Invoke(new GenerationProgress(k + 1, steps, 0));
                continue;
            }
            Tensor vCond = _dit.Forward(Backend, latents, cond, t);
            Tensor vUncond = _dit.Forward(Backend, latents, uncond, t);
            // Device-resident CFG + ascending Euler (matches FlowStepAscending: z += dt·(uncond + cfg·(cond−uncond))).
            // Keeps `latents` on the GPU across the whole loop — no per-step D2H drain of the two velocities (the
            // graph replays stay async; the stream drains only once, at the post-loop scale-factor readback).
            Backend.GuidedEulerStep(latents, vCond, vUncond, GuidedEulerSpec.Lerp(guidance, dt));
            vCond.Dispose();
            vUncond.Dispose();
            onProgress?.Invoke(new GenerationProgress(k + 1, steps, 0));
        }
        Backend.Sync();
        Backend.FreeWeights(_dit.EnumerateWeights());
        cond.Dispose();
        uncond?.Dispose();
        Probe($"dit-loop ({steps} steps x2 CFG)");

        // 3. Scale the denoised latent (latents /= scale_factor), then ShapeVAE decode → occupancy → mesh.
        Tensor scaled = new(latents.Shape, DType.F32);
        Backend.Scale(scaled, latents, 1f / _vae.Config.VaeScaleFactor);
        latents.Dispose();
        latents = scaled;

        Backend.PreloadWeights(_vae.EnumerateWeights());
        float iso = requestIso != 0f ? requestIso : _cfg.IsoLevel;
        ScalarField3D field = _flashVdm
            ? _vae.DecodeFlashVdm(Backend, latents, gridRes, _cfg.BoundingBox, iso)
            : _vae.Decode(Backend, latents, gridRes, _cfg.BoundingBox);
        Backend.FreeWeights(_vae.EnumerateWeights());
        latents.Dispose();
        Probe($"vae-decode ({gridRes}^3)");

        Mesh mesh = Hunyuan3DSurface.Extract(field, iso);
        Probe("marching-cubes");
        return new ThreeDResult { Mesh = mesh, Seed = seed };
    }

}
