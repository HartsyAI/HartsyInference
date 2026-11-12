using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Sparse;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Models.Sam3dObjects;
using HartsyInference.ThreeD.Models.Trellis;
using HartsyInference.ThreeD.Pipelines.Stages;
using HartsyInference.Vision.Moge;

namespace HartsyInference.ThreeD.Pipelines;

/// <summary>
/// SAM 3D Objects (<c>facebookresearch/sam-3d-objects</c>): image + mask + pointmap to Gaussian splats and a pose. Preprocess, structure condition and flow (mixture-of-transformers
/// ShortCut sampler with guidance), occupancy decode with prune and downsample, pose decode, SLAT condition and flow, Gaussian decode. Each stage opens its checkpoint, runs and releases
/// it, so only one generator is resident at a time. The checkpoints are read from <c>.safetensors</c> or Lightning <c>.ckpt</c> files (the format is chosen by the file's bytes; a
/// <c>.safetensors</c> file wins when both exist). Not ported yet: MoGe (the caller supplies the pointmap), the mesh decoder, layout optimisation. No stage is compared with the Python reference.
/// </summary>
public sealed class Sam3dObjectsPipeline : ThreeDPipelineBase
{
    // slat_mean and slat_std of pipeline.yaml (the values the released pipeline uses, not the code constants).
    private static readonly float[] SlatMean = [0.12211431f, 0.37204156f, -1.26521907f, -2.05276058f, -3.10432536f, -0.11294304f, -0.85146744f, 0.45506954f];
    private static readonly float[] SlatStd = [2.37326008f, 2.13174402f, 2.2413953f, 2.30589401f, 2.1191894f, 1.8969511f, 2.41684989f, 2.08374642f];

    private const string MogeFile = "moge_vitl.safetensors";

    private readonly string _modelDirectory;

    /// <param name="backend">The compute backend; not owned.</param>
    /// <param name="modelDirectory">The folder holding <c>ss_generator</c>, <c>ss_decoder</c>, <c>slat_generator</c> and <c>slat_decoder_gs</c> (each as <c>.safetensors</c> or <c>.ckpt</c>).</param>
    /// <exception cref="DirectoryNotFoundException">The folder does not exist.</exception>
    public Sam3dObjectsPipeline(IBackend backend, string modelDirectory) : base(backend)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelDirectory);
        if (!Directory.Exists(modelDirectory))
            throw new DirectoryNotFoundException($"SAM 3D model folder not found: '{modelDirectory}'.");
        _modelDirectory = modelDirectory;
    }

    /// <summary>True when every checkpoint of the pipeline is present under <paramref name="modelDirectory"/>.</summary>
    public static bool HasWeights(string modelDirectory) =>
        Directory.Exists(modelDirectory) && new[] { "ss_generator", "ss_decoder", "slat_generator", "slat_decoder_gs" }.All(stem => Find(modelDirectory, stem) is not null);

    /// <summary>True when MoGe's weights are present, so a request without a pointmap can be served (<c>moge_vitl.safetensors</c> only: the upstream pickle nests its state dict under <c>model</c> and is not read; its config JSON is optional).</summary>
    public static bool HasMoge(string modelDirectory) => Directory.Exists(modelDirectory) && File.Exists(Path.Combine(modelDirectory, MogeFile));

    /// <summary>Generates the splats and the pose of one object.</summary>
    /// <param name="log">Receives one line per stage with the elapsed time since the call started.</param>
    /// <param name="onStep">Called after each sampler step with the stage name (<c>structure</c> or <c>slat</c>) and the step count so far.</param>
    public Sam3dObjectsResult Generate(Sam3dObjectsRequest request, Action<string>? log = null, Action<string, int>? onStep = null)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(request);
        int pixels = request.Width * request.Height;
        if (request.Width <= 0 || request.Height <= 0 || request.Rgb.Length != 3 * pixels || request.Mask.Length != pixels)
            throw new ArgumentException($"Expected {request.Width} x {request.Height} RGB ({3 * pixels} bytes) and a mask of {pixels} bytes.", nameof(request));
        int seed = request.Seed ?? SeedGenerator.RandomSeed();
        Stopwatch clock = Stopwatch.StartNew();
        void Log(string message) => log?.Invoke($"[{clock.Elapsed.TotalSeconds,7:F1}s] {message}");

        if (request.Pointmap is null && request.MogeResolutionLevel is < 0 or > 9)
            throw new ArgumentOutOfRangeException(nameof(request), request.MogeResolutionLevel, "The MoGe resolution level is 0 to 9.");
        (float[] rgb, float[] mask) = Planar(request);
        SamPointmapInput pointmap = request.Pointmap ?? EstimatePointmap(rgb, request, Log);
        SamPrepared prepared = SamPreprocessor.Prepare(rgb, mask, pointmap, request.Height, request.Width);
        Log("preprocessed");

        (int[] coords, SamPose pose) = StructureStage(request, prepared, seed, Log, onStep);
        int voxels = coords.Length / 4;
        if (voxels == 0)
            return new Sam3dObjectsResult { Splats = null, Pose = pose, Voxels = 0, Seed = seed };

        GaussianSplatCloud cloud = SlatStage(prepared, coords, request.SlatSteps, seed, Log, onStep);
        return new Sam3dObjectsResult { Splats = cloud, Pose = pose, Voxels = voxels, Seed = seed };
    }

    /// <summary>The planar float image (0 to 1) and the mask (alpha over 255, so 0 to 1 with soft edges kept) of the request.</summary>
    private static (float[] Rgb, float[] Mask) Planar(Sam3dObjectsRequest request)
    {
        int pixels = request.Width * request.Height;
        float[] rgb = new float[3 * pixels];
        float[] mask = new float[pixels];
        for (int i = 0; i < pixels; i++)
        {
            for (int c = 0; c < 3; c++)
                rgb[c * pixels + i] = request.Rgb[i * 3 + c] / 255f;
            mask[i] = request.Mask[i] / 255f;
        }

        return (rgb, mask);
    }

    /// <summary>
    /// MoGe on the whole image (<c>depth_model(image)</c> with <c>force_projection=False</c>), then the camera rotation to the PyTorch3D convention, <c>diag(-1, -1, 1)</c>: x and y flip
    /// sign (infinite stays infinite). The MoGe weights are released again before the structure stage loads.
    /// </summary>
    private SamPointmapInput EstimatePointmap(float[] rgb, Sam3dObjectsRequest request, Action<string> log)
    {
        string checkpointPath = Path.Combine(_modelDirectory, MogeFile);
        if (!File.Exists(checkpointPath))
            throw new FileNotFoundException($"The request has no pointmap and '{_modelDirectory}' holds no '{MogeFile}' to estimate one with.", checkpointPath);
        string configPath = Path.Combine(_modelDirectory, "moge_vitl_config.json");
        MogeConfig config = File.Exists(configPath) ? MogeConfig.FromJson(File.ReadAllText(configPath)) : MogeConfig.Released;
        MogeResult result;
        using (SamCheckpoint checkpoint = SamCheckpoint.Load(checkpointPath))
        {
            MogeModel model = new(config, checkpoint.Tensors);
            // The model's own tensors, not the checkpoint's: the backbone q, k and v are views of the fused ones, and the backend caches by tensor object.
            List<Tensor> weights = [.. model.EnumerateWeights()];
            try
            {
                Backend.PreloadWeights(weights);
                result = model.Infer(Backend, rgb, request.Height, request.Width, request.MogeResolutionLevel);
            }
            finally
            {
                Backend.FreeWeights(weights);
            }
        }

        Sweep();
        int plane = request.Height * request.Width;
        float[] points = result.Points;
        for (int i = 0; i < 2 * plane; i++)
            points[i] = -points[i];
        log($"MoGe pointmap (focal {result.Focal:F3}, shift {result.Shift:F3}, {result.Mask.Count(valid => valid)} of {plane} points valid)");
        return new SamPointmapInput(points, request.Height, request.Width);
    }

    /// <summary>Structure condition, flow, occupancy decode, prune, downsample and the pose.</summary>
    private (int[] Coords, SamPose Pose) StructureStage(Sam3dObjectsRequest request, SamPrepared prepared, int seed, Action<string> log, Action<string, int>? onStep)
    {
        Dictionary<string, Tensor> latents;
        using (SamCheckpoint checkpoint = OpenCheckpoint("ss_generator"))
        {
            log($"ss_generator opened ({checkpoint.Tensors.Count} tensors)");
            Dictionary<string, Tensor> embedder = checkpoint.Section(SamCheckpointSection.ConditionEmbedder);
            using Tensor cond = BuildCondition(SamStage.Structure, embedder, prepared.Structure);
            Backend.FreeWeights(embedder.Values);
            Sweep(cond);
            log($"structure condition {cond.Shape}");

            Dictionary<string, Tensor> flowWeights = SamMotFlow.Select(checkpoint.Section(SamCheckpointSection.Generator));
            List<Tensor> narrowed = [];
            Dictionary<string, Tensor> narrowedWeights = SamFlowWeights.Narrow(flowWeights, FlowLinearDType, narrowed);
            SamMotFlow flow = new(SamMotFlowConfig.Released, narrowedWeights);
            using Tensor zeros = Zeros(cond.Shape);
            Dictionary<string, Tensor> noise = [];
            try
            {
                int index = 0;
                foreach (SamMotLatent latent in SamMotFlowConfig.Released.Latents)
                    noise[latent.Name] = SeedGenerator.CreateNoise(new TensorShape(1, latent.Name == "shape" ? flow.TokensOf("shape") : 1, latent.Channels), seed + index++);
                SamSamplerParams parameters = SamSamplerParams.StructureDefault with { Steps = request.StructureSteps };
                latents = SamFlowSampler.Sample(
                    Backend, noise, parameters,
                    (x, tModel, dModel, unconditional) => flow.Forward(Backend, x, tModel, unconditional ? zeros : cond, dModel),
                    step => onStep?.Invoke("structure", step));
            }
            finally
            {
                foreach (Tensor tensor in noise.Values)
                    tensor.Dispose();
            }

            Backend.FreeWeights(flowWeights.Values);
            Backend.FreeWeights(narrowed);
            foreach (Tensor tensor in narrowed)
                tensor.Dispose();
            Sweep([.. latents.Values]);
            log($"structure flow sampled ({request.StructureSteps} steps)");
        }

        try
        {
            using Tensor occupancy = DecodeOccupancy(latents["shape"]);
            Sweep(occupancy);
            log($"occupancy decoded {occupancy.Shape}");
            int[] triples = ActiveTriples(occupancy);
            int original = triples.Length / 3;
            triples = SamSparseStructure.Prune(triples, 1);
            (int[] downsampled, int factor) = SamSparseStructure.Downsample(triples, SamSparseStructure.MaxCoords, 2, seed);
            log($"voxels {original} -> pruned {triples.Length / 3} -> downsampled {downsampled.Length / 3} (factor {factor})");

            SamPose pose = SamPoseDecoder.Decode(
                latents["6drotation_normalized"].AsReadOnlySpan<float>(), latents["translation"].AsReadOnlySpan<float>(), latents["scale"].AsReadOnlySpan<float>(),
                prepared.PointmapScale!, prepared.PointmapShift!, factor);
            int[] coords = new int[downsampled.Length / 3 * 4];
            for (int i = 0; i < downsampled.Length / 3; i++)
            {
                coords[i * 4] = 0;
                coords[i * 4 + 1] = downsampled[i * 3];
                coords[i * 4 + 2] = downsampled[i * 3 + 1];
                coords[i * 4 + 3] = downsampled[i * 3 + 2];
            }

            return (coords, pose);
        }
        finally
        {
            foreach (Tensor tensor in latents.Values)
                tensor.Dispose();
        }
    }

    /// <summary><c>ss_decoder(shape.permute(0, 2, 1).view(B, 8, 16, 16, 16))</c>: the 4096 x 8 tokens become 8 channels over a 16-cube, token <c>n</c> at <c>(n / 256, n / 16 % 16, n % 16)</c>.</summary>
    private Tensor DecodeOccupancy(Tensor shapeLatent)
    {
        using SamCheckpoint checkpoint = OpenCheckpoint("ss_decoder");
        Dictionary<string, Tensor> weights = checkpoint.Section(SamCheckpointSection.Root);
        SparseStructureDecoder decoder = new();
        decoder.LoadWeights(weights);
        using Tensor grid = new(new TensorShape(new long[] { 1, 8, 16, 16, 16 }), DType.F32);
        ReadOnlySpan<float> tokens = shapeLatent.AsReadOnlySpan<float>();
        Span<float> target = grid.AsSpan<float>();
        for (int n = 0; n < 4096; n++)
        {
            for (int c = 0; c < 8; c++)
                target[c * 4096 + n] = tokens[n * 8 + c];
        }

        Backend.PreloadWeights(decoder.EnumerateWeights());
        Tensor occupancy = decoder.Decode(Backend, grid);
        Backend.FreeWeights(decoder.EnumerateWeights());
        return occupancy;
    }

    /// <summary><c>argwhere(ss &gt; 0)</c> of a <c>[1, 1, 64, 64, 64]</c> logit grid as flat <c>(a, b, c)</c> triples in row-major order.</summary>
    private static int[] ActiveTriples(Tensor occupancy)
    {
        int d = (int)occupancy.Shape[2], h = (int)occupancy.Shape[3], w = (int)occupancy.Shape[4];
        ReadOnlySpan<float> values = occupancy.AsReadOnlySpan<float>();
        List<int> triples = [];
        for (int z = 0; z < d; z++)
        {
            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    if (values[(z * h + y) * w + x] > 0f)
                    {
                        triples.Add(z);
                        triples.Add(y);
                        triples.Add(x);
                    }
                }
            }
        }

        return [.. triples];
    }

    /// <summary>SLAT condition and flow over the voxels, de-normalisation and the Gaussian decode.</summary>
    private GaussianSplatCloud SlatStage(SamPrepared prepared, int[] coords, int steps, int seed, Action<string> log, Action<string, int>? onStep)
    {
        int voxels = coords.Length / 4;
        Tensor latent;
        using (SamCheckpoint checkpoint = OpenCheckpoint("slat_generator"))
        {
            log($"slat_generator opened ({checkpoint.Tensors.Count} tensors)");
            Dictionary<string, Tensor> embedder = checkpoint.Section(SamCheckpointSection.ConditionEmbedder);
            using Tensor cond = BuildCondition(SamStage.Slat, embedder, prepared.Slat);
            Backend.FreeWeights(embedder.Values);
            Sweep(cond);
            log($"SLAT condition {cond.Shape}");

            Dictionary<string, Tensor> flowWeights = SamMotFlow.Select(checkpoint.Section(SamCheckpointSection.Generator));
            List<Tensor> narrowed = [];
            SlatFlowModel flow = new();
            flow.LoadWeights(SamFlowWeights.Narrow(flowWeights, FlowLinearDType, narrowed));
            using Tensor zeros = Zeros(cond.Shape);
            Dictionary<string, Tensor> noise = new()
            {
                ["slat"] = SeedGenerator.CreateNoise(new TensorShape(1, voxels, 8), seed + 100),
            };
            Dictionary<string, Tensor> sampled;
            try
            {
                SamSamplerParams parameters = SamSamplerParams.SlatDefault with { Steps = steps };
                Backend.PreloadWeights(flow.EnumerateWeights());
                sampled = SamFlowSampler.Sample(
                    Backend, noise, parameters,
                    (x, tModel, _, unconditional) =>
                    {
                        SparseTensor velocity = flow.Forward(Backend, new SparseTensor(x["slat"], coords, 64), tModel, unconditional ? zeros : cond);
                        return new Dictionary<string, Tensor> { ["slat"] = velocity.Feats };
                    },
                    step => onStep?.Invoke("slat", step));
            }
            finally
            {
                foreach (Tensor tensor in noise.Values)
                    tensor.Dispose();
            }

            Backend.FreeWeights(flow.EnumerateWeights());
            foreach (Tensor tensor in narrowed)
                tensor.Dispose();
            latent = sampled["slat"];
            Sweep(latent);
            log($"SLAT flow sampled ({steps} steps)");
        }

        using (Tensor mean = Constant(SlatMean))
        using (Tensor std = Constant(SlatStd))
        {
            SparseTensor slat = new(latent, coords, 64);
            TrellisSlatSampler.Denormalize(slat, mean, std);
        }

        try
        {
            using SamCheckpoint decoderCheckpoint = OpenCheckpoint("slat_decoder_gs");
            SlatGaussianDecoder decoder = new();
            decoder.LoadWeights(decoderCheckpoint.Section(SamCheckpointSection.Root));
            Backend.PreloadWeights(decoder.EnumerateWeights());
            SparseTensor gs = decoder.Forward(Backend, new SparseTensor(latent, coords, 64));
            Backend.FreeWeights(decoder.EnumerateWeights());
            GaussianSplatCloud cloud = TrellisGaussianRepresentation.Build(gs);
            log($"Gaussians decoded ({voxels} voxels)");
            return cloud;
        }
        finally
        {
            latent.Dispose();
        }
    }

    /// <summary>
    /// Returns the device memory the stage before left reserved (cached activations, attention plans, pool reservations) after offloading what a later stage still reads. On Windows a process
    /// past about 8 GiB of device memory runs 10 to 100 times slower, and the structure stage's pool alone is 7 GB: the first full run took 28 s for the first ten SLAT steps and 350 s for
    /// the other fifteen once the pool and the SLAT activations together passed it. The same sweep the TRELLIS.2 stages use.
    /// </summary>
    /// <summary>What the flows' transformer-block linear weights are held as: F16 when <c>numerics.flowF16Weights</c> is set (half the memory, a 16-bit GEMM with F32 accumulation), otherwise the checkpoint's F32.</summary>
    private static DType FlowLinearDType => EngineKnobs.FlowF16Weights.Value ? DType.F16 : DType.F32;

    private void Sweep(params Tensor[] live) => Trellis2DeviceMemory.ReleaseBetweenStages(Backend, live);

    private Tensor BuildCondition(SamStage stage, Dictionary<string, Tensor> embedder, SamConditionInputs inputs)
    {
        Backend.PreloadWeights(embedder.Values);
        using SamStageCondition condition = new(stage, embedder);
        return condition.Build(Backend, inputs);
    }

    private static Tensor Zeros(TensorShape shape)
    {
        Tensor zeros = new(shape, DType.F32);
        zeros.AsSpan<float>().Clear();
        return zeros;
    }

    private static Tensor Constant(float[] values)
    {
        Tensor tensor = new(new TensorShape(values.Length), DType.F32);
        values.CopyTo(tensor.AsSpan<float>());
        return tensor;
    }

    private SamCheckpoint OpenCheckpoint(string stem) =>
        SamCheckpoint.Load(Find(_modelDirectory, stem) ?? throw new FileNotFoundException($"No '{stem}.safetensors' or '{stem}.ckpt' in '{_modelDirectory}'."));

    /// <summary>The <c>.safetensors</c> file of a stem if there is one, else the <c>.ckpt</c>.</summary>
    private static string? Find(string directory, string stem)
    {
        string safetensors = Path.Combine(directory, stem + ".safetensors");
        if (File.Exists(safetensors))
            return safetensors;
        string ckpt = Path.Combine(directory, stem + ".ckpt");
        return File.Exists(ckpt) ? ckpt : null;
    }

    /// <inheritdoc/>
    protected override void DisposeCore()
    {
    }
}
