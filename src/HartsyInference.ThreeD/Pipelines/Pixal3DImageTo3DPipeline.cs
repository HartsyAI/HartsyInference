using System.Numerics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Runtime;
using HartsyInference.Core.Sparse;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Geometry.Ops;
using HartsyInference.ThreeD.Models.Pixal3D;
using HartsyInference.ThreeD.Models.TrellisV2;
using HartsyInference.ThreeD.Pipelines.Requests;
using HartsyInference.ThreeD.Pipelines.Stages;

namespace HartsyInference.ThreeD.Pipelines;

/// <summary>
/// Pixal3D single-image to shape-only mesh (positions and computed normals, no texture). The stages are TRELLIS.2's cascade with the image
/// reaching each flow as global DINOv3 tokens plus per-voxel features read at the voxel's projected position (and, for the shape flows,
/// upsampled with NAF): foreground preprocessing with a 10 % wider crop, the dense structure flow and decoder, the 512 shape flow, the
/// upsample and the rounding quantizer, the 1024 shape flow at 1024 or 1536, the shape VAE decode, and dual-grid meshing. Only the 1024 and 1536
/// cascades exist. The camera is a field of view (upstream estimates it with MoGe, which is not ported), so the caller can pass one. Each stage sits
/// behind its interface, so a test or another model can replace any of them. The output is in glTF's Y-up frame.
/// </summary>
public sealed class Pixal3DImageTo3DPipeline : ThreeDPipelineBase
{
    /// <summary>Final dual-grid resolution when a request names none (upstream's default is 1536; 1024 fits a 12 GB card, as upstream's low-VRAM mode does).</summary>
    public const int DefaultGrid = 1024;

    /// <summary>Largest step override accepted.</summary>
    public const int MaxSteps = Trellis2ImageTo3DPipeline.MaxSteps;

    /// <summary>Grid the structure stage pools to; the 512 shape flow starts on it.</summary>
    public const int StructureGrid = Trellis2ImageTo3DPipeline.StructureGrid;

    /// <summary>Upstream's crop margin: <c>int(size * 1.1)</c> where TRELLIS.2 uses <c>int(size * 1)</c>.</summary>
    public const double CropScale = 1.1;

    private const int StructureImage = 512, CoarseImage = 512, FineImage = 1024, NafSize = 512;
    private const int StructureConditioningGrid = 16, UpsampleLevels = 4;

    // Channels of the dense structure latent, and the grid the 1024 flow was trained on (it serves up to 96 cells: its RoPE reads integer coordinates).
    private const int StructureLatentChannels = 8, FineTrainingGrid = 64;

    private readonly Trellis2PipelineConfig _config;
    private readonly IForegroundMatter? _matter;
    private readonly IPixal3DConditioner _conditioner;
    private readonly IPixal3DStructureStage _structure;
    private readonly IPixal3DShapeFlowStage _shape;
    private readonly ISparseShapeUpsampler _upsampler;
    private readonly ISparseShapeDecoder _decoder;
    private readonly IDualGridMesher _mesher;
    private readonly int _maxTokens;
    private readonly List<IDisposable> _owned;

    /// <param name="matter">Foreground matter for images without alpha; null requires an alpha channel.</param>
    /// <param name="maxTokens">Cap on the high-resolution flow's tokens; upstream uses <see cref="Trellis2Cascade.DefaultMaxTokens"/>.</param>
    /// <param name="owned">Disposables released with the pipeline: the stages that hold weights, and the matter if it should go too.</param>
    public Pixal3DImageTo3DPipeline(
        IBackend backend,
        Trellis2PipelineConfig config,
        IForegroundMatter? matter,
        IPixal3DConditioner conditioner,
        IPixal3DStructureStage structure,
        IPixal3DShapeFlowStage shape,
        ISparseShapeUpsampler upsampler,
        ISparseShapeDecoder decoder,
        IDualGridMesher mesher,
        IEnumerable<IDisposable> owned,
        int maxTokens = Trellis2Cascade.DefaultMaxTokens) : base(backend)
    {
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(conditioner);
        ArgumentNullException.ThrowIfNull(structure);
        ArgumentNullException.ThrowIfNull(shape);
        ArgumentNullException.ThrowIfNull(upsampler);
        ArgumentNullException.ThrowIfNull(decoder);
        ArgumentNullException.ThrowIfNull(mesher);
        ArgumentNullException.ThrowIfNull(owned);
        ArgumentOutOfRangeException.ThrowIfLessThan(maxTokens, 1);
        _config = config;
        _matter = matter;
        _conditioner = conditioner;
        _structure = structure;
        _shape = shape;
        _upsampler = upsampler;
        _decoder = decoder;
        _mesher = mesher;
        _maxTokens = maxTokens;
        _owned = [.. owned];
    }

    /// <summary>Loads the pipeline from a Pixal3D folder (for example <c>Models/3D/pixal3d</c>) or any file in it. The structure and shape decoders and
    /// DINOv3 are the TRELLIS.2 files and may sit beside it (see <see cref="Pixal3DModelFiles"/>); NAF is read with the safe-subset pickle reader.
    /// The two shape flows and the structure flow are read from disk for each request.</summary>
    /// <param name="matter">Foreground matter for images without alpha; the pipeline does not own it.</param>
    public static Pixal3DImageTo3DPipeline LoadFromPath(IBackend backend, string path, IForegroundMatter? matter = null)
    {
        ArgumentNullException.ThrowIfNull(backend);
        Pixal3DModelFiles files = Pixal3DModelFiles.Resolve(path);
        Trellis2PipelineConfig config = Trellis2PipelineConfig.Load(files.Pipeline);
        CheckFlowConfigs(files);

        List<IDisposable> owned = [];
        try
        {
            Pixal3DNaf naf = Pixal3DNaf.Load(files.Naf);
            owned.Add(naf);
            Pixal3DConditionerStage conditioner = new(Trellis2ImageConditioner.Load(files.Dino), naf);
            owned.Add(conditioner);
            Pixal3DStructureStage structure = Pixal3DStructureStage.Load(files.StructureFlow, files.StructureDecoder);
            owned.Add(structure);
            ShapeVaeDecoderStage decoder = ShapeVaeDecoderStage.LoadNextDcF16C32(files.ShapeDecoder, owned);
            owned.Add(decoder);
            return new Pixal3DImageTo3DPipeline(
                backend,
                config,
                matter,
                conditioner,
                structure,
                new Pixal3DShapeFlowStage(files.ShapeFlow512, files.ShapeFlow1024, config),
                decoder,
                decoder,
                new DualGridMesherStage(),
                owned);
        }
        catch
        {
            foreach (IDisposable disposable in owned)
                disposable.Dispose();
            throw;
        }
    }

    /// <summary>The final dual-grid resolution for a request; 0 means <see cref="DefaultGrid"/>.</summary>
    /// <exception cref="ArgumentException">The resolution is not 1024 or 1536.</exception>
    public static int ResolveGrid(int grid)
    {
        int resolved = grid == 0 ? DefaultGrid : grid;
        if (resolved is not (1024 or 1536))
        {
            throw new ArgumentException(
                $"Pixal3D grid must be 1024 or 1536 (the final dual-grid resolution; it has no 512 pipeline type); got {grid}.", nameof(grid));
        }

        return resolved;
    }

    /// <summary>Checks the request without loading any weight, so a bad one fails before the checkpoints are read.</summary>
    /// <param name="canMatte">Whether a matter is available for an image without alpha.</param>
    /// <exception cref="ArgumentException">The grid, seed, step count, camera or image is invalid.</exception>
    /// <exception cref="HartsyInferenceException">The image has no alpha and no matter is available.</exception>
    public static void ValidateRequest(Pixal3DOptions options, Trellis2InputImage image, bool canMatte)
    {
        ArgumentNullException.ThrowIfNull(options);
        ArgumentNullException.ThrowIfNull(image);
        ResolveGrid(options.Grid);
        if (options.Seed is < 0)
            throw new ArgumentException($"Seed must not be negative; got {options.Seed}.", nameof(options));
        if (options.Steps < 0 || options.Steps > MaxSteps)
            throw new ArgumentException($"Steps must be within [0, {MaxSteps}]; got {options.Steps}.", nameof(options));
        try
        {
            ViewOf(options).Validate();
        }
        catch (ArgumentOutOfRangeException error)
        {
            throw new ArgumentException(error.Message, nameof(options), error);
        }

        image.Validate();
        if (!canMatte && !image.HasForegroundAlpha)
        {
            throw new HartsyInferenceException(
                "Pixal3D needs the foreground cut out: supply an image with an alpha channel, or place a BiRefNet matting "
                + "checkpoint (Models/Vision/BiRefNet-matting/model.safetensors) so the engine can matte it.");
        }
    }

    /// <summary>Generates the shape mesh for one image.</summary>
    /// <param name="onProgress">Reports one step per sampler step and per stage; it may throw to cancel, and every tensor is released.</param>
    /// <exception cref="HartsyInferenceException">The sampled structure or shape holds no geometry.</exception>
    public Mesh Generate(Trellis2InputImage image, Pixal3DOptions? options = null, Action<GenerationProgress>? onProgress = null)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(image);
        options ??= new Pixal3DOptions();
        ValidateRequest(options, image, _matter is { IsAvailable: true });
        int target = ResolveGrid(options.Grid);
        Pixal3DView view = ViewOf(options);
        int seed = options.Seed ?? SeedGenerator.RandomSeed();
        Trellis2SamplerParams structureParams = WithSteps(_config.SparseStructureSampler, options.Steps);
        Trellis2SamplerParams shapeParams = WithSteps(_config.ShapeSampler, options.Steps);

        // Structure conditioning, structure, coarse conditioning, coarse shape, upsample, fine conditioning, fine shape, decode, mesh.
        int total = 1 + structureParams.Steps + 1 + shapeParams.Steps + 1 + 1 + shapeParams.Steps + 2;
        int done = 0;
        long started = MonotonicClock.NowNs();
        void Report() => onProgress?.Invoke(new GenerationProgress(done, total, (MonotonicClock.NowNs() - started) / 1e6));
        void Stage()
        {
            done++;
            Report();
        }

        void Tick(int _) => Stage();

        Trellis2PreparedImage prepared = Trellis2ImagePreprocessor.Prepare(Backend, image, _matter, CropScale);
        Pixal3DCondition? cond = null, negCond = null;
        SparseTensor? coarse = null, fine = null;
        try
        {
            cond = _conditioner.Encode(Backend, prepared, new Pixal3DConditionSpec(StructureImage, StructureConditioningGrid, null, 0), view);
            negCond = cond.ZerosLike();
            Stage();
            int[] voxels = _structure.Sample(Backend, cond, negCond, seed, structureParams, StructureGrid, Tick);
            DisposeConditioning(ref cond, ref negCond);

            cond = _conditioner.Encode(Backend, prepared, new Pixal3DConditionSpec(CoarseImage, StructureGrid, voxels, NafSize), view);
            negCond = cond.ZerosLike();
            Stage();
            coarse = _shape.Sample(Backend, Trellis2ShapeFlowKind.Res512, voxels, StructureGrid, cond, negCond, unchecked(seed + 1), shapeParams, Tick);
            DisposeConditioning(ref cond, ref negCond);

            int[] upsampled = _upsampler.Upsample(Backend, coarse, UpsampleLevels);
            coarse.Feats.Dispose();
            coarse = null;
            Stage();
            (int[] fineCoords, int fineResolution) = Trellis2Cascade.FitTokenCap(
                upsampled, Trellis2ImageTo3DPipeline.CoarseShapeResolution, target, _maxTokens, Trellis2CascadeQuantizer.RoundToCorners);
            int fineGrid = fineResolution / Trellis2Cascade.LatentStride;
            cond = _conditioner.Encode(Backend, prepared, new Pixal3DConditionSpec(FineImage, fineGrid, fineCoords, NafSize), view);
            negCond = cond.ZerosLike();
            Stage();
            fine = _shape.Sample(Backend, Trellis2ShapeFlowKind.Res1024, fineCoords, fineGrid, cond, negCond, unchecked(seed + 2), shapeParams, Tick);
            DisposeConditioning(ref cond, ref negCond);

            DualGridField field = _decoder.Decode(Backend, fine);
            fine.Feats.Dispose();
            fine = null;
            Stage();
            SceneGrid grid = new() { Resolution = fineResolution, Origin = new Vector3(-0.5f), VoxelSize = 1f / fineResolution };
            Mesh mesh = _mesher.Extract(field, grid);
            if (mesh.TriangleCount == 0)
                throw new HartsyInferenceException("The decoded shape holds no surface. Try another seed or a different grid.");
            FaceOrientation.Unify(mesh);
            MeshOps.ComputeVertexNormals(mesh);
            ToExportFrame(mesh);
            Stage();
            return mesh;
        }
        finally
        {
            // A cancel from onProgress or a failed stage releases what was built; Tensor.Dispose is idempotent.
            cond?.Dispose();
            negCond?.Dispose();
            coarse?.Feats.Dispose();
            fine?.Feats.Dispose();
        }
    }

    /// <summary>Turns the mesh from the frame the flows sample in into the exported one: the upstream export is a half turn about the up axis,
    /// <c>(x, y, z) -> (-x, y, -z)</c>, for positions and normals. Pixal3D samples with Y up and the camera on +Z, so unlike TRELLIS.2's Z-up frame
    /// it takes no up-axis conversion, and the half turn keeps the winding.</summary>
    internal static void ToExportFrame(Mesh mesh)
    {
        Rotate(mesh.Vertices);
        if (mesh.Normals is not null)
            Rotate(mesh.Normals);

        static void Rotate(float[] values)
        {
            for (int i = 0; i < values.Length; i += 3)
            {
                values[i] = -values[i];
                values[i + 2] = -values[i + 2];
            }
        }
    }

    /// <inheritdoc/>
    protected override void DisposeCore()
    {
        foreach (IDisposable disposable in _owned)
            disposable.Dispose();
        _owned.Clear();
    }

    private static Pixal3DView ViewOf(Pixal3DOptions options) => new(options.FovRadians ?? Pixal3DView.DefaultFovRadians, options.MeshScale);

    private static Trellis2SamplerParams WithSteps(Trellis2SamplerParams parameters, int steps)
        => steps > 0 ? parameters with { Steps = steps } : parameters;

    private static void DisposeConditioning(ref Pixal3DCondition? cond, ref Pixal3DCondition? negCond)
    {
        cond?.Dispose();
        negCond?.Dispose();
        cond = null;
        negCond = null;
    }

    // Reads the sidecar configs so a mismatched flow fails at load, not after the first stage has run.
    private static void CheckFlowConfigs(Pixal3DModelFiles files)
    {
        Trellis2FlowConfig structure = ReadConfig(files.StructureFlow);
        Trellis2FlowConfig coarse = ReadConfig(files.ShapeFlow512);
        Trellis2FlowConfig fine = ReadConfig(files.ShapeFlow1024);
        if (!structure.Dense || structure.Resolution != StructureConditioningGrid || structure.InChannels != StructureLatentChannels || !structure.HasProjection)
            throw new HartsyInferenceException($"'{files.StructureFlow}' is not the 16^3 x 8 channel projection structure flow.");
        foreach ((Trellis2FlowConfig config, string path) in new[] { (coarse, files.ShapeFlow512), (fine, files.ShapeFlow1024) })
        {
            if (config.Dense || !config.HasProjection || config.InChannels != Trellis2PipelineConfig.ShapeLatentChannels
                || config.CondChannels != structure.CondChannels || config.ProjInChannels != 2 * config.CondChannels)
                throw new HartsyInferenceException($"'{path}' is not a {Trellis2PipelineConfig.ShapeLatentChannels}-channel projection shape flow with NAF features.");
        }

        if (coarse.Resolution != StructureGrid)
            throw new HartsyInferenceException($"'{files.ShapeFlow512}' is trained for a {coarse.Resolution} grid; expected {StructureGrid}.");
        if (fine.Resolution != FineTrainingGrid)
            throw new HartsyInferenceException($"'{files.ShapeFlow1024}' is trained for a {fine.Resolution} grid; expected {FineTrainingGrid}.");
    }

    private static Trellis2FlowConfig ReadConfig(string safetensorsPath)
    {
        string sidecar = Path.ChangeExtension(safetensorsPath, ".json");
        if (!File.Exists(sidecar))
            throw new FileNotFoundException($"Pixal3D flow config not found beside the checkpoint: '{sidecar}'.", sidecar);
        return Trellis2FlowConfig.FromJson(File.ReadAllText(sidecar));
    }
}
