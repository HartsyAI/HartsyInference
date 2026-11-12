using System.Numerics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Runtime;
using HartsyInference.Core.Sparse;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Geometry.Ops;
using HartsyInference.ThreeD.Io;
using HartsyInference.ThreeD.Models.TrellisV2;
using HartsyInference.ThreeD.Pipelines.Requests;
using HartsyInference.ThreeD.Pipelines.Stages;

namespace HartsyInference.ThreeD.Pipelines;

/// <summary>
/// TRELLIS.2 single-image to mesh. <see cref="Generate"/> follows upstream's shape path: foreground preprocessing, DINOv3
/// conditioning, the dense structure flow and decoder, the shape flow (once at 512, or the 512 to 1024 cascade), the shape VAE
/// decode, and dual-grid meshing with the triangle winding unified; its output is positions and computed normals in glTF's Y-up
/// frame, rotated from TRELLIS.2's Z-up as upstream's GLB export does. <see cref="GenerateTextured"/> adds the texture flow, the
/// texture decoder guided by the shape decode's subdivisions, and the mesh post-processing that bakes the PBR attributes into a
/// UV-mapped material. Each stage sits behind its interface, so a test or another model can replace any of them.
/// </summary>
public sealed class Trellis2ImageTo3DPipeline : ThreeDPipelineBase
{
    /// <summary>Final dual-grid resolution when a request names none, upstream's default <c>1024_cascade</c>.</summary>
    public const int DefaultGrid = 1024;

    /// <summary>Baked texture edge in texels when a request names none, upstream's <c>to_glb</c> default.</summary>
    public const int DefaultTextureSize = 2048;

    /// <summary>Face count a textured mesh is decimated to when a request names none, upstream's <c>to_glb</c> default.</summary>
    public const int DefaultDecimationTarget = 1_000_000;

    /// <summary>Largest texture edge accepted.</summary>
    public const int MaxTextureSize = 8192;

    /// <summary>Largest step override accepted.</summary>
    public const int MaxSteps = 200;

    /// <summary>Grid the structure stage pools to; every pipeline type starts the shape flow on it.</summary>
    public const int StructureGrid = 32;

    /// <summary>Latent grid of the 1024 flows, twice <see cref="StructureGrid"/>.</summary>
    public const int FineLatentGrid = StructureGrid * 2;

    /// <summary>Resolution of the 512 shape flow's output once upsampled: <c>StructureGrid · 16</c>.</summary>
    public const int CoarseShapeResolution = StructureGrid * Trellis2Cascade.LatentStride;

    private const int CoarseConditioning = 512;
    private const int FineConditioning = 1024;
    private const int UpsampleLevels = 4;

    private readonly Trellis2PipelineConfig _config;
    private readonly IForegroundMatter? _matter;
    private readonly ITrellis2Conditioner _conditioner;
    private readonly ITrellis2StructureStage _structure;
    private readonly ITrellis2ShapeFlowStage _shape;
    private readonly ISparseShapeUpsampler _upsampler;
    private readonly ISparseShapeDecoder _decoder;
    private readonly IDualGridMesher _mesher;
    private readonly ITrellis2TextureFlowStage? _textureFlow;
    private readonly ITrellis2TextureDecoderStage? _textureDecoder;
    private readonly int _maxTokens;
    private readonly List<IDisposable> _owned;

    /// <param name="matter">Foreground matter for images without alpha; null requires an alpha channel.</param>
    /// <param name="maxTokens">Cap on the high-resolution flow's tokens; upstream uses <see cref="Trellis2Cascade.DefaultMaxTokens"/>.</param>
    /// <param name="owned">Disposables released with the pipeline: the stages that hold weights, and the matter if it should go too.</param>
    /// <param name="textureFlow">The texture sampler stage; null leaves the pipeline shape-only. Needs <paramref name="textureDecoder"/>.</param>
    /// <param name="textureDecoder">The texture decoder stage; the shape <paramref name="decoder"/> must then be an <see cref="ISparseShapeSubdivisionDecoder"/>.</param>
    public Trellis2ImageTo3DPipeline(
        IBackend backend,
        Trellis2PipelineConfig config,
        IForegroundMatter? matter,
        ITrellis2Conditioner conditioner,
        ITrellis2StructureStage structure,
        ITrellis2ShapeFlowStage shape,
        ISparseShapeUpsampler upsampler,
        ISparseShapeDecoder decoder,
        IDualGridMesher mesher,
        IEnumerable<IDisposable> owned,
        int maxTokens,
        ITrellis2TextureFlowStage? textureFlow,
        ITrellis2TextureDecoderStage? textureDecoder) : base(backend)
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
        if ((textureFlow is null) != (textureDecoder is null))
            throw new ArgumentException("The texture flow and the texture decoder come together; give both or neither.");
        if (textureFlow is not null && decoder is not ISparseShapeSubdivisionDecoder)
            throw new ArgumentException("The texture decoder follows the shape decoder's subdivisions; the shape decoder must record them.", nameof(decoder));
        _textureFlow = textureFlow;
        _textureDecoder = textureDecoder;
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

    /// <summary>A shape-only pipeline; see the full constructor for the texture stages.</summary>
    public Trellis2ImageTo3DPipeline(
        IBackend backend,
        Trellis2PipelineConfig config,
        IForegroundMatter? matter,
        ITrellis2Conditioner conditioner,
        ITrellis2StructureStage structure,
        ITrellis2ShapeFlowStage shape,
        ISparseShapeUpsampler upsampler,
        ISparseShapeDecoder decoder,
        IDualGridMesher mesher,
        IEnumerable<IDisposable> owned,
        int maxTokens = Trellis2Cascade.DefaultMaxTokens)
        : this(backend, config, matter, conditioner, structure, shape, upsampler, decoder, mesher, owned, maxTokens, null, null)
    {
    }

    /// <summary>Loads the pipeline from a TRELLIS.2 folder (for example <c>Models/3D/trellis2</c>) or any file in it. The shape VAE decoder
    /// and DINOv3 may sit in sibling folders (see <see cref="Trellis2ModelFiles"/>). The two shape flows, the structure flow and the
    /// texture flows are read from disk for each request. The texture path is enabled when the texture decoder, a texture flow and the
    /// <c>pipeline.json</c> texture entries are all present.</summary>
    /// <param name="matter">Foreground matter for images without alpha; the pipeline does not own it.</param>
    public static Trellis2ImageTo3DPipeline LoadFromPath(IBackend backend, string path, IForegroundMatter? matter = null)
    {
        ArgumentNullException.ThrowIfNull(backend);
        Trellis2ModelFiles files = Trellis2ModelFiles.Resolve(path);
        Trellis2PipelineConfig config = Trellis2PipelineConfig.Load(files.Pipeline);
        CheckFlowConfigs(files);

        List<IDisposable> owned = [];
        try
        {
            Trellis2ConditionerStage conditioner = new(Trellis2ImageConditioner.Load(files.Dino));
            owned.Add(conditioner);
            Trellis2StructureStage structure = Trellis2StructureStage.Load(files.StructureFlow, files.StructureDecoder);
            owned.Add(structure);
            ShapeVaeDecoderStage decoder = ShapeVaeDecoderStage.LoadNextDcF16C32(files.ShapeDecoder, owned);
            owned.Add(decoder);
            ITrellis2TextureFlowStage? textureFlow = null;
            ITrellis2TextureDecoderStage? textureDecoder = null;
            if (files.HasTexture && config.HasTexture)
            {
                CheckTextureFlowConfigs(files, ReadConfig(files.StructureFlow).CondChannels);
                textureFlow = new Trellis2TextureFlowStage(files.TextureFlow512, files.TextureFlow1024, config);
                Trellis2TextureDecoderStage decoderStage = new(files.TextureDecoder!);
                owned.Add(decoderStage);
                textureDecoder = decoderStage;
            }

            return new Trellis2ImageTo3DPipeline(
                backend,
                config,
                matter,
                conditioner,
                structure,
                new Trellis2ShapeFlowStage(files.ShapeFlow512, files.ShapeFlow1024, config),
                decoder,
                decoder,
                new DualGridMesherStage(),
                owned,
                Trellis2Cascade.DefaultMaxTokens,
                textureFlow,
                textureDecoder);
        }
        catch
        {
            foreach (IDisposable disposable in owned)
                disposable.Dispose();
            throw;
        }
    }

    /// <summary>Whether the texture path is installed: <see cref="GenerateTextured"/> works only then.</summary>
    public bool HasTexture => _textureFlow is not null;

    /// <summary>The pipeline type for a final grid resolution; 0 means <see cref="DefaultGrid"/>.</summary>
    /// <exception cref="ArgumentException">The resolution is not 512, 1024 or 1536.</exception>
    public static Trellis2PipelineType ResolveType(int grid)
    {
        int resolved = grid == 0 ? DefaultGrid : grid;
        return resolved switch
        {
            512 => Trellis2PipelineType.Res512,
            1024 => Trellis2PipelineType.Cascade1024,
            1536 => Trellis2PipelineType.Cascade1536,
            _ => throw new ArgumentException(
                $"TRELLIS.2 grid must be 512, 1024 or 1536 (the final dual-grid resolution); got {grid}.", nameof(grid)),
        };
    }

    /// <summary>Checks the request without loading any weight, so a bad one fails before the checkpoints are read.</summary>
    /// <param name="canMatte">Whether a matter is available for an image without alpha.</param>
    /// <exception cref="ArgumentException">The grid, seed, step count or image is invalid.</exception>
    /// <exception cref="HartsyInferenceException">The image has no alpha and no matter is available.</exception>
    public static void ValidateRequest(Trellis2Options options, Trellis2InputImage image, bool canMatte)
    {
        ArgumentNullException.ThrowIfNull(options);
        ArgumentNullException.ThrowIfNull(image);
        ResolveType(options.Grid);
        if (options.Seed is < 0)
            throw new ArgumentException($"Seed must not be negative; got {options.Seed}.", nameof(options));
        if (options.Steps < 0 || options.Steps > MaxSteps)
            throw new ArgumentException($"Steps must be within [0, {MaxSteps}]; got {options.Steps}.", nameof(options));
        if (options.TextureSize != 0 && (options.TextureSize < 16 || options.TextureSize > MaxTextureSize))
            throw new ArgumentException($"TextureSize must be 0 or within [16, {MaxTextureSize}]; got {options.TextureSize}.", nameof(options));
        if (options.DecimationTarget < 0)
            throw new ArgumentException($"DecimationTarget must not be negative; got {options.DecimationTarget}.", nameof(options));
        image.Validate();
        if (!canMatte && !image.HasForegroundAlpha)
        {
            throw new HartsyInferenceException(
                "TRELLIS.2 needs the foreground cut out: supply an image with an alpha channel, or place a BiRefNet matting "
                + "checkpoint (Models/Vision/BiRefNet-matting/model.safetensors) so the engine can matte it.");
        }
    }

    /// <summary>Generates the shape mesh for one image; <see cref="Trellis2Options.Texture"/> is ignored (see <see cref="GenerateTextured"/>).</summary>
    /// <param name="onProgress">Reports one step per sampler step and per stage; it may throw to cancel, and every tensor is released.</param>
    /// <exception cref="HartsyInferenceException">The sampled structure or shape holds no geometry.</exception>
    public Mesh Generate(Trellis2InputImage image, Trellis2Options? options = null, Action<GenerationProgress>? onProgress = null)
        => Run(image, options, onProgress, textured: false, CancellationToken.None).Mesh;

    /// <summary>
    /// Generates a textured PBR model for one image: the shape path, then the texture flow over the shape latent's voxels, the
    /// texture decoder guided by the shape decode's subdivisions, and <see cref="PbrMeshPostProcessor"/> (clean, decimate, unwrap, bake,
    /// inpaint). The result is in TRELLIS.2's Z-up frame; <see cref="ToGlb"/> rotates it into glTF's Y-up.
    /// </summary>
    /// <param name="onProgress">Reports one step per sampler step and per stage; it may throw to cancel, and every tensor is released.</param>
    /// <param name="cancellationToken">Checked between the post-processing stages and inside decimation and baking.</param>
    /// <exception cref="HartsyInferenceException">The texture checkpoints are not installed, or the sampled shape holds no geometry.</exception>
    public PbrPostProcessResult GenerateTextured(
        Trellis2InputImage image, Trellis2Options? options = null, Action<GenerationProgress>? onProgress = null,
        CancellationToken cancellationToken = default)
        => Run(image, options, onProgress, textured: true, cancellationToken).Textured!;

    /// <summary>Serializes a <see cref="GenerateTextured"/> result: the Z-up frame is rotated to glTF's Y-up and the material is
    /// double-sided, as upstream's <c>to_glb</c> marks the standard (non-remeshed) path.</summary>
    public static byte[] ToGlb(PbrPostProcessResult result)
    {
        ArgumentNullException.ThrowIfNull(result);
        return PbrGlbWriter.Write(result.Mesh, result.Material, new PbrGlbOptions { ConvertZUpToYUp = true, DoubleSided = true });
    }

    private Trellis2Output Run(Trellis2InputImage image, Trellis2Options? options, Action<GenerationProgress>? onProgress, bool textured,
        CancellationToken cancellationToken)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(image);
        options ??= new Trellis2Options();
        ValidateRequest(options, image, _matter is { IsAvailable: true });
        Trellis2PipelineType type = ResolveType(options.Grid);
        bool cascade = type != Trellis2PipelineType.Res512;
        Trellis2ShapeFlowKind textureKind = cascade ? Trellis2ShapeFlowKind.Res1024 : Trellis2ShapeFlowKind.Res512;
        if (textured)
            RequireTexture(textureKind);
        int seed = options.Seed ?? SeedGenerator.RandomSeed();
        Trellis2SamplerParams structureParams = WithSteps(_config.SparseStructureSampler, options.Steps);
        Trellis2SamplerParams shapeParams = WithSteps(_config.ShapeSampler, options.Steps);
        Trellis2SamplerParams? textureParams = textured ? WithSteps(_config.TextureSampler!, options.Steps) : null;

        // Conditioning, structure, coarse shape, [upsample, fine shape,] decode, then the mesh, or for texture its flow, decode and bake.
        int total = 1 + structureParams.Steps + shapeParams.Steps * (cascade ? 2 : 1) + (cascade ? 1 : 0) + 1
            + (textured ? textureParams!.Steps + 2 : 1);
        int done = 0;
        long started = MonotonicClock.NowNs();
        void Report() => onProgress?.Invoke(new GenerationProgress(done, total, (MonotonicClock.NowNs() - started) / 1e6));
        void Tick(int step)
        {
            done++;
            Report();
        }

        Trellis2PreparedImage prepared = Trellis2ImagePreprocessor.Prepare(Backend, image, _matter);
        Tensor? cond = null, negCond = null, fineCond = null, fineNeg = null;
        SparseTensor? coarse = null, fine = null, textureLatent = null, textureVoxels = null;
        ShapeVaeSubdivisions? guides = textured ? new ShapeVaeSubdivisions() : null;
        try
        {
            cond = _conditioner.Encode(Backend, prepared, CoarseConditioning);
            negCond = Zeros(cond.Shape);
            done++;
            Report();

            int[] voxels = _structure.Sample(Backend, cond, negCond, seed, structureParams, StructureGrid, Tick);
            coarse = _shape.Sample(Backend, Trellis2ShapeFlowKind.Res512, voxels, StructureGrid, cond, negCond, unchecked(seed + 1), shapeParams, Tick);

            SparseTensor latent;
            int resolution;
            if (cascade)
            {
                int[] upsampled = _upsampler.Upsample(Backend, coarse, UpsampleLevels);
                coarse.Feats.Dispose();
                coarse = null;
                done++;
                Report();
                (int[] fineCoords, int fineResolution) = Trellis2Cascade.FitTokenCap(
                    upsampled, CoarseShapeResolution, (int)type, _maxTokens);
                fineCond = _conditioner.Encode(Backend, prepared, FineConditioning);
                fineNeg = Zeros(fineCond.Shape);
                fine = _shape.Sample(
                    Backend, Trellis2ShapeFlowKind.Res1024, fineCoords, fineResolution / Trellis2Cascade.LatentStride,
                    fineCond, fineNeg, unchecked(seed + 2), shapeParams, Tick);
                latent = fine;
                resolution = fineResolution;
            }
            else
            {
                latent = coarse;
                resolution = (int)type;
            }

            DualGridField field = guides is null
                ? _decoder.Decode(Backend, latent)
                : ((ISparseShapeSubdivisionDecoder)_decoder).Decode(Backend, latent, guides);
            done++;
            Report();
            SceneGrid grid = new() { Resolution = resolution, Origin = new Vector3(-0.5f), VoxelSize = 1f / resolution };
            Mesh mesh = _mesher.Extract(field, grid);
            if (mesh.TriangleCount == 0)
                throw new HartsyInferenceException("The decoded shape holds no surface. Try another seed or a different grid.");
            if (!textured)
            {
                mesh = ShapeMeshFinisher.Finish(mesh, options.CleanMesh);
                ToYUp(mesh);
                done++;
                Report();
                return new Trellis2Output(mesh, null);
            }

            // The post-processor unifies the winding after decimation and recomputes the normals, so the raw dual-grid mesh goes in.
            Tensor texCond = cascade ? fineCond! : cond;
            Tensor texNeg = cascade ? fineNeg! : negCond;
            // The texture flow stage sweeps the device before it starts; the guides the texture decoder reads later must survive that.
            for (int level = 0; level < guides!.Count; level++)
                Backend.OffloadActivation(guides.Level(level));
            textureLatent = _textureFlow!.Sample(
                Backend, textureKind, latent, texCond, texNeg, unchecked(seed + 3), textureParams!, Tick);
            textureVoxels = _textureDecoder!.Decode(Backend, textureLatent, guides!);
            done++;
            Report();
            SparseAttributeVolume volume = BuildVolume(textureVoxels, resolution);
            PbrPostProcessOptions post = new()
            {
                TextureSize = options.TextureSize > 0 ? options.TextureSize : DefaultTextureSize,
                DecimationTarget = options.DecimationTarget > 0 ? options.DecimationTarget : DefaultDecimationTarget,
            };
            PbrPostProcessResult baked = PbrMeshPostProcessor.Process(mesh, volume, PbrAttributeLayout.Default, post, cancellationToken);
            done++;
            Report();
            return new Trellis2Output(baked.Mesh, baked);
        }
        finally
        {
            // A cancel from onProgress or a failed stage releases what was built; Tensor.Dispose is idempotent.
            cond?.Dispose();
            negCond?.Dispose();
            fineCond?.Dispose();
            fineNeg?.Dispose();
            coarse?.Feats.Dispose();
            fine?.Feats.Dispose();
            textureLatent?.Feats.Dispose();
            textureVoxels?.Feats.Dispose();
            guides?.Dispose();
        }
    }

    /// <summary>The texture decoder's voxels as a <see cref="SparseAttributeVolume"/> on the mesh's grid: voxel <c>i</c> spans
    /// <c>[i, i + 1)</c> voxel units from the corner <c>(-0.5, -0.5, -0.5)</c>, as upstream's <c>MeshWithVoxel</c> places it.</summary>
    internal static SparseAttributeVolume BuildVolume(SparseTensor textureVoxels, int resolution)
    {
        if (textureVoxels.Resolution != resolution)
        {
            throw new HartsyInferenceException(
                $"The texture voxels are on a {textureVoxels.Resolution} grid but the mesh is on {resolution}; the decoders disagree.");
        }

        int n = textureVoxels.Count;
        if (textureVoxels.Channels != ShapeVaeKeyMap.TextureOutputChannels)
            throw new HartsyInferenceException($"The texture voxels hold {textureVoxels.Channels} channels; expected {ShapeVaeKeyMap.TextureOutputChannels}.");
        int[] coords = new int[n * 3];
        for (int i = 0; i < n; i++)
        {
            coords[i * 3] = textureVoxels.Coords[i * 4 + 1];
            coords[i * 3 + 1] = textureVoxels.Coords[i * 4 + 2];
            coords[i * 3 + 2] = textureVoxels.Coords[i * 4 + 3];
        }

        return new SparseAttributeVolume(
            coords, textureVoxels.Feats.AsReadOnlySpan<float>().ToArray(), textureVoxels.Channels, (resolution, resolution, resolution),
            new Vector3(-0.5f), 1f / resolution);
    }

    private void RequireTexture(Trellis2ShapeFlowKind kind)
    {
        if (_textureFlow is null || _textureDecoder is null)
        {
            throw new HartsyInferenceException(
                "TRELLIS.2 texture is not installed: it needs the texture decoder, a texture flow and the pipeline.json texture entries. "
                + "Place the files named in docs/Research/TRELLIS2_ARCHITECTURE.md beside the shape checkpoints, or request a shape-only mesh.");
        }

        if (!_textureFlow.Supports(kind))
        {
            throw new HartsyInferenceException(
                kind == Trellis2ShapeFlowKind.Res512
                    ? "Grid 512 needs the 512 texture flow, which is not installed."
                    : "Grids 1024 and 1536 need the 1024 texture flow, which is not installed.");
        }
    }

    private sealed record Trellis2Output(Mesh Mesh, PbrPostProcessResult? Textured);

    /// <inheritdoc/>
    protected override void DisposeCore()
    {
        foreach (IDisposable disposable in _owned)
            disposable.Dispose();
        _owned.Clear();
    }

    /// <summary>Rotates TRELLIS.2's Z-up frame into glTF's Y-up, <c>(x, y, z) -> (x, z, -y)</c>, for positions and normals.</summary>
    internal static void ToYUp(Mesh mesh)
    {
        Rotate(mesh.Vertices);
        if (mesh.Normals is not null)
            Rotate(mesh.Normals);

        static void Rotate(float[] values)
        {
            for (int i = 0; i < values.Length; i += 3)
            {
                float y = values[i + 1];
                values[i + 1] = values[i + 2];
                values[i + 2] = -y;
            }
        }
    }

    private static Trellis2SamplerParams WithSteps(Trellis2SamplerParams parameters, int steps)
        => steps > 0 ? parameters with { Steps = steps } : parameters;

    private Tensor Zeros(TensorShape shape)
    {
        Tensor zeros = new(shape, DType.F32);
        Backend.Fill(zeros, 0f);
        return zeros;
    }

    // Reads the sidecar configs so a mismatched flow fails at load, not after the first stage has run.
    private static void CheckFlowConfigs(Trellis2ModelFiles files)
    {
        Trellis2FlowConfig structure = ReadConfig(files.StructureFlow);
        Trellis2FlowConfig coarse = ReadConfig(files.ShapeFlow512);
        Trellis2FlowConfig fine = ReadConfig(files.ShapeFlow1024);
        if (!structure.Dense || structure.Resolution != 16 || structure.InChannels != 8)
            throw new HartsyInferenceException($"'{files.StructureFlow}' is not the 16^3 x 8 channel structure flow.");
        foreach ((Trellis2FlowConfig config, string path) in new[] { (coarse, files.ShapeFlow512), (fine, files.ShapeFlow1024) })
        {
            if (config.Dense || config.InChannels != Trellis2PipelineConfig.ShapeLatentChannels || config.CondChannels != structure.CondChannels)
                throw new HartsyInferenceException($"'{path}' is not a {Trellis2PipelineConfig.ShapeLatentChannels}-channel shape flow.");
        }

        if (coarse.Resolution != StructureGrid)
            throw new HartsyInferenceException($"'{files.ShapeFlow512}' is trained for a {coarse.Resolution} grid; expected {StructureGrid}.");
    }

    // The texture flows take the shape latent as extra input channels and predict a 32-channel latent on the same grids.
    private static void CheckTextureFlowConfigs(Trellis2ModelFiles files, int condChannels)
    {
        int latent = Trellis2PipelineConfig.ShapeLatentChannels;
        foreach ((string? path, int grid) in new[] { (files.TextureFlow512, StructureGrid), (files.TextureFlow1024, FineLatentGrid) })
        {
            if (path is null)
                continue;
            Trellis2FlowConfig config = ReadConfig(path);
            if (config.Dense || config.OutChannels != latent || config.InChannels != 2 * latent || config.CondChannels != condChannels)
            {
                throw new HartsyInferenceException(
                    $"'{path}' is not a texture flow ({2 * latent} in, {latent} out channels, {condChannels} conditioning channels).");
            }

            if (config.Resolution != grid)
                throw new HartsyInferenceException($"'{path}' is trained for a {config.Resolution} grid; expected {grid}.");
        }
    }

    private static Trellis2FlowConfig ReadConfig(string safetensorsPath)
    {
        string sidecar = Path.ChangeExtension(safetensorsPath, ".json");
        if (!File.Exists(sidecar))
            throw new FileNotFoundException($"TRELLIS.2 flow config not found beside the checkpoint: '{sidecar}'.", sidecar);
        return Trellis2FlowConfig.FromJson(File.ReadAllText(sidecar));
    }
}
