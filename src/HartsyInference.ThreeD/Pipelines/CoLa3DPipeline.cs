using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Runtime;
using HartsyInference.Core.Sparse;
using HartsyInference.Diffusion.Requests;
using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Models.CoLa3D;
using HartsyInference.ThreeD.Models.TrellisV2;
using HartsyInference.ThreeD.Models.Utonia;
using HartsyInference.ThreeD.Pipelines.Stages;

namespace HartsyInference.ThreeD.Pipelines;

/// <summary>
/// CoLa3D scene decomposition: a triangle scene mesh becomes one mesh per object. The stages are: dual-grid voxelization,
/// shape-VAE encoding, Utonia point features, CoLa3D object decomposition, shape-VAE decoding per object, and dual-grid
/// meshing. Each stage sits behind its interface in <c>Pipelines/Stages</c>, so another model can reuse any of them.
/// </summary>
public sealed class CoLa3DPipeline : ThreeDPipelineBase
{
    /// <summary>Grid used when a request names none; the 1024³ grid the shape VAE was trained on is too slow to default to.</summary>
    public const int DefaultResolution = 256;

    /// <summary>Largest accepted grid, the 1024³ the shape VAE was trained on; object keys pack each coordinate into 16 bits.</summary>
    public const int MaxResolution = 1024;

    /// <summary>Downsampling from the dual grid to the latent grid: the shape VAE's four 2x stages, which Utonia matches.</summary>
    public const int LatentStride = 16;

    private readonly CoLa3DConfig _config;

    // Voxelize, encode and extract, and the split; the object count adds to these once the split is known.
    private const int FixedSteps = 3;

    private readonly ISceneVoxelizer _voxelizer;
    private readonly ISparseShapeEncoder _encoder;
    private readonly IPointFeatureExtractor _features;
    private readonly IObjectDecomposer _decomposer;
    private readonly ISparseShapeDecoder _decoder;
    private readonly IDualGridMesher _mesher;
    private readonly List<IDisposable> _owned;

    /// <param name="config">Architecture the decomposer was built with; prompts are checked against it before any stage runs.</param>
    /// <param name="owned">Disposables the pipeline releases with itself: the weight sets and any stage it was given that holds weights.</param>
    public CoLa3DPipeline(
        IBackend backend,
        CoLa3DConfig config,
        ISceneVoxelizer voxelizer,
        ISparseShapeEncoder encoder,
        IPointFeatureExtractor features,
        IObjectDecomposer decomposer,
        ISparseShapeDecoder decoder,
        IDualGridMesher mesher,
        IEnumerable<IDisposable> owned) : base(backend)
    {
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(voxelizer);
        ArgumentNullException.ThrowIfNull(encoder);
        ArgumentNullException.ThrowIfNull(features);
        ArgumentNullException.ThrowIfNull(decomposer);
        ArgumentNullException.ThrowIfNull(decoder);
        ArgumentNullException.ThrowIfNull(mesher);
        ArgumentNullException.ThrowIfNull(owned);
        _config = config;
        _voxelizer = voxelizer;
        _encoder = encoder;
        _features = features;
        _decomposer = decomposer;
        _decoder = decoder;
        _mesher = mesher;
        _owned = [.. owned];
    }

    /// <summary>Loads the pipeline from a Models root holding <c>CoLa3D</c>, <c>TRELLIS2-ShapeVAE</c> and <c>Utonia</c> checkpoints.</summary>
    /// <param name="source">Decomposer variant: free queries, mask prompts or point prompts. Each variant is its own pipeline.</param>
    public static CoLa3DPipeline LoadFromPath(
        IBackend backend,
        string modelRoot,
        CoLa3DConfig.QuerySourceKind source = CoLa3DConfig.QuerySourceKind.Fixed)
    {
        ArgumentNullException.ThrowIfNull(backend);
        if (!Directory.Exists(modelRoot))
            throw new DirectoryNotFoundException($"CoLa3D model directory not found: '{modelRoot}'.");

        CoLa3DWeights weights = CoLa3DCheckpointConverter.Load(modelRoot, source);
        CoLa3DConfig config = null!;
        ShapeVaeEncoderStage encoder = null!;
        ShapeVaeDecoderStage decoder = null!;
        UtoniaPointFeatureStage features = null!;
        CoLa3DObjectDecomposer decomposer = null!;
        try
        {
            config = CoLa3DConfig.ForQuerySource(source);
            ShapeVaeConfig shape = ShapeVaeConfig.NextDcF16C32;
            encoder = new ShapeVaeEncoderStage(new ShapeVaeEncoder(shape.Encoder, weights.ShapeEncoder));
            decoder = new ShapeVaeDecoderStage(new ShapeVaeDecoder(shape.Decoder, weights.ShapeDecoder));
            features = new UtoniaPointFeatureStage(new UtoniaEncoder(backend, UtoniaConfig.Cola3DEncoder, weights.Utonia), CoLa3DPipeline.LatentStride);
            decomposer = new CoLa3DObjectDecomposer(config, new CoLa3DDecomposer(config, weights.Decomposer));
        }
        catch
        {
            encoder?.Dispose();
            decoder?.Dispose();
            features?.Dispose();
            decomposer?.Dispose();
            weights.Dispose();
            throw;
        }

        return new CoLa3DPipeline(
            backend,
            config,
            new DualGridSceneVoxelizer(),
            encoder,
            features,
            decomposer,
            decoder,
            new DualGridMesherStage(),
            [encoder, decoder, features, decomposer, weights]);
    }

    /// <summary>Decomposes a triangle scene mesh into object meshes.</summary>
    /// <param name="scene">Scene triangles in any world units; the grid is fitted to its largest extent.</param>
    /// <param name="resolution">Dual-grid voxels per axis (a multiple of 16); 0 uses <see cref="DefaultResolution"/>.</param>
    /// <param name="onProgress">Reports the three fixed stages, then once per decoded object. TotalSteps is 3 until the
    /// split fixes the object count.</param>
    /// <param name="prompt">Objects to decompose, as masks or points in <paramref name="scene"/>'s units. Null for free queries;
    /// required by a prompt checkpoint and rejected by the free one. Point prompts need the 1024 grid.</param>
    /// <exception cref="HartsyInferenceException">The scene decomposes into no object with triangles.</exception>
    public CoLa3DResult Generate(Mesh scene, int resolution = 0, Action<GenerationProgress>? onProgress = null, CoLa3DPrompt? prompt = null)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(scene);
        ValidateRequest(_config, resolution, prompt);
        int res = resolution > 0 ? resolution : DefaultResolution;
        long started = MonotonicClock.NowNs();
        SceneGrid grid = SceneGrid.Fit(scene, res);

        DualGridVoxels voxels = _voxelizer.Voxelize(scene, grid);
        if (voxels.Coords.Length == 0)
            throw new HartsyInferenceException("Voxelizing the scene produced no occupied voxels.");
        onProgress?.Invoke(new GenerationProgress(1, FixedSteps, ElapsedMs(started)));

        SparseTensor latent = _encoder.Encode(Backend, voxels, res);
        SparseTensor? features = null;
        IReadOnlyList<SparseTensor> objects;
        try
        {
            using CoLa3DPromptInputs? inputs = prompt is null ? null : CoLa3DPromptInputs.Create(prompt, grid, latent);
            features = _features.Extract(Backend, SparseCoords.WithBatch(voxels.Coords), res);
            onProgress?.Invoke(new GenerationProgress(2, FixedSteps, ElapsedMs(started)));
            objects = _decomposer.Decompose(Backend, latent, features, inputs?.Prompts);
        }
        finally
        {
            // A cancel from onProgress or a failed stage releases what was built; Tensor.Dispose is idempotent.
            latent.Feats.Dispose();
            features?.Feats.Dispose();
        }

        List<Mesh> meshes = [];
        int totalSteps = FixedSteps + objects.Count;
        try
        {
            onProgress?.Invoke(new GenerationProgress(FixedSteps, totalSteps, ElapsedMs(started)));
            for (int k = 0; k < objects.Count; k++)
            {
                Mesh mesh = _mesher.Extract(_decoder.Decode(Backend, objects[k]), grid);
                objects[k].Feats.Dispose();
                if (mesh.TriangleCount > 0)
                    meshes.Add(mesh);
                onProgress?.Invoke(new GenerationProgress(FixedSteps + 1 + k, totalSteps, ElapsedMs(started)));
            }
        }
        finally
        {
            // A cancel from onProgress or a failed decode stops the loop; release every object it did not reach.
            foreach (SparseTensor part in objects)
                part.Feats.Dispose();
        }

        if (meshes.Count == 0)
            throw new HartsyInferenceException("The scene decomposed into no object geometry (try a finer grid).");
        return new CoLa3DResult { Objects = meshes };
    }

    /// <summary>Checks the grid and the prompt against a checkpoint's architecture. Loads no weights, so it can reject a request first.</summary>
    /// <param name="config">Architecture of the checkpoint the request targets, from <see cref="CoLa3DConfig.ForQuerySource"/>.</param>
    /// <param name="resolution">Dual-grid voxels per axis (a multiple of 16); 0 uses <see cref="DefaultResolution"/>.</param>
    /// <param name="prompt">Prompt for a prompt checkpoint; null for the free one.</param>
    public static void ValidateRequest(CoLa3DConfig config, int resolution, CoLa3DPrompt? prompt)
    {
        ArgumentNullException.ThrowIfNull(config);
        int res = resolution > 0 ? resolution : DefaultResolution;
        if (res % LatentStride != 0)
            throw new ArgumentException($"Grid resolution {res} must be a multiple of {LatentStride}.", nameof(resolution));
        if (res > MaxResolution)
            throw new ArgumentException($"Grid resolution {res} exceeds the maximum of {MaxResolution}.", nameof(resolution));
        prompt?.Validate();
        if (prompt is null)
        {
            if (config.QuerySource != CoLa3DConfig.QuerySourceKind.Fixed)
                throw new ArgumentException($"This checkpoint takes {config.QuerySource} queries and requires a prompt.", nameof(prompt));
            return;
        }

        if (config.QuerySource == CoLa3DConfig.QuerySourceKind.Fixed)
            throw new ArgumentException("This checkpoint takes free queries; pass no prompt.", nameof(prompt));
        if (prompt.Kind != config.QuerySource)
        {
            throw new ArgumentException(
                $"This checkpoint takes {config.QuerySource} queries; the prompt gives {prompt.Kind}.", nameof(prompt));
        }

        if (prompt.Queries > config.MaxParts)
        {
            throw new ArgumentException(
                $"The prompt names {prompt.Queries} objects; this checkpoint takes at most {config.MaxParts}.", nameof(prompt));
        }

        if (prompt.Kind == CoLa3DConfig.QuerySourceKind.PromptPoint && res / LatentStride != config.LatentResolution)
        {
            int grid = config.LatentResolution * LatentStride;
            throw new ArgumentException(
                $"Point prompts index a {config.LatentResolution}^3 latent, so the grid must be {grid}; this request uses {res}.",
                nameof(prompt));
        }
    }

    private static double ElapsedMs(long startedNs) => (MonotonicClock.NowNs() - startedNs) / 1e6;

    /// <inheritdoc/>
    protected override void DisposeCore()
    {
        foreach (IDisposable disposable in _owned)
            disposable.Dispose();
        _owned.Clear();
    }
}
