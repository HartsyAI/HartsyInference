using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
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
    /// <summary>Grid used when a request names none: the 1024³ dual grid the shape VAE was trained on.</summary>
    public const int DefaultResolution = 1024;

    /// <summary>Downsampling from the dual grid to the latent grid: the shape VAE's four 2x stages, which Utonia matches.</summary>
    public const int LatentStride = 16;

    private readonly ISceneVoxelizer _voxelizer;
    private readonly ISparseShapeEncoder _encoder;
    private readonly IPointFeatureExtractor _features;
    private readonly IObjectDecomposer _decomposer;
    private readonly ISparseShapeDecoder _decoder;
    private readonly IDualGridMesher _mesher;
    private readonly List<IDisposable> _owned;

    /// <param name="owned">Disposables the pipeline releases with itself: the weight sets and any stage it was given that holds weights.</param>
    public CoLa3DPipeline(
        IBackend backend,
        ISceneVoxelizer voxelizer,
        ISparseShapeEncoder encoder,
        IPointFeatureExtractor features,
        IObjectDecomposer decomposer,
        ISparseShapeDecoder decoder,
        IDualGridMesher mesher,
        IEnumerable<IDisposable> owned) : base(backend)
    {
        ArgumentNullException.ThrowIfNull(voxelizer);
        ArgumentNullException.ThrowIfNull(encoder);
        ArgumentNullException.ThrowIfNull(features);
        ArgumentNullException.ThrowIfNull(decomposer);
        ArgumentNullException.ThrowIfNull(decoder);
        ArgumentNullException.ThrowIfNull(mesher);
        ArgumentNullException.ThrowIfNull(owned);
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
        ShapeVaeEncoderStage encoder = null!;
        ShapeVaeDecoderStage decoder = null!;
        UtoniaPointFeatureStage features = null!;
        CoLa3DObjectDecomposer decomposer = null!;
        try
        {
            CoLa3DConfig config = CoLa3DConfig.ForQuerySource(source);
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
    /// <param name="onProgress">Reports one step per stage and per decoded object.</param>
    /// <param name="prompt">Objects to decompose, as masks or points in <paramref name="scene"/>'s units. Null for free queries;
    /// required by a prompt checkpoint and rejected by the free one. Point prompts need the 1024 grid.</param>
    public CoLa3DResult Generate(Mesh scene, int resolution = 0, Action<GenerationProgress>? onProgress = null, CoLa3DPrompt? prompt = null)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(scene);
        int res = resolution > 0 ? resolution : DefaultResolution;
        if (res % LatentStride != 0)
            throw new ArgumentException($"Grid resolution {res} must be a multiple of {LatentStride}.", nameof(resolution));
        prompt?.Validate();
        SceneGrid grid = SceneGrid.Fit(scene, res);

        DualGridVoxels voxels = _voxelizer.Voxelize(scene, grid);
        if (voxels.Coords.Length == 0)
            throw new HartsyInferenceException("Voxelizing the scene produced no occupied voxels.");
        onProgress?.Invoke(new GenerationProgress(1, 3, 0));

        SparseTensor latent = _encoder.Encode(Backend, voxels, res);
        SparseTensor features;
        try
        {
            features = _features.Extract(Backend, SparseCoords.WithBatch(voxels.Coords), res);
        }
        catch
        {
            latent.Feats.Dispose();
            throw;
        }

        IReadOnlyList<SparseTensor> objects;
        try
        {
            using CoLa3DPromptInputs? inputs = prompt is null ? null : CoLa3DPromptInputs.Create(prompt, grid, latent);
            objects = _decomposer.Decompose(Backend, latent, features, inputs?.Prompts);
        }
        finally
        {
            latent.Feats.Dispose();
            features.Feats.Dispose();
        }
        onProgress?.Invoke(new GenerationProgress(2, 3, 0));

        List<Mesh> meshes = [];
        for (int k = 0; k < objects.Count; k++)
        {
            try
            {
                Mesh mesh = _mesher.Extract(_decoder.Decode(Backend, objects[k]), grid);
                if (mesh.TriangleCount > 0)
                    meshes.Add(mesh);
            }
            finally
            {
                objects[k].Feats.Dispose();
            }

            onProgress?.Invoke(new GenerationProgress(3, 3, k + 1));
        }

        return new CoLa3DResult { Objects = meshes };
    }

    /// <inheritdoc/>
    protected override void DisposeCore()
    {
        foreach (IDisposable disposable in _owned)
            disposable.Dispose();
        _owned.Clear();
    }
}
