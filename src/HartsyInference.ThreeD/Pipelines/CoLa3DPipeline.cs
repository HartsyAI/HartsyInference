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
    public static CoLa3DPipeline LoadFromPath(IBackend backend, string modelRoot)
    {
        ArgumentNullException.ThrowIfNull(backend);
        if (!Directory.Exists(modelRoot))
            throw new DirectoryNotFoundException($"CoLa3D model directory not found: '{modelRoot}'.");

        CoLa3DWeights weights = CoLa3DCheckpointConverter.Load(modelRoot);
        ShapeVaeEncoderStage encoder = null!;
        ShapeVaeDecoderStage decoder = null!;
        UtoniaPointFeatureStage features = null!;
        CoLa3DObjectDecomposer decomposer = null!;
        try
        {
            CoLa3DConfig config = CoLa3DConfig.FreeQ20;
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
    /// <param name="onProgress">Reports once the scene is split into objects, then once per decoded object.</param>
    /// <exception cref="HartsyInferenceException">The scene decomposes into no object with triangles.</exception>
    public CoLa3DResult Generate(Mesh scene, int resolution = 0, Action<GenerationProgress>? onProgress = null)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(scene);
        int res = resolution > 0 ? resolution : DefaultResolution;
        if (res % LatentStride != 0)
            throw new ArgumentException($"Grid resolution {res} must be a multiple of {LatentStride}.", nameof(resolution));
        if (res > MaxResolution)
            throw new ArgumentException($"Grid resolution {res} exceeds the maximum of {MaxResolution}.", nameof(resolution));
        long started = MonotonicClock.NowNs();
        SceneGrid grid = SceneGrid.Fit(scene, res);

        DualGridVoxels voxels = _voxelizer.Voxelize(scene, grid);
        if (voxels.Coords.Length == 0)
            throw new HartsyInferenceException("Voxelizing the scene produced no occupied voxels.");

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
            objects = _decomposer.Decompose(Backend, latent, features);
        }
        finally
        {
            latent.Feats.Dispose();
            features.Feats.Dispose();
        }

        List<Mesh> meshes = [];
        try
        {
            onProgress?.Invoke(new GenerationProgress(1, 1 + objects.Count, ElapsedMs(started)));
            for (int k = 0; k < objects.Count; k++)
            {
                Mesh mesh = _mesher.Extract(_decoder.Decode(Backend, objects[k]), grid);
                objects[k].Feats.Dispose();
                if (mesh.TriangleCount > 0)
                    meshes.Add(mesh);
                onProgress?.Invoke(new GenerationProgress(2 + k, 1 + objects.Count, ElapsedMs(started)));
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

    private static double ElapsedMs(long startedNs) => (MonotonicClock.NowNs() - startedNs) / 1e6;

    /// <inheritdoc/>
    protected override void DisposeCore()
    {
        foreach (IDisposable disposable in _owned)
            disposable.Dispose();
        _owned.Clear();
    }
}
