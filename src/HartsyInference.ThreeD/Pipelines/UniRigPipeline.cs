using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.Diffusion.Requests;
using HartsyInference.ModelAssets.PyTorch;
using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Io;
using HartsyInference.ThreeD.Models.UniRig;

namespace HartsyInference.ThreeD.Pipelines;

/// <summary>
/// UniRig: a triangle mesh becomes a rigged GLB. Stage 1 samples a bone tree from points on the surface, stage 2 predicts per-vertex
/// skin weights for it, and the result is written with up to four influences per vertex. Each stage's checkpoint is read, used and
/// released in turn, so only one of the two (1.4 GB and 4.4 GB of F32) is resident at a time.
/// </summary>
public sealed class UniRigPipeline : ThreeDPipelineBase
{
    /// <summary>Checkpoint of the skeleton stage, relative to the model directory.</summary>
    public const string SkeletonCheckpoint = "skeleton/articulation-xl_quantization_256/model.ckpt";

    /// <summary>Checkpoint of the skin stage, relative to the model directory.</summary>
    public const string SkinCheckpoint = "skin/articulation-xl/model.ckpt";

    /// <summary>Points sampled for the skeleton stage.</summary>
    public const int SkeletonPoints = 65536;

    /// <summary>Of those, how many are mesh vertices rather than surface samples.</summary>
    public const int SkeletonVertexSamples = 8192;

    private const int Steps = 4;

    private readonly SemaphoreSlim _run = new(1, 1);
    private readonly string _skeletonPath;
    private readonly string _skinPath;

    /// <param name="modelDirectory">Folder holding <see cref="SkeletonCheckpoint"/> and <see cref="SkinCheckpoint"/>.</param>
    public UniRigPipeline(IBackend backend, string modelDirectory) : base(backend)
    {
        ArgumentException.ThrowIfNullOrEmpty(modelDirectory);
        _skeletonPath = Path.Combine(modelDirectory, SkeletonCheckpoint);
        _skinPath = Path.Combine(modelDirectory, SkinCheckpoint);
        foreach (string path in new[] { _skeletonPath, _skinPath })
        {
            if (!RepackedCheckpoint.Exists(path))
            {
                throw new FileNotFoundException($"UniRig checkpoint not found: {path}. Expected the VAST-AI/UniRig layout under '{modelDirectory}'.");
            }
        }
    }

    /// <summary>Checks a request before any weights load; throws for input the pipeline cannot rig.</summary>
    public static void ValidateMesh(Mesh mesh)
    {
        ArgumentNullException.ThrowIfNull(mesh);
        if (mesh.VertexCount == 0 || mesh.TriangleCount == 0)
        {
            throw new HartsyInferenceException("UniRig needs a mesh with at least one triangle.");
        }
    }

    /// <summary>Rigs <paramref name="mesh"/>.</summary>
    /// <param name="seed">Seed of point sampling and token sampling; <c>null</c> draws one.</param>
    public UniRigResult Generate(Mesh mesh, int? seed = null, Action<GenerationProgress>? onProgress = null, CancellationToken cancel = default)
    {
        ThrowIfDisposed();
        ValidateMesh(mesh);
        int resolvedSeed = seed ?? Random.Shared.Next();

        // One run at a time: each run holds a stage's checkpoint (up to 4.4 GB) while it works.
        _run.Wait(cancel);
        try
        {
            return Run(mesh, resolvedSeed, onProgress, cancel);
        }
        finally
        {
            _run.Release();
        }
    }

    private UniRigResult Run(Mesh mesh, int resolvedSeed, Action<GenerationProgress>? onProgress, CancellationToken cancel)
    {
        Stopwatch clock = Stopwatch.StartNew();

        UniRigPreparedMesh prepared = UniRigMeshPrep.Prepare(mesh);
        Report(onProgress, 1, clock, cancel);

        UniRigSkeleton skeleton;
        using (UniRigCheckpoint checkpoint = UniRigCheckpoint.Load(_skeletonPath))
        using (UniRigSkeletonModel model = new(Backend, checkpoint))
        {
            UniRigPointSample sample = UniRigSurfaceSampler.Sample(prepared, SkeletonPoints, SkeletonVertexSamples, resolvedSeed);
            skeleton = model.Generate(sample.Points, sample.Normals, new UniRigSamplingOptions { Seed = resolvedSeed }, cancel);
        }

        Report(onProgress, 2, clock, cancel);

        UniRigSkinResult skin;
        using (UniRigCheckpoint checkpoint = UniRigCheckpoint.Load(_skinPath))
        using (UniRigSkinModel model = new(Backend, checkpoint))
        {
            skin = UniRigSkinner.Skin(model, prepared, skeleton, seed: resolvedSeed, cancel: cancel);
        }

        Report(onProgress, 3, clock, cancel);

        UniRigSkeleton inSource = ToSource(skeleton, prepared);
        byte[] glb = WriteGlb(mesh, prepared, inSource, skin);
        Report(onProgress, 4, clock, cancel);
        return new UniRigResult { Glb = glb, Skeleton = inSource, Seed = resolvedSeed };
    }

    /// <summary>The skeleton in the input mesh's coordinates: <c>normalized * Scale + Center</c> for heads and tails.</summary>
    internal static UniRigSkeleton ToSource(UniRigSkeleton skeleton, UniRigPreparedMesh prepared) =>
        skeleton with { Joints = prepared.ToSource(skeleton.Joints), Tails = prepared.ToSource(skeleton.Tails) };

    private static byte[] WriteGlb(Mesh mesh, UniRigPreparedMesh prepared, UniRigSkeleton skeleton, UniRigSkinResult skin)
    {
        SkinnedBone[] bones = new SkinnedBone[skeleton.JointCount];
        for (int i = 0; i < bones.Length; i++)
        {
            bones[i] = new SkinnedBone(
                skeleton.Names[i],
                skeleton.Parents[i],
                (skeleton.Joints[3 * i], skeleton.Joints[3 * i + 1], skeleton.Joints[3 * i + 2]),
                (skeleton.Tails[3 * i], skeleton.Tails[3 * i + 1], skeleton.Tails[3 * i + 2]));
        }

        (int[] welded, float[] weldedWeights) = skin.Limit(4);
        int[] joints = new int[4 * mesh.VertexCount];
        float[] weights = new float[4 * mesh.VertexCount];
        for (int v = 0; v < mesh.VertexCount; v++)
        {
            int source = prepared.SourceVertex[v];
            if (source < 0)
            {
                weights[4 * v] = 1f;
                continue;
            }

            Array.Copy(welded, 4 * source, joints, 4 * v, 4);
            Array.Copy(weldedWeights, 4 * source, weights, 4 * v, 4);
        }

        return SkinnedGlbWriter.Write(mesh, bones, joints, weights);
    }

    private static void Report(Action<GenerationProgress>? onProgress, int step, Stopwatch clock, CancellationToken cancel)
    {
        cancel.ThrowIfCancellationRequested();
        onProgress?.Invoke(new GenerationProgress(step, Steps, clock.Elapsed.TotalMilliseconds));
    }
}
