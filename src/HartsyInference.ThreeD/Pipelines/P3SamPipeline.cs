using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Requests;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Geometry.Ops;
using HartsyInference.ThreeD.Io;
using HartsyInference.ThreeD.Models.Hunyuan3D.Part;

namespace HartsyInference.ThreeD.Pipelines;

/// <summary>The parts of a mesh found by P3-SAM: the mesh as segmented (cleaned, so its faces are not the input's), one label per face, the labels of the parts with their
/// boxes, and a GLB with one coloured node per part.</summary>
/// <param name="Mesh">The cleaned mesh the labels refer to.</param>
/// <param name="FaceIds">One label per triangle of <paramref name="Mesh"/>; -1 for a face with no mask and -2 for one with no sampled point.</param>
/// <param name="PartIds">The labels of the parts, ascending.</param>
/// <param name="Boxes">Per part, <c>[minX, minY, minZ, maxX, maxY, maxZ]</c> over the vertices of its faces, in the order of <paramref name="PartIds"/>.</param>
/// <param name="Glb">The mesh split into one coloured node per part.</param>
public sealed record P3SamResult(Mesh Mesh, int[] FaceIds, int[] PartIds, float[][] Boxes, byte[] Glb);

/// <summary>P3-SAM (Hunyuan3D-Part): a triangle mesh is segmented into parts. 100,000 surface points get Sonata features, 400 farthest-point prompts each predict
/// three masks and their IoUs, and the masks become per-face part labels (<see cref="P3SamAutoMask"/>).</summary>
public sealed class P3SamPipeline : IDisposable
{
    /// <summary>The checkpoint file the pipeline reads (<c>tencent/Hunyuan3D-Part</c>, <c>p3sam/p3sam.safetensors</c>).</summary>
    public const string Checkpoint = "p3sam.safetensors";

    /// <summary>Progress steps: surface points, features, prompts, masks, labels.</summary>
    public const int Steps = 5;

    private readonly IBackend _backend;
    private readonly SafeTensorsLoader _loader;
    private readonly P3SamFeatures _features;
    private readonly P3SamHeads _heads;

    /// <summary>Loads <paramref name="checkpointPath"/> (the <c>p3sam.safetensors</c> file).</summary>
    public P3SamPipeline(IBackend backend, string checkpointPath)
    {
        ArgumentNullException.ThrowIfNull(backend);
        if (!File.Exists(checkpointPath))
            throw new FileNotFoundException($"The P3-SAM checkpoint '{checkpointPath}' was not found.", checkpointPath);
        _backend = backend;
        _loader = new SafeTensorsLoader();
        try
        {
            _loader.Load(checkpointPath);
            Dictionary<string, Tensor> state = _loader.GetAllTensors();
            _features = new P3SamFeatures(backend, state);
            try
            {
                _heads = new P3SamHeads(state);
            }
            catch
            {
                _features.Dispose();
                throw;
            }
        }
        catch
        {
            _loader.Dispose();
            throw;
        }
    }

    /// <summary>Rejects a mesh with no triangles or an index outside its vertices, before any checkpoint loads.</summary>
    public static void ValidateMesh(Mesh mesh)
    {
        ArgumentNullException.ThrowIfNull(mesh);
        if (mesh.TriangleCount == 0 || mesh.Vertices.Length % 3 != 0 || mesh.Indices.Length % 3 != 0)
            throw new HartsyInferenceException("P3-SAM needs a triangle mesh with at least one face.");
        int vertexCount = mesh.VertexCount;
        if (mesh.Indices.Any(i => (uint)i >= (uint)vertexCount))
            throw new HartsyInferenceException("The mesh has a face index outside its vertices.");
    }

    /// <summary>Segments <paramref name="mesh"/>. The mesh is cleaned first the way upstream does (vertices merged, degenerate and duplicate faces and unreferenced
    /// vertices removed), so the result's faces refer to <see cref="P3SamResult.Mesh"/>. <paramref name="pointCount"/> and <paramref name="promptCount"/> default to upstream's
    /// 100,000 and 400; <paramref name="seed"/> fixes the surface sampling (default 42).</summary>
    public P3SamResult Generate(Mesh mesh, int? seed = null, int? pointCount = null, int? promptCount = null, Action<GenerationProgress>? onProgress = null,
        CancellationToken cancel = default)
    {
        ValidateMesh(mesh);
        Stopwatch clock = Stopwatch.StartNew();
        Mesh cleaned = Clean(mesh);
        P3SamAutoMask.Options defaults = new();
        P3SamAutoMask.Options options = defaults with
        {
            Seed = seed ?? defaults.Seed,
            PointCount = pointCount ?? defaults.PointCount,
            PromptCount = promptCount ?? defaults.PromptCount,
        };
        if (options.PointCount < 1 || options.PromptCount < 1)
            throw new ArgumentOutOfRangeException(nameof(pointCount), "P3-SAM needs at least one point and one prompt.");

        int step = 0;
        P3SamAutoMask auto = new(_backend, _features, _heads)
        {
            StageLog = line =>
            {
                // One line per finished stage; the progress lines every 50 prompts start with spaces.
                if (!line.StartsWith(' '))
                    onProgress?.Invoke(new GenerationProgress(++step, Steps, clock.Elapsed.TotalMilliseconds));
            },
        };
        P3SamPartLabeler.Result labels = auto.Run(cleaned.Vertices, cleaned.Indices, options, cancel);
        byte[] glb = PartGlbWriter.Write(cleaned, labels.FaceIds, labels.PartIds, labels.Boxes);
        return new P3SamResult(cleaned, labels.FaceIds, labels.PartIds, labels.Boxes, glb);
    }

    // merge_vertices and process(validate=True): merged vertices, no degenerate or duplicate faces, no unreferenced vertices.
    private static Mesh Clean(Mesh mesh)
    {
        Mesh cleaned = MeshCleaner.Clean(mesh, new MeshCleanOptions
        {
            WeldVertices = true,
            RemoveDegenerateFaces = true,
            RemoveDuplicateFaces = true,
            RepairNonManifoldEdges = false,
            SmallComponentAreaRatio = 0,
            HoleFillMaxPerimeter = 0,
            UnifyOrientation = false,
            OrientOutward = false,
        });
        if (cleaned.TriangleCount == 0)
            throw new HartsyInferenceException("The mesh has no faces left after removing degenerate and duplicate ones.");
        return DropUnreferencedVertices(cleaned);
    }

    internal static Mesh DropUnreferencedVertices(Mesh mesh)
    {
        int[] remap = new int[mesh.VertexCount];
        Array.Fill(remap, -1);
        List<float> vertices = [];
        int[] indices = new int[mesh.Indices.Length];
        for (int i = 0; i < indices.Length; i++)
        {
            int source = mesh.Indices[i];
            if (remap[source] < 0)
            {
                remap[source] = vertices.Count / 3;
                vertices.Add(mesh.Vertices[source * 3]);
                vertices.Add(mesh.Vertices[source * 3 + 1]);
                vertices.Add(mesh.Vertices[source * 3 + 2]);
            }

            indices[i] = remap[source];
        }

        return new Mesh { Vertices = [.. vertices], Indices = indices };
    }

    /// <summary>Releases the checkpoint and the model tensors.</summary>
    public void Dispose()
    {
        _heads.Dispose();
        _features.Dispose();
        _loader.Dispose();
    }
}
