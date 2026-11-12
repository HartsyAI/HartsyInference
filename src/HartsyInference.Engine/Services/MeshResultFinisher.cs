using System.Buffers.Binary;
using System.Numerics;
using System.Text;
using HartsyInference.Core.Logging;
using HartsyInference.Engine.Requests;
using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Geometry.Ops;
using HartsyInference.ThreeD.Io;

namespace HartsyInference.Engine.Services;

/// <summary>What happens to a model's raw result before a host sees it: the frame, the measurements and the preview. A finished generation is never lost to these steps: each one that fails leaves the raw bytes, adds a warning and lets the rest run.</summary>
internal static class MeshResultFinisher
{
    /// <summary>Above this many triangles the topology is not measured (the weld holds every vertex in a hash map); the counts and bounds still come from the GLB's accessors.</summary>
    internal const long TopologyTriangleLimit = 4_000_000;

    /// <summary>Models whose export frame is turned half a turn about Y from glTF's: Pixal3D samples with the camera on +Z and exports a half turn, so its front faces -Z.</summary>
    private static readonly HashSet<string> HalfTurnedAboutY = new(StringComparer.OrdinalIgnoreCase) { MeshModelIds.Pixal3D };

    /// <summary>Applies the request's frame, measures the result and makes the preview.</summary>
    public static MeshResult Finish(string id, MeshRequest request, MeshResult raw, CancellationToken cancel)
    {
        if (raw.Format != "glb")
            return raw with { Info = SplatInfo(raw.Data) };

        List<string> warnings = [.. raw.Warnings];
        byte[] data = raw.Data;
        MeshFrame frame = MeshFrame.Native;
        if (request.Frame == MeshFrame.GltfYUpMetres)
        {
            byte[]? framed = Step("The frame was not normalised", warnings, () => GlbFrame.Normalize(raw.Data, FrameRotation(id)));
            if (framed is not null)
            {
                data = framed;
                frame = MeshFrame.GltfYUpMetres;
                if (!HalfTurnedAboutY.Contains(id))
                    warnings.Add($"'{id}' was centred and scaled to one metre; its front direction is not verified, so it was not turned.");
            }
        }
        cancel.ThrowIfCancellationRequested();

        GlbSummary? summary = Step("The glb could not be inspected", warnings, () => GlbInspector.Read(data));
        bool wantsPreview = request.PreviewTriangles > 0 && (summary is null || summary.TriangleCount > request.PreviewTriangles);
        bool measure = summary is null || summary.TriangleCount <= TopologyTriangleLimit;
        if (!measure)
            warnings.Add($"The topology was not measured: the mesh has more than {TopologyTriangleLimit:N0} triangles.");
        Mesh? mesh = measure || wantsPreview ? Step("The mesh could not be read, so its topology was not measured and no preview was made", warnings, () => GlbReader.Parse(data)) : null;
        cancel.ThrowIfCancellationRequested();

        MeshInfo info = BuildInfo(data, summary, mesh, measure, frame);
        byte[]? preview = null;
        if (mesh is not null && wantsPreview && mesh.TriangleCount > request.PreviewTriangles)
            preview = Step("The preview could not be made", warnings, () => MakePreview(mesh, request.PreviewTriangles, cancel));
        return raw with { Data = data, Frame = frame, Info = info, Preview = preview, Warnings = warnings };
    }

    /// <summary>The rotation that takes a model's native glb into glTF's frame; identity where the native front is not verified.</summary>
    internal static Matrix4x4 FrameRotation(string id) => HalfTurnedAboutY.Contains(id) ? Matrix4x4.CreateRotationY(MathF.PI) : Matrix4x4.Identity;

    /// <summary>A geometry-only glb of <paramref name="mesh"/> decimated to about <paramref name="triangles"/>, with normals computed.</summary>
    internal static byte[] MakePreview(Mesh mesh, int triangles, CancellationToken cancel)
    {
        Mesh reduced = MeshDecimator.DecimateStaged(mesh, triangles, cancellationToken: cancel);
        MeshOps.ComputeVertexNormals(reduced);
        return GlbWriter.Write(reduced);
    }

    /// <summary>Runs one finishing step. This is the boundary where a failure must cost a warning and not the finished generation, so every failure but a cancellation is caught, logged and reported.</summary>
    private static T? Step<T>(string what, List<string> warnings, Func<T> step) where T : class
    {
        try
        {
            return step();
        }
        catch (OperationCanceledException)
        {
            throw;
        }
        catch (Exception ex)
        {
            Logs.Warning($"[Mesh] {what}: {ex.Message}");
            warnings.Add($"{what}: {ex.Message}");
            return null;
        }
    }

    private static MeshInfo BuildInfo(byte[] glb, GlbSummary? summary, Mesh? mesh, bool measured, MeshFrame frame)
    {
        MeshStats? stats = mesh is null || !measured ? null : MeshStatistics.Measure(mesh);
        // Accessor bounds are in the file's own space; the frame node moves them, so they are not the bounds of the framed result.
        bool accessorBoundsValid = frame == MeshFrame.Native;
        return new MeshInfo
        {
            ByteSize = glb.LongLength,
            TriangleCount = stats?.TriangleCount ?? mesh?.TriangleCount ?? summary?.TriangleCount ?? 0,
            VertexCount = stats?.VertexCount ?? mesh?.VertexCount ?? summary?.VertexCount ?? 0,
            BoundsMin = stats?.Min ?? (accessorBoundsValid ? summary?.Min : null),
            BoundsMax = stats?.Max ?? (accessorBoundsValid ? summary?.Max : null),
            Components = stats?.Components,
            BoundaryEdges = stats?.BoundaryEdges,
            NonManifoldEdges = stats?.NonManifoldEdges,
            Watertight = stats?.Watertight,
            HasNormals = summary?.HasNormals ?? false,
            HasTexture = summary?.HasTexture ?? false,
            HasSkin = summary?.HasSkin ?? false,
            AnimationNames = summary?.AnimationNames ?? [],
        };
    }

    /// <summary>The splat count and bounds of a binary-little-endian Gaussian-splat PLY: the vertex count and the offsets of the x, y and z properties are read from the header; a header with a non-float property gives the count only.</summary>
    internal static MeshInfo SplatInfo(byte[] ply)
    {
        string marker = "end_header\n";
        int headerEnd = IndexOf(ply, Encoding.ASCII.GetBytes(marker));
        if (headerEnd < 0)
            return new MeshInfo { ByteSize = ply.LongLength };
        int count = 0, properties = 0;
        int[] axis = [-1, -1, -1];
        bool allFloats = true;
        foreach (string line in Encoding.ASCII.GetString(ply, 0, headerEnd).Split('\n'))
        {
            string[] parts = line.Split(' ');
            if (parts is ["element", "vertex", _])
            {
                int.TryParse(parts[2], out count);
            }
            else if (parts is ["property", _, _])
            {
                allFloats &= parts[1] == "float";
                int slot = Array.IndexOf(new[] { "x", "y", "z" }, parts[2]);
                if (slot >= 0)
                    axis[slot] = properties;
                properties++;
            }
        }
        int body = headerEnd + marker.Length;
        if (count == 0 || !allFloats || axis.Any(i => i < 0) || ply.Length < body + (long)count * properties * 4)
            return new MeshInfo { ByteSize = ply.LongLength, SplatCount = count };
        float[] min = [float.PositiveInfinity, float.PositiveInfinity, float.PositiveInfinity];
        float[] max = [float.NegativeInfinity, float.NegativeInfinity, float.NegativeInfinity];
        for (int i = 0; i < count; i++)
        {
            for (int c = 0; c < 3; c++)
            {
                float value = BinaryPrimitives.ReadSingleLittleEndian(ply.AsSpan(body + (i * properties + axis[c]) * 4));
                min[c] = Math.Min(min[c], value);
                max[c] = Math.Max(max[c], value);
            }
        }
        return new MeshInfo { ByteSize = ply.LongLength, SplatCount = count, BoundsMin = min, BoundsMax = max };
    }

    private static int IndexOf(byte[] data, byte[] pattern)
    {
        int limit = Math.Min(data.Length, 1 << 16) - pattern.Length;
        for (int i = 0; i <= limit; i++)
        {
            if (data.AsSpan(i, pattern.Length).SequenceEqual(pattern))
                return i;
        }
        return -1;
    }
}
