using System.Buffers.Binary;
using System.Numerics;
using System.Text;
using System.Text.Json.Nodes;
using HartsyInference.Core.Exceptions;
using HartsyInference.Engine.Planning;
using HartsyInference.Engine.Requests;
using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Geometry.Ops;
using HartsyInference.ThreeD.Io;

namespace HartsyInference.Engine.Services;

/// <summary>What happens to a model's raw result before a host sees it: the frame, the measurements and the preview.</summary>
internal static class MeshResultFinisher
{
    /// <summary>Models whose export frame is turned half a turn about Y from glTF's: Pixal3D samples with the camera on +Z and exports a half turn, so its front faces -Z.</summary>
    private static readonly HashSet<string> HalfTurnedAboutY = new(StringComparer.OrdinalIgnoreCase) { MeshModelIds.Pixal3D };

    /// <summary>Applies the request's frame, measures the result and makes the preview.</summary>
    public static MeshResult Finish(string id, MeshRequest request, MeshResult raw, MeshSettingsUsed settings, CancellationToken cancel)
    {
        if (raw.Format != "glb")
        {
            byte[] ply = raw.Data;
            List<string> plyWarnings = [.. raw.Warnings];
            try
            {
                ply = MeshMetadata.SetPly(ply, Describe(settings, MeshFrame.Native));
            }
            catch (HartsyInferenceException ex)
            {
                plyWarnings.Add($"The settings were not embedded: {ex.Message}");
            }
            return raw with { Data = ply, Info = SplatInfo(ply), Settings = settings, Warnings = plyWarnings };
        }

        List<string> warnings = [.. raw.Warnings];
        byte[] data = raw.Data;
        MeshFrame frame = MeshFrame.Native;
        if (request.Frame == MeshFrame.GltfYUpMetres)
        {
            try
            {
                data = GlbFrame.Normalize(data, FrameRotation(id));
                frame = MeshFrame.GltfYUpMetres;
                if (!HalfTurnedAboutY.Contains(id))
                    warnings.Add($"'{id}' was centred and scaled to one metre; its front direction is not verified, so it was not turned.");
            }
            catch (HartsyInferenceException ex)
            {
                warnings.Add($"The frame was not normalised: {ex.Message}");
            }
        }
        try
        {
            data = MeshMetadata.SetGlb(data, Describe(settings, frame));
        }
        catch (Exception ex) when (ex is HartsyInferenceException or System.Text.Json.JsonException)
        {
            warnings.Add($"The settings were not embedded: {ex.Message}");
        }
        cancel.ThrowIfCancellationRequested();

        MeshInfo info;
        byte[]? preview = null;
        GlbSummary? summary = SummaryOf(data, warnings);
        Mesh? mesh = ReadMesh(data, warnings);
        info = BuildInfo(data, summary, mesh);
        if (mesh is not null && request.PreviewTriangles > 0 && mesh.TriangleCount > request.PreviewTriangles)
            preview = MakePreview(mesh, request.PreviewTriangles, cancel);
        return raw with { Data = data, Frame = frame, Info = info, Preview = preview, Warnings = warnings, Settings = settings };
    }

    /// <summary>The description of how a result was made, as embedded in its file.</summary>
    internal static JsonObject Describe(MeshSettingsUsed settings, MeshFrame frame)
    {
        JsonObject parameters = [];
        foreach ((string name, ResolvedParameter parameter) in settings.Parameters)
            parameters[name] = new JsonObject { ["value"] = parameter.Value, ["source"] = parameter.Source.ToString() };
        return new JsonObject
        {
            ["generator"] = "HartsyInference " + settings.EngineVersion,
            ["model"] = settings.ModelId,
            ["seed"] = settings.Seed,
            ["frame"] = frame.ToString(),
            ["settings"] = parameters,
        };
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

    private static GlbSummary? SummaryOf(byte[] glb, List<string> warnings)
    {
        try
        {
            return GlbInspector.Read(glb);
        }
        catch (Exception ex) when (ex is HartsyInferenceException or System.Text.Json.JsonException or InvalidOperationException or KeyNotFoundException)
        {
            warnings.Add($"The glb could not be inspected: {ex.Message}");
            return null;
        }
    }

    private static Mesh? ReadMesh(byte[] glb, List<string> warnings)
    {
        try
        {
            return GlbReader.Parse(glb);
        }
        catch (HartsyInferenceException ex)
        {
            warnings.Add($"The topology was not measured and no preview was made: {ex.Message}");
            return null;
        }
    }

    private static MeshInfo BuildInfo(byte[] glb, GlbSummary? summary, Mesh? mesh)
    {
        MeshStats? stats = mesh is null ? null : MeshStatistics.Measure(mesh);
        return new MeshInfo
        {
            ByteSize = glb.LongLength,
            TriangleCount = stats?.TriangleCount ?? 0,
            VertexCount = stats?.VertexCount ?? 0,
            BoundsMin = stats?.Min ?? summary?.Min,
            BoundsMax = stats?.Max ?? summary?.Max,
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

    /// <summary>The splat count and bounds of a Gaussian-splat PLY written by this engine (binary little endian, x y z first).</summary>
    internal static MeshInfo SplatInfo(byte[] ply)
    {
        string marker = "end_header\n";
        int headerEnd = IndexOf(ply, Encoding.ASCII.GetBytes(marker));
        if (headerEnd < 0)
            return new MeshInfo { ByteSize = ply.LongLength };
        string[] header = Encoding.ASCII.GetString(ply, 0, headerEnd).Split('\n');
        int count = 0, floats = 0;
        foreach (string line in header)
        {
            string[] parts = line.Split(' ');
            if (parts is ["element", "vertex", _])
                int.TryParse(parts[2], out count);
            else if (parts is ["property", "float", _])
                floats++;
        }
        int body = headerEnd + marker.Length;
        if (count == 0 || floats < 3 || ply.Length < body + (long)count * floats * 4)
            return new MeshInfo { ByteSize = ply.LongLength, SplatCount = count };
        float[] min = [float.PositiveInfinity, float.PositiveInfinity, float.PositiveInfinity];
        float[] max = [float.NegativeInfinity, float.NegativeInfinity, float.NegativeInfinity];
        for (int i = 0; i < count; i++)
        {
            for (int c = 0; c < 3; c++)
            {
                float value = BinaryPrimitives.ReadSingleLittleEndian(ply.AsSpan(body + (i * floats + c) * 4));
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
