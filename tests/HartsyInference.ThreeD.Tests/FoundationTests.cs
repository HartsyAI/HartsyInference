using System.Buffers.Binary;
using System.Text;
using System.Text.Json;
using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Geometry.Ops;
using HartsyInference.ThreeD.Io;
using Xunit;

namespace HartsyInference.ThreeD.Tests;

/// <summary>CPU-only structural tests for the representation-agnostic 3D foundation (no GPU, no checkpoint):
/// marching cubes correctness/watertightness on an analytic sphere, exporter round-trips, and grid sampling.</summary>
public sealed class FoundationTests
{
    private static ScalarField3D SphereSdf(int res, float radius)
    {
        float[] vals = new float[res * res * res];
        // Box [-1,1]^3; SDF = |p| - radius (negative inside).
        for (int z = 0; z < res; z++)
        for (int y = 0; y < res; y++)
        for (int x = 0; x < res; x++)
        {
            float px = -1f + 2f * x / (res - 1);
            float py = -1f + 2f * y / (res - 1);
            float pz = -1f + 2f * z / (res - 1);
            vals[x + res * (y + res * z)] = MathF.Sqrt(px * px + py * py + pz * pz) - radius;
        }
        return new ScalarField3D
        {
            Values = vals, ResX = res, ResY = res, ResZ = res,
            Min = (-1f, -1f, -1f), Max = (1f, 1f, 1f),
        };
    }

    [Fact]
    public void MarchingCubes_Sphere_ProducesWatertightMeshOnSurface()
    {
        ScalarField3D field = SphereSdf(48, 0.6f);
        Mesh mesh = MarchingCubes.Extract(field, isoLevel: 0f);

        Assert.True(mesh.VertexCount > 0);
        Assert.True(mesh.TriangleCount > 0);

        // Watertight: every undirected edge is shared by exactly two triangles (closed manifold).
        Dictionary<(int, int), int> edges = new();
        int[] idx = mesh.Indices;
        for (int t = 0; t < idx.Length; t += 3)
        {
            AddEdge(edges, idx[t], idx[t + 1]);
            AddEdge(edges, idx[t + 1], idx[t + 2]);
            AddEdge(edges, idx[t + 2], idx[t]);
        }
        Assert.All(edges.Values, c => Assert.Equal(2, c));

        // Vertices lie on the iso-surface |p| ≈ radius (within one grid cell).
        float cell = 2f / (48 - 1);
        float[] v = mesh.Vertices;
        for (int i = 0; i < v.Length; i += 3)
        {
            float r = MathF.Sqrt(v[i] * v[i] + v[i + 1] * v[i + 1] + v[i + 2] * v[i + 2]);
            Assert.InRange(r, 0.6f - cell, 0.6f + cell);
        }
    }

    [Fact]
    public void MeshOps_ComputeNormals_PointOutwardOnSphere()
    {
        Mesh mesh = MeshOps.ComputeVertexNormals(MarchingCubes.Extract(SphereSdf(40, 0.6f), 0f));
        Assert.NotNull(mesh.Normals);
        float[] v = mesh.Vertices; float[] n = mesh.Normals!;
        int outward = 0, total = 0;
        for (int i = 0; i < v.Length; i += 3)
        {
            float dot = v[i] * n[i] + v[i + 1] * n[i + 1] + v[i + 2] * n[i + 2]; // n · radial-dir (same sign as |p|·cos)
            if (dot > 0) outward++;
            total++;
        }
        Assert.True(outward > total * 0.95, $"{outward}/{total} normals outward");
    }

    [Fact]
    public void GlbWriter_ProducesValidGltfHeaderAndJson()
    {
        Mesh mesh = MeshOps.ComputeVertexNormals(MarchingCubes.Extract(SphereSdf(24, 0.6f), 0f));
        byte[] glb = GlbWriter.Write(mesh);

        Assert.Equal(0x46546C67u, BinaryPrimitives.ReadUInt32LittleEndian(glb.AsSpan(0)));   // "glTF"
        Assert.Equal(2u, BinaryPrimitives.ReadUInt32LittleEndian(glb.AsSpan(4)));            // version
        Assert.Equal((uint)glb.Length, BinaryPrimitives.ReadUInt32LittleEndian(glb.AsSpan(8)));

        uint jsonLen = BinaryPrimitives.ReadUInt32LittleEndian(glb.AsSpan(12));
        Assert.Equal(0x4E4F534Au, BinaryPrimitives.ReadUInt32LittleEndian(glb.AsSpan(16)));  // "JSON"
        Assert.Equal(0, (int)(jsonLen % 4));
        string json = Encoding.UTF8.GetString(glb, 20, (int)jsonLen);
        using JsonDocument doc = JsonDocument.Parse(json);
        JsonElement accessors = doc.RootElement.GetProperty("accessors");
        Assert.Equal(3, accessors.GetArrayLength()); // POSITION, NORMAL, indices

        // BIN chunk after the JSON chunk; type tag "BIN\0".
        int binTagPos = 20 + (int)jsonLen + 4;
        Assert.Equal(0x004E4942u, BinaryPrimitives.ReadUInt32LittleEndian(glb.AsSpan(binTagPos)));
    }

    [Fact]
    public void ObjAndPlyWriters_RoundTripCounts()
    {
        Mesh mesh = MarchingCubes.Extract(SphereSdf(20, 0.6f), 0f);
        string obj = ObjWriter.Write(mesh);
        int vLines = obj.Split('\n').Count(l => l.StartsWith("v ", StringComparison.Ordinal));
        int fLines = obj.Split('\n').Count(l => l.StartsWith("f ", StringComparison.Ordinal));
        Assert.Equal(mesh.VertexCount, vLines);
        Assert.Equal(mesh.TriangleCount, fLines);

        string ply = PlyWriter.WriteMesh(mesh);
        Assert.Contains($"element vertex {mesh.VertexCount}", ply);
        Assert.Contains($"element face {mesh.TriangleCount}", ply);
    }

    [Fact]
    public void GridSampler_Trilinear_MatchesHandComputed()
    {
        // 2x2x2 grid, value = x index (0 or 1). Sampling u along X should be linear 0→1.
        float[] g = new float[8];
        for (int z = 0; z < 2; z++) for (int y = 0; y < 2; y++) for (int x = 0; x < 2; x++) g[x + 2 * (y + 2 * z)] = x;
        Assert.Equal(0.0f, GridSampler.Trilinear(g, 2, 2, 2, 0f, 0.3f, 0.7f), 5);
        Assert.Equal(0.25f, GridSampler.Trilinear(g, 2, 2, 2, 0.25f, 0.5f, 0.5f), 5);
        Assert.Equal(1.0f, GridSampler.Trilinear(g, 2, 2, 2, 1f, 0.1f, 0.9f), 5);
    }

    private static void AddEdge(Dictionary<(int, int), int> edges, int a, int b)
    {
        (int, int) key = a < b ? (a, b) : (b, a);
        edges[key] = edges.GetValueOrDefault(key) + 1;
    }
}
