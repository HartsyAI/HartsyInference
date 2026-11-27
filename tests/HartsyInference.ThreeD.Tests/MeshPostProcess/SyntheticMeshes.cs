using HartsyInference.ThreeD.Geometry;

namespace HartsyInference.ThreeD.Tests.MeshPostProcess;

/// <summary>Procedural meshes and topology checks shared by the mesh post-processing tests. Everything here is synthetic.</summary>
internal static class SyntheticMeshes
{
    /// <summary>Unit-radius-scaled icosphere with <c>20 * 4^subdivisions</c> outward-wound faces.</summary>
    public static Mesh Icosphere(int subdivisions, float radius = 1f)
    {
        float t = (1f + MathF.Sqrt(5f)) / 2f;
        List<float[]> verts =
        [
            [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1],
            [-t, 0, -1], [-t, 0, 1],
        ];
        List<int> idx =
        [
            0, 11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11, 1, 5, 9, 5, 11, 4, 11, 10, 2, 10, 7, 6, 7, 1, 8, 3, 9, 4, 3, 4, 2, 3, 2, 6,
            3, 6, 8, 3, 8, 9, 4, 9, 5, 2, 4, 11, 6, 2, 10, 8, 6, 7, 9, 8, 1,
        ];
        for (int s = 0; s < subdivisions; s++)
        {
            Dictionary<long, int> mid = [];
            List<int> next = [];
            for (int f = 0; f < idx.Count; f += 3)
            {
                int a = idx[f], b = idx[f + 1], c = idx[f + 2];
                int ab = Midpoint(verts, mid, a, b), bc = Midpoint(verts, mid, b, c), ca = Midpoint(verts, mid, c, a);
                next.AddRange([a, ab, ca, b, bc, ab, c, ca, bc, ab, bc, ca]);
            }
            idx = next;
        }
        float[] flat = new float[verts.Count * 3];
        for (int i = 0; i < verts.Count; i++)
        {
            float len = MathF.Sqrt(verts[i][0] * verts[i][0] + verts[i][1] * verts[i][1] + verts[i][2] * verts[i][2]);
            for (int k = 0; k < 3; k++) flat[i * 3 + k] = verts[i][k] / len * radius;
        }
        return new Mesh { Vertices = flat, Indices = [.. idx] };
    }

    /// <summary>Torus around the z axis with <paramref name="nu"/> x <paramref name="nv"/> quads, outward-wound.</summary>
    public static Mesh Torus(int nu, int nv, float major = 1f, float minor = 0.35f)
    {
        float[] verts = new float[nu * nv * 3];
        for (int i = 0; i < nu; i++)
            for (int j = 0; j < nv; j++)
            {
                float u = 2 * MathF.PI * i / nu, v = 2 * MathF.PI * j / nv;
                float r = major + minor * MathF.Cos(v);
                int o = (i * nv + j) * 3;
                verts[o] = r * MathF.Cos(u); verts[o + 1] = r * MathF.Sin(u); verts[o + 2] = minor * MathF.Sin(v);
            }
        List<int> idx = new(nu * nv * 6);
        for (int i = 0; i < nu; i++)
            for (int j = 0; j < nv; j++)
            {
                int a = i * nv + j, b = ((i + 1) % nu) * nv + j, c = ((i + 1) % nu) * nv + (j + 1) % nv, d = i * nv + (j + 1) % nv;
                idx.AddRange([a, b, c, a, c, d]);
            }
        return new Mesh { Vertices = verts, Indices = [.. idx] };
    }

    /// <summary>Cube [-0.5,0.5]^3 whose six faces are each an <paramref name="n"/> x <paramref name="n"/> grid, outward-wound, welded.</summary>
    public static Mesh Cube(int n)
    {
        List<float> verts = [];
        List<int> idx = [];
        Dictionary<(int, int, int), int> ids = [];
        int Vertex(int x, int y, int z)
        {
            if (!ids.TryGetValue((x, y, z), out int id))
            {
                id = verts.Count / 3;
                ids[(x, y, z)] = id;
                verts.AddRange([(float)x / n - 0.5f, (float)y / n - 0.5f, (float)z / n - 0.5f]);
            }
            return id;
        }
        for (int axis = 0; axis < 3; axis++)
            for (int side = 0; side < 2; side++)
                for (int i = 0; i < n; i++)
                    for (int j = 0; j < n; j++)
                    {
                        int[] p00 = Face(axis, side, i, j, n), p10 = Face(axis, side, i + 1, j, n);
                        int[] p11 = Face(axis, side, i + 1, j + 1, n), p01 = Face(axis, side, i, j + 1, n);
                        int a = Vertex(p00[0], p00[1], p00[2]), b = Vertex(p10[0], p10[1], p10[2]);
                        int c = Vertex(p11[0], p11[1], p11[2]), d = Vertex(p01[0], p01[1], p01[2]);
                        // (axis, side) with i along the next axis and j along the one after: even permutation points +axis.
                        if (side == 1) idx.AddRange([a, b, c, a, c, d]); else idx.AddRange([a, c, b, a, d, c]);
                    }
        return new Mesh { Vertices = [.. verts], Indices = [.. idx] };
    }

    /// <summary>Flat <paramref name="n"/> x <paramref name="n"/> grid over [0,1]^2 at z = 0, facing +z.</summary>
    public static Mesh Grid(int n)
    {
        float[] verts = new float[(n + 1) * (n + 1) * 3];
        for (int y = 0; y <= n; y++)
            for (int x = 0; x <= n; x++)
            {
                int o = (y * (n + 1) + x) * 3;
                verts[o] = (float)x / n; verts[o + 1] = (float)y / n;
            }
        List<int> idx = [];
        for (int y = 0; y < n; y++)
            for (int x = 0; x < n; x++)
            {
                int a = y * (n + 1) + x, b = a + 1, c = a + n + 2, d = a + n + 1;
                idx.AddRange([a, b, c, a, c, d]);
            }
        return new Mesh { Vertices = verts, Indices = [.. idx] };
    }

    private static int[] Face(int axis, int side, int i, int j, int n)
    {
        int[] p = new int[3];
        p[axis] = side * n;
        p[(axis + 1) % 3] = i;
        p[(axis + 2) % 3] = j;
        return p;
    }

    private static int Midpoint(List<float[]> verts, Dictionary<long, int> cache, int a, int b)
    {
        long key = a < b ? ((long)a << 32) | (uint)b : ((long)b << 32) | (uint)a;
        if (cache.TryGetValue(key, out int id)) return id;
        id = verts.Count;
        verts.Add([(verts[a][0] + verts[b][0]) / 2, (verts[a][1] + verts[b][1]) / 2, (verts[a][2] + verts[b][2]) / 2]);
        cache[key] = id;
        return id;
    }

    /// <summary>Counts of edges by how many faces use them, and whether directed edges are unique (consistent winding).</summary>
    public static (int Edges, int Boundary, int NonManifold, bool ConsistentWinding) Topology(Mesh mesh)
    {
        Dictionary<(int, int), int> undirected = [];
        HashSet<(int, int)> directed = [];
        bool consistent = true;
        for (int f = 0; f < mesh.TriangleCount; f++)
            for (int c = 0; c < 3; c++)
            {
                int a = mesh.Indices[f * 3 + c], b = mesh.Indices[f * 3 + (c + 1) % 3];
                consistent &= directed.Add((a, b));
                (int, int) key = a < b ? (a, b) : (b, a);
                undirected[key] = undirected.GetValueOrDefault(key) + 1;
            }
        int boundary = 0, nonManifold = 0;
        foreach (int count in undirected.Values)
        {
            if (count == 1) boundary++;
            else if (count > 2) nonManifold++;
        }
        return (undirected.Count, boundary, nonManifold, consistent);
    }

    /// <summary>Euler characteristic V - E + F over the vertices that faces reference.</summary>
    public static int EulerCharacteristic(Mesh mesh)
    {
        HashSet<int> used = [.. mesh.Indices];
        return used.Count - Topology(mesh).Edges + mesh.TriangleCount;
    }

    /// <summary>True for a watertight, consistently wound, two-manifold-edge mesh.</summary>
    public static bool IsClosedManifold(Mesh mesh)
    {
        (int _, int boundary, int nonManifold, bool consistent) = Topology(mesh);
        return boundary == 0 && nonManifold == 0 && consistent;
    }

    /// <summary>Signed volume by the divergence theorem; positive for outward winding.</summary>
    public static double SignedVolume(Mesh mesh)
    {
        double volume = 0;
        float[] v = mesh.Vertices;
        for (int f = 0; f < mesh.TriangleCount; f++)
        {
            int a = mesh.Indices[f * 3] * 3, b = mesh.Indices[f * 3 + 1] * 3, c = mesh.Indices[f * 3 + 2] * 3;
            volume += (v[a] * (v[b + 1] * v[c + 2] - v[b + 2] * v[c + 1]) - v[a + 1] * (v[b] * v[c + 2] - v[b + 2] * v[c])
                + v[a + 2] * (v[b] * v[c + 1] - v[b + 1] * v[c])) / 6.0;
        }
        return volume;
    }

    /// <summary>Axis-aligned bounds (min xyz, max xyz) over referenced vertices.</summary>
    public static (double[] Min, double[] Max) Bounds(Mesh mesh)
    {
        double[] min = [double.MaxValue, double.MaxValue, double.MaxValue];
        double[] max = [double.MinValue, double.MinValue, double.MinValue];
        foreach (int i in new HashSet<int>(mesh.Indices))
            for (int a = 0; a < 3; a++)
            {
                min[a] = Math.Min(min[a], mesh.Vertices[i * 3 + a]);
                max[a] = Math.Max(max[a], mesh.Vertices[i * 3 + a]);
            }
        return (min, max);
    }
}
