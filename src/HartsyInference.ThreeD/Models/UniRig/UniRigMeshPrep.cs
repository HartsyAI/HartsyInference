using HartsyInference.Core.Exceptions;
using HartsyInference.ThreeD.Geometry;

namespace HartsyInference.ThreeD.Models.UniRig;

/// <summary>The mesh preparation upstream gets from Blender and trimesh: vertex welding, normals and normalization into [-1, 1]³.</summary>
public static class UniRigMeshPrep
{
    /// <summary>Welds identical positions, drops non-finite triangles, computes normals and normalizes the bounding box into [-1, 1]³ by its longest side.</summary>
    public static UniRigPreparedMesh Prepare(Mesh mesh)
    {
        ArgumentNullException.ThrowIfNull(mesh);
        if (mesh.TriangleCount == 0 || mesh.VertexCount == 0)
        {
            throw new HartsyInferenceException("UniRig needs a mesh with at least one triangle.");
        }

        (float[] vertices, int[] faces, int[] sourceVertex) = Weld(mesh);
        float[] lo = [float.MaxValue, float.MaxValue, float.MaxValue];
        float[] hi = [float.MinValue, float.MinValue, float.MinValue];
        for (int i = 0; i < vertices.Length; i++)
        {
            lo[i % 3] = Math.Min(lo[i % 3], vertices[i]);
            hi[i % 3] = Math.Max(hi[i % 3], vertices[i]);
        }

        float[] center = [(lo[0] + hi[0]) / 2f, (lo[1] + hi[1]) / 2f, (lo[2] + hi[2]) / 2f];
        float scale = Math.Max(hi[0] - lo[0], Math.Max(hi[1] - lo[1], hi[2] - lo[2])) / 2f;
        if (!(scale > 0f) || float.IsInfinity(scale))
        {
            throw new HartsyInferenceException("UniRig mesh has a degenerate or non-finite bounding box.");
        }

        float[] normalized = new float[vertices.Length];
        for (int i = 0; i < vertices.Length; i++)
        {
            normalized[i] = (vertices[i] - center[i % 3]) / scale;
        }

        (float[] vertexNormals, float[] faceNormals) = Normals(normalized, faces);
        return new UniRigPreparedMesh
        {
            Vertices = normalized,
            Faces = faces,
            VertexNormals = vertexNormals,
            FaceNormals = faceNormals,
            SourceVertex = sourceVertex,
            Center = center,
            Scale = scale,
        };
    }

    private static (float[] Vertices, int[] Faces, int[] SourceVertex) Weld(Mesh mesh)
    {
        Dictionary<(int, int, int), int> index = new(mesh.VertexCount);
        int[] remap = new int[mesh.VertexCount];
        List<float> vertices = new(mesh.Vertices.Length);
        for (int v = 0; v < mesh.VertexCount; v++)
        {
            float x = mesh.Vertices[3 * v] + 0f;
            float y = mesh.Vertices[3 * v + 1] + 0f;
            float z = mesh.Vertices[3 * v + 2] + 0f;
            if (!float.IsFinite(x) || !float.IsFinite(y) || !float.IsFinite(z))
            {
                remap[v] = -1;
                continue;
            }

            (int, int, int) key = (BitConverter.SingleToInt32Bits(x), BitConverter.SingleToInt32Bits(y), BitConverter.SingleToInt32Bits(z));
            if (!index.TryGetValue(key, out int merged))
            {
                merged = vertices.Count / 3;
                index[key] = merged;
                vertices.Add(x);
                vertices.Add(y);
                vertices.Add(z);
            }

            remap[v] = merged;
        }

        List<int> faces = new(mesh.Indices.Length);
        for (int t = 0; t < mesh.TriangleCount; t++)
        {
            int a = remap[mesh.Indices[3 * t]];
            int b = remap[mesh.Indices[3 * t + 1]];
            int c = remap[mesh.Indices[3 * t + 2]];
            if (a < 0 || b < 0 || c < 0)
            {
                continue;
            }

            faces.Add(a);
            faces.Add(b);
            faces.Add(c);
        }

        return faces.Count == 0
            ? throw new HartsyInferenceException("UniRig mesh has no finite triangles.")
            : ([.. vertices], [.. faces], remap);
    }

    private static (float[] VertexNormals, float[] FaceNormals) Normals(float[] vertices, int[] faces)
    {
        int faceCount = faces.Length / 3;
        float[] faceNormals = new float[faces.Length];
        double[] accumulated = new double[vertices.Length];
        for (int f = 0; f < faceCount; f++)
        {
            int[] corner = [faces[3 * f], faces[3 * f + 1], faces[3 * f + 2]];
            double[][] p = new double[3][];
            for (int k = 0; k < 3; k++)
            {
                p[k] = [vertices[3 * corner[k]], vertices[3 * corner[k] + 1], vertices[3 * corner[k] + 2]];
            }

            double[] n = Cross(Sub(p[1], p[0]), Sub(p[2], p[0]));
            double length = Math.Sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
            if (length > 1e-20)
            {
                n[0] /= length;
                n[1] /= length;
                n[2] /= length;
            }
            else
            {
                n[0] = n[1] = n[2] = 0;
            }

            for (int axis = 0; axis < 3; axis++)
            {
                faceNormals[3 * f + axis] = (float)n[axis];
            }

            for (int k = 0; k < 3; k++)
            {
                double angle = Angle(Sub(p[(k + 1) % 3], p[k]), Sub(p[(k + 2) % 3], p[k]));
                for (int axis = 0; axis < 3; axis++)
                {
                    accumulated[3 * corner[k] + axis] += n[axis] * angle;
                }
            }
        }

        float[] vertexNormals = new float[vertices.Length];
        for (int v = 0; v < vertices.Length / 3; v++)
        {
            double length = Math.Sqrt(accumulated[3 * v] * accumulated[3 * v] + accumulated[3 * v + 1] * accumulated[3 * v + 1]
                + accumulated[3 * v + 2] * accumulated[3 * v + 2]);
            for (int axis = 0; axis < 3; axis++)
            {
                vertexNormals[3 * v + axis] = length > 1e-20 ? (float)(accumulated[3 * v + axis] / length) : 0f;
            }
        }

        return (vertexNormals, faceNormals);
    }

    private static double[] Sub(double[] a, double[] b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];

    private static double[] Cross(double[] a, double[] b) =>
        [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];

    private static double Angle(double[] a, double[] b)
    {
        double la = Math.Sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
        double lb = Math.Sqrt(b[0] * b[0] + b[1] * b[1] + b[2] * b[2]);
        if (la < 1e-20 || lb < 1e-20)
        {
            return 0;
        }

        double cos = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / (la * lb);
        return Math.Acos(Math.Clamp(cos, -1.0, 1.0));
    }
}
