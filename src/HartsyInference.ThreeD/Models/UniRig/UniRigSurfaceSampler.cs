namespace HartsyInference.ThreeD.Models.UniRig;

/// <summary>
/// Upstream's <c>mix</c> sampler: random mesh vertices first, the rest on the surface. Faces are picked with probability
/// proportional to the squared length of their edge cross product, which is what upstream's code does (not plain area).
/// </summary>
public static class UniRigSurfaceSampler
{
    /// <summary>Draws <paramref name="total"/> points: <c>min(vertexSamples, VertexCount)</c> distinct vertices, then surface points.</summary>
    public static UniRigPointSample Sample(UniRigPreparedMesh mesh, int total, int vertexSamples, int seed)
    {
        ArgumentNullException.ThrowIfNull(mesh);
        ArgumentOutOfRangeException.ThrowIfLessThan(total, vertexSamples);
        Random random = new(seed);
        int fromVertices = Math.Min(vertexSamples, mesh.VertexCount);
        int fromSurface = total - fromVertices;
        float[] points = new float[3 * total];
        float[] normals = new float[3 * total];
        int[] vertex = new int[total];
        int[] face = new int[total];
        float[] bary = new float[2 * total];

        int[] chosen = UniRigPointSelection.Choose(mesh.VertexCount, fromVertices, random.Next());
        for (int i = 0; i < fromVertices; i++)
        {
            Array.Copy(mesh.Vertices, 3 * chosen[i], points, 3 * i, 3);
            Array.Copy(mesh.VertexNormals, 3 * chosen[i], normals, 3 * i, 3);
            vertex[i] = chosen[i];
            face[i] = -1;
        }

        double[] cumulative = new double[mesh.FaceCount];
        double running = 0;
        for (int f = 0; f < mesh.FaceCount; f++)
        {
            running += CrossLengthSquared(mesh, f);
            cumulative[f] = running;
        }

        for (int i = fromVertices; i < total; i++)
        {
            double pick = random.NextDouble() * running;
            int f = Math.Min(LowerBound(cumulative, pick), mesh.FaceCount - 1);
            double u = random.NextDouble();
            double v = random.NextDouble();
            if (u + v > 1.0)
            {
                u = Math.Abs(u - 1.0);
                v = Math.Abs(v - 1.0);
            }

            int a = mesh.Faces[3 * f];
            int b = mesh.Faces[3 * f + 1];
            int c = mesh.Faces[3 * f + 2];
            for (int axis = 0; axis < 3; axis++)
            {
                float pa = mesh.Vertices[3 * a + axis];
                points[3 * i + axis] = pa + (float)(u * (mesh.Vertices[3 * b + axis] - pa) + v * (mesh.Vertices[3 * c + axis] - pa));
                normals[3 * i + axis] = mesh.FaceNormals[3 * f + axis];
            }

            vertex[i] = -1;
            face[i] = f;
            bary[2 * i] = (float)u;
            bary[2 * i + 1] = (float)v;
        }

        return new UniRigPointSample { Points = points, Normals = normals, Vertex = vertex, Face = face, Barycentric = bary };
    }

    private static double CrossLengthSquared(UniRigPreparedMesh mesh, int f)
    {
        int a = 3 * mesh.Faces[3 * f];
        int b = 3 * mesh.Faces[3 * f + 1];
        int c = 3 * mesh.Faces[3 * f + 2];
        double e1x = mesh.Vertices[b] - mesh.Vertices[a];
        double e1y = mesh.Vertices[b + 1] - mesh.Vertices[a + 1];
        double e1z = mesh.Vertices[b + 2] - mesh.Vertices[a + 2];
        double e2x = mesh.Vertices[c] - mesh.Vertices[a];
        double e2y = mesh.Vertices[c + 1] - mesh.Vertices[a + 1];
        double e2z = mesh.Vertices[c + 2] - mesh.Vertices[a + 2];
        double x = e1y * e2z - e1z * e2y;
        double y = e1z * e2x - e1x * e2z;
        double z = e1x * e2y - e1y * e2x;
        return x * x + y * y + z * z;
    }

    private static int LowerBound(double[] sorted, double value)
    {
        int lo = 0;
        int hi = sorted.Length;
        while (lo < hi)
        {
            int mid = (lo + hi) / 2;
            if (sorted[mid] < value)
            {
                lo = mid + 1;
            }
            else
            {
                hi = mid;
            }
        }

        return lo;
    }
}
