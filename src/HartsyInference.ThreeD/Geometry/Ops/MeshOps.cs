using HartsyInference.ThreeD.Geometry;

namespace HartsyInference.ThreeD.Geometry.Ops;

/// <summary>Mesh post-processing helpers shared by every mesh-producing pipeline.</summary>
public static class MeshOps
{
    /// <summary>Computes smooth per-vertex normals by area-weighted accumulation of face normals,
    /// then normalizing. Writes <see cref="Mesh.Normals"/> in place and returns the mesh.</summary>
    public static Mesh ComputeVertexNormals(Mesh mesh)
    {
        ArgumentNullException.ThrowIfNull(mesh);
        float[] v = mesh.Vertices;
        int[] idx = mesh.Indices;
        float[] n = new float[v.Length];

        for (int t = 0; t < idx.Length; t += 3)
        {
            int ia = idx[t] * 3, ib = idx[t + 1] * 3, ic = idx[t + 2] * 3;
            float ax = v[ia], ay = v[ia + 1], az = v[ia + 2];
            float bx = v[ib], by = v[ib + 1], bz = v[ib + 2];
            float cx = v[ic], cy = v[ic + 1], cz = v[ic + 2];
            // (b-a) x (c-a) — magnitude is 2*triangle area, so this is area-weighted.
            float e1x = bx - ax, e1y = by - ay, e1z = bz - az;
            float e2x = cx - ax, e2y = cy - ay, e2z = cz - az;
            float fx = e1y * e2z - e1z * e2y;
            float fy = e1z * e2x - e1x * e2z;
            float fz = e1x * e2y - e1y * e2x;
            n[ia] += fx; n[ia + 1] += fy; n[ia + 2] += fz;
            n[ib] += fx; n[ib + 1] += fy; n[ib + 2] += fz;
            n[ic] += fx; n[ic + 1] += fy; n[ic + 2] += fz;
        }

        for (int i = 0; i < n.Length; i += 3)
        {
            float len = MathF.Sqrt(n[i] * n[i] + n[i + 1] * n[i + 1] + n[i + 2] * n[i + 2]);
            if (len > 1e-12f) { n[i] /= len; n[i + 1] /= len; n[i + 2] /= len; }
        }

        mesh.Normals = n;
        return mesh;
    }
}
