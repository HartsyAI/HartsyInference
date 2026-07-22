using System.Buffers.Binary;
using System.Globalization;
using System.Text;
using HartsyInference.ThreeD.Geometry;

namespace HartsyInference.ThreeD.Io;

/// <summary>Writes PLY files: an ASCII triangle mesh, and the de-facto Gaussian-splat binary PLY
/// (INRIA layout) for the splat pipelines that land with later models. Pure managed C#.</summary>
public static class PlyWriter
{
    /// <summary>Serializes <paramref name="mesh"/> to ASCII PLY text (vertices + triangle faces, with
    /// optional normals and per-vertex RGB).</summary>
    public static string WriteMesh(Mesh mesh)
    {
        ArgumentNullException.ThrowIfNull(mesh);
        bool hasN = mesh.Normals is { Length: > 0 } && mesh.Normals.Length == mesh.Vertices.Length;
        bool hasC = mesh.VertexColors is { Length: > 0 } && mesh.VertexColors.Length == mesh.Vertices.Length;
        int vc = mesh.VertexCount, fc = mesh.TriangleCount;

        StringBuilder sb = new();
        sb.Append("ply\nformat ascii 1.0\ncomment HartsyInference.ThreeD\n");
        sb.Append("element vertex ").Append(vc).Append('\n');
        sb.Append("property float x\nproperty float y\nproperty float z\n");
        if (hasN) sb.Append("property float nx\nproperty float ny\nproperty float nz\n");
        if (hasC) sb.Append("property uchar red\nproperty uchar green\nproperty uchar blue\n");
        sb.Append("element face ").Append(fc).Append('\n');
        sb.Append("property list uchar int vertex_indices\n");
        sb.Append("end_header\n");

        float[] v = mesh.Vertices;
        for (int i = 0; i < vc; i++)
        {
            int b = i * 3;
            sb.Append(F(v[b])).Append(' ').Append(F(v[b + 1])).Append(' ').Append(F(v[b + 2]));
            if (hasN) sb.Append(' ').Append(F(mesh.Normals![b])).Append(' ').Append(F(mesh.Normals![b + 1])).Append(' ').Append(F(mesh.Normals![b + 2]));
            if (hasC) sb.Append(' ').Append(U8(mesh.VertexColors![b])).Append(' ').Append(U8(mesh.VertexColors![b + 1])).Append(' ').Append(U8(mesh.VertexColors![b + 2]));
            sb.Append('\n');
        }
        int[] idx = mesh.Indices;
        for (int t = 0; t < idx.Length; t += 3)
            sb.Append("3 ").Append(idx[t]).Append(' ').Append(idx[t + 1]).Append(' ').Append(idx[t + 2]).Append('\n');
        return sb.ToString();
    }

    /// <summary>Writes <paramref name="mesh"/> as ASCII PLY to <paramref name="path"/>.</summary>
    public static void SaveMesh(string path, Mesh mesh) => File.WriteAllText(path, WriteMesh(mesh));

    /// <summary>Serializes <paramref name="cloud"/> to a binary-little-endian Gaussian-splat PLY (INRIA layout:
    /// x,y,z, nx,ny,nz, f_dc_{0..2}, f_rest_*, opacity, scale_{0..2}, rot_{0..3}). The conventional format
    /// read by splat viewers. <paramref name="cloud"/>'s SH coefficients are stored coeff-major RGB and
    /// reordered here to INRIA's channel-major DC+rest ordering.</summary>
    public static byte[] WriteSplats(GaussianSplatCloud cloud)
    {
        ArgumentNullException.ThrowIfNull(cloud);
        int n = cloud.Count, k = cloud.ShCoeffsPerSplat;
        int restPerChannel = Math.Max(0, k - 1);

        StringBuilder h = new();
        h.Append("ply\nformat binary_little_endian 1.0\ncomment HartsyInference.ThreeD\n");
        h.Append("element vertex ").Append(n).Append('\n');
        foreach (string p in new[] { "x", "y", "z", "nx", "ny", "nz" }) h.Append("property float ").Append(p).Append('\n');
        for (int c = 0; c < 3; c++) h.Append("property float f_dc_").Append(c).Append('\n');
        for (int r = 0; r < restPerChannel * 3; r++) h.Append("property float f_rest_").Append(r).Append('\n');
        h.Append("property float opacity\n");
        for (int c = 0; c < 3; c++) h.Append("property float scale_").Append(c).Append('\n');
        for (int c = 0; c < 4; c++) h.Append("property float rot_").Append(c).Append('\n');
        h.Append("end_header\n");

        int floatsPerSplat = 6 + 3 + restPerChannel * 3 + 1 + 3 + 4;
        byte[] body = new byte[n * floatsPerSplat * 4];
        int off = 0;
        for (int i = 0; i < n; i++)
        {
            Put(body, ref off, cloud.Positions[i * 3]); Put(body, ref off, cloud.Positions[i * 3 + 1]); Put(body, ref off, cloud.Positions[i * 3 + 2]);
            Put(body, ref off, 0f); Put(body, ref off, 0f); Put(body, ref off, 0f); // normals (unused by viewers)
            // SH: stored [coeff][rgb]; emit DC (coeff0) for R,G,B then rest grouped by channel.
            for (int c = 0; c < 3; c++) Put(body, ref off, Sh(cloud, i, 0, c));
            for (int c = 0; c < 3; c++)
                for (int co = 1; co < k; co++) Put(body, ref off, Sh(cloud, i, co, c));
            Put(body, ref off, cloud.Opacities[i]);
            Put(body, ref off, cloud.Scales[i * 3]); Put(body, ref off, cloud.Scales[i * 3 + 1]); Put(body, ref off, cloud.Scales[i * 3 + 2]);
            Put(body, ref off, cloud.Rotations[i * 4]); Put(body, ref off, cloud.Rotations[i * 4 + 1]); Put(body, ref off, cloud.Rotations[i * 4 + 2]); Put(body, ref off, cloud.Rotations[i * 4 + 3]);
        }

        byte[] header = Encoding.ASCII.GetBytes(h.ToString());
        byte[] full = new byte[header.Length + body.Length];
        Buffer.BlockCopy(header, 0, full, 0, header.Length);
        Buffer.BlockCopy(body, 0, full, header.Length, body.Length);
        return full;
    }

    /// <summary>Writes <paramref name="cloud"/> as a binary-little-endian Gaussian-splat PLY to <paramref name="path"/>.</summary>
    public static void SaveSplats(string path, GaussianSplatCloud cloud) => File.WriteAllBytes(path, WriteSplats(cloud));

    private static float Sh(GaussianSplatCloud c, int splat, int coeff, int channel) =>
        c.ShCoefficients[(splat * c.ShCoeffsPerSplat + coeff) * 3 + channel];

    private static void Put(byte[] dst, ref int off, float f)
    {
        BinaryPrimitives.WriteSingleLittleEndian(dst.AsSpan(off), f); off += 4;
    }

    private static string F(float f) => f.ToString("R", CultureInfo.InvariantCulture);
    private static int U8(float c) => Math.Clamp((int)MathF.Round(c * 255f), 0, 255);
}
