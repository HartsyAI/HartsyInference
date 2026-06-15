using System.Globalization;
using System.Text;
using HartsyInference.ThreeD.Geometry;

namespace HartsyInference.ThreeD.Io;

/// <summary>Writes a <see cref="Mesh"/> to Wavefront OBJ (ASCII) — a human-readable debug/interop format.
/// Emits <c>v</c>, optional <c>vn</c>, and triangle <c>f</c> records (OBJ is 1-indexed).</summary>
public static class ObjWriter
{
    /// <summary>Serializes <paramref name="mesh"/> to OBJ text.</summary>
    public static string Write(Mesh mesh)
    {
        ArgumentNullException.ThrowIfNull(mesh);
        bool hasNormals = mesh.Normals is { Length: > 0 } && mesh.Normals.Length == mesh.Vertices.Length;
        StringBuilder sb = new();
        sb.Append("# HartsyInference.ThreeD OBJ export\n");

        float[] v = mesh.Vertices;
        for (int i = 0; i < v.Length; i += 3)
            sb.Append("v ").Append(F(v[i])).Append(' ').Append(F(v[i + 1])).Append(' ').Append(F(v[i + 2])).Append('\n');

        if (hasNormals)
        {
            float[] n = mesh.Normals!;
            for (int i = 0; i < n.Length; i += 3)
                sb.Append("vn ").Append(F(n[i])).Append(' ').Append(F(n[i + 1])).Append(' ').Append(F(n[i + 2])).Append('\n');
        }

        int[] idx = mesh.Indices;
        for (int t = 0; t < idx.Length; t += 3)
        {
            int a = idx[t] + 1, b = idx[t + 1] + 1, c = idx[t + 2] + 1;
            if (hasNormals)
                sb.Append("f ").Append(a).Append("//").Append(a).Append(' ')
                  .Append(b).Append("//").Append(b).Append(' ')
                  .Append(c).Append("//").Append(c).Append('\n');
            else
                sb.Append("f ").Append(a).Append(' ').Append(b).Append(' ').Append(c).Append('\n');
        }
        return sb.ToString();
    }

    /// <summary>Writes <paramref name="mesh"/> as OBJ to <paramref name="path"/>.</summary>
    public static void Save(string path, Mesh mesh) => File.WriteAllText(path, Write(mesh));

    private static string F(float f) => f.ToString("R", CultureInfo.InvariantCulture);
}
