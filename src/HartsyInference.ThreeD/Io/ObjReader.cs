using System.Globalization;
using HartsyInference.Core.Exceptions;
using HartsyInference.ThreeD.Geometry;

namespace HartsyInference.ThreeD.Io;

/// <summary>Reads triangle geometry from Wavefront OBJ text: <c>v</c> and <c>f</c> records only. Normals, texture
/// coordinates, materials and groups are ignored, and polygons are fan-triangulated.</summary>
public static class ObjReader
{
    /// <summary>Parses OBJ text. Face references may be <c>v</c>, <c>v/vt</c>, <c>v//vn</c> or <c>v/vt/vn</c>, and may be negative (relative).</summary>
    public static Mesh Parse(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        List<float> vertices = [];
        List<int> indices = [];
        foreach (string raw in text.Split('\n'))
        {
            string line = raw.Trim();
            if (line.StartsWith("v ", StringComparison.Ordinal))
                vertices.AddRange(ParseVertex(line));
            else if (line.StartsWith("f ", StringComparison.Ordinal))
                AddFace(line, vertices.Count / 3, indices);
        }

        if (indices.Count == 0)
            throw new HartsyInferenceException("OBJ scene has no faces.");
        return new Mesh { Vertices = [.. vertices], Indices = [.. indices] };
    }

    /// <summary>Reads and parses an OBJ file.</summary>
    public static Mesh Load(string path) => Parse(File.ReadAllText(path));

    private static float[] ParseVertex(string line)
    {
        string[] parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        if (parts.Length < 4)
            throw new HartsyInferenceException($"OBJ vertex needs three coordinates: '{line}'.");
        float[] xyz = new float[3];
        for (int a = 0; a < 3; a++)
        {
            if (!float.TryParse(parts[a + 1], NumberStyles.Float, CultureInfo.InvariantCulture, out float value) || !float.IsFinite(value))
                throw new HartsyInferenceException($"OBJ vertex coordinate is not a finite number: '{line}'.");
            xyz[a] = value;
        }
        return xyz;
    }

    private static void AddFace(string line, int vertexCount, List<int> indices)
    {
        string[] parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        if (parts.Length < 4)
            throw new HartsyInferenceException($"OBJ face needs at least three vertices: '{line}'.");
        int first = ResolveIndex(parts[1], vertexCount);
        for (int k = 2; k + 1 < parts.Length; k++)
        {
            indices.Add(first);
            indices.Add(ResolveIndex(parts[k], vertexCount));
            indices.Add(ResolveIndex(parts[k + 1], vertexCount));
        }
    }

    // OBJ indices are 1-based, and negative values count back from the last vertex defined so far.
    private static int ResolveIndex(string token, int vertexCount)
    {
        int slash = token.IndexOf('/');
        string head = slash < 0 ? token : token[..slash];
        if (!int.TryParse(head, NumberStyles.Integer, CultureInfo.InvariantCulture, out int value) || value == 0)
            throw new HartsyInferenceException($"OBJ face index '{token}' is not a valid vertex reference.");
        int zeroBased = value > 0 ? value - 1 : vertexCount + value;
        if (zeroBased < 0 || zeroBased >= vertexCount)
            throw new HartsyInferenceException($"OBJ face index '{token}' is outside the {vertexCount} vertices defined.");
        return zeroBased;
    }
}
