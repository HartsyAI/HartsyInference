using System.Globalization;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Logging;
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
        HashSet<string> skipped = [];
        foreach (string raw in text.Split('\n'))
        {
            string[] tokens = raw.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
            if (tokens.Length == 0 || tokens[0][0] == '#')
                continue;
            switch (tokens[0])
            {
                case "v":
                    vertices.AddRange(ParseVertex(tokens));
                    break;
                case "f":
                    AddFace(tokens, vertices.Count / 3, indices);
                    break;
                case "vt" or "vn" or "vp" or "g" or "o" or "s" or "usemtl" or "mtllib":
                    break;
                default:
                    skipped.Add(tokens[0]);
                    break;
            }
        }

        foreach (string keyword in skipped)
            Logs.Warning($"OBJ records of type '{keyword}' are not read and were skipped.");
        if (indices.Count == 0)
            throw new HartsyInferenceException("OBJ scene has no faces.");
        return new Mesh { Vertices = [.. vertices], Indices = [.. indices] };
    }

    /// <summary>Reads and parses an OBJ file.</summary>
    public static Mesh Load(string path) => Parse(File.ReadAllText(path));

    private static float[] ParseVertex(string[] tokens)
    {
        if (tokens.Length < 4)
            throw new HartsyInferenceException($"OBJ vertex needs three coordinates: '{string.Join(' ', tokens)}'.");
        float[] xyz = new float[3];
        for (int a = 0; a < 3; a++)
        {
            if (!float.TryParse(tokens[a + 1], NumberStyles.Float, CultureInfo.InvariantCulture, out float value) || !float.IsFinite(value))
                throw new HartsyInferenceException($"OBJ vertex coordinate is not a finite number: '{string.Join(' ', tokens)}'.");
            xyz[a] = value;
        }
        return xyz;
    }

    private static void AddFace(string[] tokens, int vertexCount, List<int> indices)
    {
        if (tokens.Length < 4)
            throw new HartsyInferenceException($"OBJ face needs at least three vertices: '{string.Join(' ', tokens)}'.");
        int first = ResolveIndex(tokens[1], vertexCount);
        for (int k = 2; k + 1 < tokens.Length; k++)
        {
            indices.Add(first);
            indices.Add(ResolveIndex(tokens[k], vertexCount));
            indices.Add(ResolveIndex(tokens[k + 1], vertexCount));
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
