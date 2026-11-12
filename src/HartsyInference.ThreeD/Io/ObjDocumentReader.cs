using System.Globalization;
using HartsyInference.Core.Exceptions;
using HartsyInference.ThreeD.Document;
using HartsyInference.ThreeD.Geometry;

namespace HartsyInference.ThreeD.Io;

/// <summary>Reads Wavefront OBJ text into a <see cref="MeshDocument"/> with its UVs (V flipped to glTF's top-left origin) and normals. A vertex that is used with different UVs or normals becomes several vertices. Polygons are fan-triangulated. Materials (<c>usemtl</c>, <c>mtllib</c>) are not read, because the MTL file is not part of the input; the document says so.</summary>
public static class ObjDocumentReader
{
    /// <summary>Parses <paramref name="text"/>.</summary>
    public static MeshDocument Parse(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        List<float> positions = [], uvs = [], normals = [];
        List<float> outPositions = [], outUvs = [], outNormals = [];
        List<int> indices = [];
        Dictionary<(int, int, int), int> unique = [];
        bool anyUv = false, anyNormal = false, materials = false;
        foreach (string raw in text.Split('\n'))
        {
            string[] tokens = raw.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
            if (tokens.Length == 0 || tokens[0][0] == '#')
                continue;
            switch (tokens[0])
            {
                case "v":
                    positions.AddRange(Numbers(tokens, 3));
                    break;
                case "vt":
                    uvs.AddRange(Numbers(tokens, 2));
                    break;
                case "vn":
                    normals.AddRange(Numbers(tokens, 3));
                    break;
                case "f":
                    AddFace(tokens);
                    break;
                case "usemtl" or "mtllib":
                    materials = true;
                    break;
            }
        }
        if (indices.Count == 0)
            throw new HartsyInferenceException("OBJ scene has no faces.");

        List<string> notes = [];
        if (materials)
            notes.Add("The OBJ names materials, which are not read (the MTL file is not part of the input).");
        Mesh mesh = new()
        {
            Vertices = [.. outPositions],
            Indices = [.. indices],
            Uvs = anyUv ? [.. outUvs] : null,
            Normals = anyNormal ? [.. outNormals] : null,
        };
        return new MeshDocument { Primitives = [new MeshPrimitive { Geometry = mesh }], Notes = notes };

        void AddFace(string[] tokens)
        {
            if (tokens.Length < 4)
                throw new HartsyInferenceException($"OBJ face needs at least three vertices: '{string.Join(' ', tokens)}'.");
            int first = Corner(tokens[1]);
            for (int k = 2; k + 1 < tokens.Length; k++)
            {
                indices.Add(first);
                indices.Add(Corner(tokens[k]));
                indices.Add(Corner(tokens[k + 1]));
            }
        }

        int Corner(string token)
        {
            string[] parts = token.Split('/');
            int v = Resolve(parts[0], positions.Count / 3, token);
            int vt = parts.Length > 1 && parts[1].Length > 0 ? Resolve(parts[1], uvs.Count / 2, token) : -1;
            int vn = parts.Length > 2 && parts[2].Length > 0 ? Resolve(parts[2], normals.Count / 3, token) : -1;
            if (unique.TryGetValue((v, vt, vn), out int existing))
                return existing;
            int index = outPositions.Count / 3;
            unique[(v, vt, vn)] = index;
            outPositions.AddRange(positions.GetRange(v * 3, 3));
            if (vt >= 0)
            {
                anyUv = true;
                outUvs.Add(uvs[vt * 2]);
                outUvs.Add(1f - uvs[vt * 2 + 1]);
            }
            else
            {
                outUvs.Add(0f);
                outUvs.Add(0f);
            }
            if (vn >= 0)
            {
                anyNormal = true;
                outNormals.AddRange(normals.GetRange(vn * 3, 3));
            }
            else
            {
                outNormals.AddRange([0f, 0f, 0f]);
            }
            return index;
        }
    }

    private static float[] Numbers(string[] tokens, int count)
    {
        if (tokens.Length < count + 1)
            throw new HartsyInferenceException($"OBJ '{tokens[0]}' record needs {count} numbers: '{string.Join(' ', tokens)}'.");
        float[] values = new float[count];
        for (int i = 0; i < count; i++)
        {
            if (!float.TryParse(tokens[i + 1], NumberStyles.Float, CultureInfo.InvariantCulture, out float value) || !float.IsFinite(value))
                throw new HartsyInferenceException($"OBJ '{tokens[0]}' value is not a finite number: '{string.Join(' ', tokens)}'.");
            values[i] = value;
        }
        return values;
    }

    // OBJ indices are 1-based, and negative values count back from the last element defined so far.
    private static int Resolve(string text, int defined, string token)
    {
        if (!int.TryParse(text, NumberStyles.Integer, CultureInfo.InvariantCulture, out int value) || value == 0)
            throw new HartsyInferenceException($"OBJ face index '{token}' is not a valid reference.");
        int zeroBased = value > 0 ? value - 1 : defined + value;
        return zeroBased >= 0 && zeroBased < defined ? zeroBased : throw new HartsyInferenceException($"OBJ face index '{token}' is outside the {defined} elements defined.");
    }
}
