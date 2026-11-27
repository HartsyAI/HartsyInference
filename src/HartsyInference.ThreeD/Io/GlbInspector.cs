using System.Text.Json;
using HartsyInference.Core.Exceptions;

namespace HartsyInference.ThreeD.Io;

/// <summary>Reads the summary of a GLB from its JSON chunk, without touching the binary chunk.</summary>
public static class GlbInspector
{
    /// <summary>The summary of <paramref name="glb"/>.</summary>
    /// <exception cref="HartsyInferenceException">The bytes are not a glTF 2.0 GLB with a JSON chunk, or the JSON does not describe meshes the way glTF requires.</exception>
    public static GlbSummary Read(byte[] glb)
    {
        byte[] json = GlbContainer.Split(glb).Json;
        try
        {
            using JsonDocument doc = JsonDocument.Parse(json);
            return Summarize(doc.RootElement);
        }
        catch (Exception ex) when (ex is JsonException or IndexOutOfRangeException or FormatException or InvalidOperationException or ArgumentOutOfRangeException or KeyNotFoundException)
        {
            throw new HartsyInferenceException($"The GLB's JSON is malformed: {ex.Message}", ex);
        }
    }

    private static GlbSummary Summarize(JsonElement root)
    {
        int meshCount = root.TryGetProperty("meshes", out JsonElement meshes) ? meshes.GetArrayLength() : 0;
        bool normals = false;
        List<int> positionAccessors = [];
        long triangles = 0;
        if (meshCount > 0)
        {
            JsonElement accessorList = root.TryGetProperty("accessors", out JsonElement found) ? found : default;
            foreach (JsonElement mesh in meshes.EnumerateArray())
            {
                if (!mesh.TryGetProperty("primitives", out JsonElement primitives))
                    continue;
                foreach (JsonElement primitive in primitives.EnumerateArray())
                {
                    if (!primitive.TryGetProperty("attributes", out JsonElement attributes))
                        continue;
                    normals |= attributes.TryGetProperty("NORMAL", out _);
                    if (!attributes.TryGetProperty("POSITION", out JsonElement position))
                        continue;
                    positionAccessors.Add(position.GetInt32());
                    // Mode 4 (triangles) is the default; an indexed primitive has three indices per triangle, a non-indexed one three vertices.
                    if (accessorList.ValueKind == JsonValueKind.Array && (!primitive.TryGetProperty("mode", out JsonElement mode) || mode.GetInt32() == 4))
                    {
                        int source = primitive.TryGetProperty("indices", out JsonElement indices) ? indices.GetInt32() : position.GetInt32();
                        triangles += accessorList[source].GetProperty("count").GetInt64() / 3;
                    }
                }
            }
        }

        float[]? min = null, max = null;
        long vertices = 0;
        if (root.TryGetProperty("accessors", out JsonElement accessors))
        {
            foreach (int index in positionAccessors.Distinct())
            {
                JsonElement accessor = accessors[index];
                vertices += accessor.GetProperty("count").GetInt64();
                if (!accessor.TryGetProperty("min", out JsonElement lo) || !accessor.TryGetProperty("max", out JsonElement hi))
                    continue;
                float[] low = [lo[0].GetSingle(), lo[1].GetSingle(), lo[2].GetSingle()];
                float[] high = [hi[0].GetSingle(), hi[1].GetSingle(), hi[2].GetSingle()];
                min = min is null ? low : [Math.Min(min[0], low[0]), Math.Min(min[1], low[1]), Math.Min(min[2], low[2])];
                max = max is null ? high : [Math.Max(max[0], high[0]), Math.Max(max[1], high[1]), Math.Max(max[2], high[2])];
            }
        }

        List<string> animations = [];
        if (root.TryGetProperty("animations", out JsonElement list))
        {
            int i = 0;
            foreach (JsonElement animation in list.EnumerateArray())
            {
                animations.Add(animation.TryGetProperty("name", out JsonElement name) && name.GetString() is { Length: > 0 } text ? text : $"animation_{i}");
                i++;
            }
        }
        return new GlbSummary(
            meshCount, normals, root.TryGetProperty("textures", out JsonElement textures) && textures.GetArrayLength() > 0,
            root.TryGetProperty("skins", out JsonElement skins) && skins.GetArrayLength() > 0, animations, min, max)
        {
            VertexCount = vertices,
            TriangleCount = triangles,
        };
    }
}
