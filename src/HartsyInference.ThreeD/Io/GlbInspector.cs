using System.Buffers.Binary;
using System.Text.Json;
using HartsyInference.Core.Exceptions;

namespace HartsyInference.ThreeD.Io;

/// <summary>Reads the summary of a GLB from its JSON chunk, and splits and rejoins the chunks for the edits that change only the JSON.</summary>
public static class GlbInspector
{
    private const uint Magic = 0x46546C67;
    private const uint JsonChunk = 0x4E4F534A;

    /// <summary>The summary of <paramref name="glb"/>.</summary>
    /// <exception cref="HartsyInferenceException">The bytes are not a glTF 2.0 GLB with a JSON chunk.</exception>
    public static GlbSummary Read(byte[] glb)
    {
        using JsonDocument doc = JsonDocument.Parse(SplitJson(glb).Json);
        JsonElement root = doc.RootElement;
        int meshCount = root.TryGetProperty("meshes", out JsonElement meshes) ? meshes.GetArrayLength() : 0;
        bool normals = false;
        List<int> positionAccessors = [];
        if (meshCount > 0)
        {
            foreach (JsonElement mesh in meshes.EnumerateArray())
            {
                if (!mesh.TryGetProperty("primitives", out JsonElement primitives))
                    continue;
                foreach (JsonElement primitive in primitives.EnumerateArray())
                {
                    if (!primitive.TryGetProperty("attributes", out JsonElement attributes))
                        continue;
                    normals |= attributes.TryGetProperty("NORMAL", out _);
                    if (attributes.TryGetProperty("POSITION", out JsonElement position))
                        positionAccessors.Add(position.GetInt32());
                }
            }
        }

        float[]? min = null, max = null;
        if (root.TryGetProperty("accessors", out JsonElement accessors))
        {
            foreach (int index in positionAccessors.Distinct())
            {
                JsonElement accessor = accessors[index];
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
            root.TryGetProperty("skins", out JsonElement skins) && skins.GetArrayLength() > 0, animations, min, max);
    }

    /// <summary>The JSON chunk's bytes and the offset and length of that chunk's payload in <paramref name="glb"/>.</summary>
    internal static (byte[] Json, int Offset, int Length) SplitJson(byte[] glb)
    {
        ArgumentNullException.ThrowIfNull(glb);
        if (glb.Length < 20 || BinaryPrimitives.ReadUInt32LittleEndian(glb) != Magic)
            throw new HartsyInferenceException("File is not a binary glTF (GLB) container.");
        if (BinaryPrimitives.ReadUInt32LittleEndian(glb.AsSpan(4)) != 2)
            throw new HartsyInferenceException("Only glTF 2.0 GLB containers are supported.");
        int length = (int)BinaryPrimitives.ReadUInt32LittleEndian(glb.AsSpan(12));
        if (BinaryPrimitives.ReadUInt32LittleEndian(glb.AsSpan(16)) != JsonChunk || 20 + length > glb.Length)
            throw new HartsyInferenceException("GLB file has no leading JSON chunk.");
        return (glb.AsSpan(20, length).ToArray(), 20, length);
    }

    /// <summary>Replaces the JSON chunk of <paramref name="glb"/> with <paramref name="json"/>, padded with spaces to four bytes, and fixes the container lengths.</summary>
    internal static byte[] ReplaceJson(byte[] glb, byte[] json)
    {
        (_, int offset, int length) = SplitJson(glb);
        int padded = (json.Length + 3) & ~3;
        byte[] result = new byte[glb.Length - length + padded];
        Buffer.BlockCopy(glb, 0, result, 0, 20);
        Buffer.BlockCopy(json, 0, result, 20, json.Length);
        Array.Fill(result, (byte)' ', 20 + json.Length, padded - json.Length);
        Buffer.BlockCopy(glb, offset + length, result, 20 + padded, glb.Length - offset - length);
        BinaryPrimitives.WriteUInt32LittleEndian(result.AsSpan(8), (uint)result.Length);
        BinaryPrimitives.WriteUInt32LittleEndian(result.AsSpan(12), (uint)padded);
        return result;
    }
}
