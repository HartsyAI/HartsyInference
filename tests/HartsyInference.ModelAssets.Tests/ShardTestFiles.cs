using System.Text;
using System.Text.Json;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>Builds tiny safetensors shards and their index for the shard-set tests.</summary>
internal static class ShardTestFiles
{
    internal readonly record struct Entry(string Name, string DType, long[] Shape, byte[] Data);

    internal static Entry F32(string name, params float[] values)
    {
        byte[] data = new byte[values.Length * sizeof(float)];
        Buffer.BlockCopy(values, 0, data, 0, data.Length);
        return new Entry(name, "F32", [values.Length], data);
    }

    /// <summary>Writes a shard with tensors packed back to back in the given order.</summary>
    internal static void WriteShard(string path, params Entry[] entries)
    {
        Dictionary<string, object> header = new Dictionary<string, object>();
        long offset = 0;
        foreach (Entry entry in entries)
        {
            header[entry.Name] = new Dictionary<string, object>
            {
                ["dtype"] = entry.DType,
                ["shape"] = entry.Shape,
                ["data_offsets"] = new long[] { offset, offset + entry.Data.Length },
            };
            offset += entry.Data.Length;
        }
        using FileStream stream = OpenWithHeader(path, JsonSerializer.Serialize(header));
        foreach (Entry entry in entries)
            stream.Write(entry.Data);
    }

    /// <summary>Writes a file from a hand-authored header and pads the data section to <paramref name="dataLength"/> zero bytes.</summary>
    internal static void WriteRaw(string path, string headerJson, long dataLength)
    {
        using FileStream stream = OpenWithHeader(path, headerJson);
        stream.SetLength(stream.Position + dataLength);
    }

    /// <summary>Writes <c>model.safetensors.index.json</c>; a null total omits the metadata block.</summary>
    internal static void WriteIndex(string directory, IReadOnlyDictionary<string, string> weightMap, long? totalSize)
    {
        Dictionary<string, object> index = new Dictionary<string, object> { ["weight_map"] = weightMap };
        if (totalSize is { } total)
            index["metadata"] = new Dictionary<string, object> { ["total_size"] = total };
        File.WriteAllText(Path.Combine(directory, "model.safetensors.index.json"), JsonSerializer.Serialize(index));
    }

    /// <summary>Three shards holding a.0 a.1 | b.0 b.1 | c.0, plus a matching index.</summary>
    internal static Dictionary<string, string> WriteThreeShardSet(string directory, out long totalBytes)
    {
        WriteShard(Path.Combine(directory, "s1.safetensors"), F32("a.0", 1, 2, 3, 4), F32("a.1", 5, 6));
        WriteShard(Path.Combine(directory, "s2.safetensors"), F32("b.0", 7, 8), F32("b.1", 9));
        WriteShard(Path.Combine(directory, "s3.safetensors"), F32("c.0", 10, 11, 12));
        Dictionary<string, string> map = new Dictionary<string, string>
        {
            ["a.0"] = "s1.safetensors", ["a.1"] = "s1.safetensors",
            ["b.0"] = "s2.safetensors", ["b.1"] = "s2.safetensors",
            ["c.0"] = "s3.safetensors",
        };
        totalBytes = 12 * sizeof(float);
        WriteIndex(directory, map, totalBytes);
        return map;
    }

    private static FileStream OpenWithHeader(string path, string headerJson)
    {
        byte[] json = Encoding.UTF8.GetBytes(headerJson);
        FileStream stream = new FileStream(path, FileMode.Create, FileAccess.ReadWrite, FileShare.None);
        stream.Write(BitConverter.GetBytes((ulong)json.Length));
        stream.Write(json);
        return stream;
    }
}
