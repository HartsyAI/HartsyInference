using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using HartsyInference.Core.Tensors;

namespace HartsyInference.LLM.Tests.DeepSeekV41;

/// <summary>Writes the vision-tower fixture out as a V4.1 safetensors directory holding only the <c>vision.*</c>, <c>aligner.*</c> and <c>image_*</c> tensors, under the official key names, with a matching config.json.</summary>
internal static class DeepSeekV41VisionFixtureCheckpoint
{
    public static readonly string[] EmbeddingKeys = ["image_start", "image_end", "image_newline"];

    /// <summary>The deterministic value of element <paramref name="index"/> of an image embedding, so a test can recompute what it wrote.</summary>
    public static float EmbeddingValue(string key, int index) => (Array.IndexOf(EmbeddingKeys, key) + 1) * 0.125f + index * 0.0625f - 0.5f;

    /// <summary>Truncates to bf16 and widens back, which is exactly what storing a value as BF16 and reading it as F32 does.</summary>
    public static float ThroughBf16(float value) => BitConverter.UInt32BitsToSingle(BitConverter.SingleToUInt32Bits(value) & 0xFFFF0000u);

    /// <param name="directory">An existing empty directory.</param>
    /// <param name="bf16">Store BF16 like the official checkpoint instead of F32.</param>
    /// <param name="omit">A key to leave out.</param>
    /// <param name="reshape">A key whose tensor is written as a transposed copy of its shape (same element count, other shape).</param>
    /// <param name="withVisionConfig">False writes a config.json with no <c>vision_config</c>.</param>
    public static void Write(string directory, bool bf16 = false, string? omit = null, string? reshape = null, bool withVisionConfig = true)
    {
        Dictionary<string, object> header = [];
        List<byte[]> blobs = [];
        long offset = 0;
        int outDim = DeepSeekV41VisionFixture.OutputDim;
        List<(string Key, float[] Values, long[] Shape)> tensors = [];
        foreach (string key in DeepSeekV41VisionFixture.ParamKeys())
            tensors.Add((key, DeepSeekV41VisionFixture.Param(key), DeepSeekV41VisionFixture.Shape(key)));
        foreach (string key in EmbeddingKeys)
            tensors.Add((key, Enumerable.Range(0, outDim).Select(i => EmbeddingValue(key, i)).ToArray(), [outDim]));

        foreach ((string key, float[] values, long[] shape) in tensors)
        {
            if (key == omit) continue;
            long[] written = key == reshape ? shape.Reverse().ToArray() : shape;
            byte[] bytes = bf16 ? ToBf16(values) : ToF32(values);
            header[key] = new Dictionary<string, object>
            {
                ["dtype"] = bf16 ? "BF16" : "F32",
                ["shape"] = written,
                ["data_offsets"] = new long[] { offset, offset + bytes.Length },
            };
            blobs.Add(bytes);
            offset += bytes.Length;
        }
        const string shard = "model-00001-of-00001.safetensors";
        byte[] json = Encoding.UTF8.GetBytes(JsonSerializer.Serialize(header));
        using (FileStream stream = new(Path.Combine(directory, shard), FileMode.Create, FileAccess.Write))
        {
            stream.Write(BitConverter.GetBytes((ulong)json.Length));
            stream.Write(json);
            foreach (byte[] blob in blobs) stream.Write(blob);
        }
        File.WriteAllText(Path.Combine(directory, "model.safetensors.index.json"), JsonSerializer.Serialize(new
        {
            metadata = new { total_size = offset },
            weight_map = header.Keys.ToDictionary(static k => k, _ => shard),
        }));
        File.WriteAllText(Path.Combine(directory, "config.json"), ConfigJson(withVisionConfig));
    }

    /// <summary>The pinned official config with the language width and the vision tower shrunk to the fixture's.</summary>
    public static string ConfigJson(bool withVisionConfig)
    {
        string official = DeepSeekV41Fixtures.OfficialConfig(text => text["hidden_size"] = DeepSeekV41VisionFixture.OutputDim);
        JsonObject root = JsonNode.Parse(official)!.AsObject();
        JsonElement c = DeepSeekV41VisionFixture.Root.GetProperty("config");
        if (!withVisionConfig)
        {
            root.Remove("vision_config");
            return root.ToJsonString();
        }
        root["vision_config"] = new JsonObject
        {
            ["model_type"] = "deepseek_v41_vision",
            ["num_hidden_layers"] = c.GetProperty("vision_n_layers").GetInt32(),
            ["hidden_size"] = c.GetProperty("vision_dim").GetInt32(),
            ["num_attention_heads"] = c.GetProperty("vision_n_heads").GetInt32(),
            ["intermediate_size"] = c.GetProperty("vision_inter_dim").GetInt32(),
            ["patch_size"] = c.GetProperty("vision_patch_size").GetInt32(),
            ["rope_theta"] = c.GetProperty("vision_rope_theta").GetDouble(),
            ["downsample_ratio"] = c.GetProperty("vision_downsample_ratio").GetInt32(),
        };
        return root.ToJsonString();
    }

    private static byte[] ToF32(float[] values)
    {
        byte[] bytes = new byte[values.Length * sizeof(float)];
        Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
        return bytes;
    }

    private static byte[] ToBf16(float[] values)
    {
        byte[] bytes = new byte[values.Length * sizeof(ushort)];
        for (int i = 0; i < values.Length; i++) BitConverter.TryWriteBytes(bytes.AsSpan(i * 2, 2), TensorCasts.F32ToBf16Bits(values[i]));
        return bytes;
    }
}
