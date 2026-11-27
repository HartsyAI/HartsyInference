using System.Runtime.InteropServices;
using System.Text.Json;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Kimi.Vision;

namespace HartsyInference.LLM.Tests.Kimi.Vision;

/// <summary>Seeded MoonViT-V2 and projector tensors under the published names and shapes, for one tiny config: the names a checkpoint index
/// carries, the values the test expects back, and a writer that stores them as a safetensors shard plus index.</summary>
internal static class KimiVisionSyntheticCheckpoint
{
    /// <summary>A two-head, width-8 tower that keeps every rotary pair and merge block real while staying tiny.</summary>
    internal static KimiVisionConfig Tiny(int layers) => new()
    {
        Hidden = 8,
        Heads = 2,
        QkvHidden = 8,
        MlpDim = 12,
        Layers = layers,
        PatchSize = 2,
        MergeSize = 2,
        InitPosGrid = 4,
        TextHidden = 6,
        ProjectorEpsilon = 1e-5f,
        BlockNormEpsilon = 1.1920929e-7f,
        RopeTheta = 10000f,
        ImageSize = 8,
    };

    internal static Dictionary<string, long[]> Dims(KimiVisionConfig c) =>
        TensorNames(c).ToDictionary(t => t.Name, t => t.Dims, StringComparer.Ordinal);

    /// <summary>Every tensor the tower and projector read, with its checkpoint shape.</summary>
    internal static List<(string Name, long[] Dims)> TensorNames(KimiVisionConfig c)
    {
        long h = c.Hidden, q = c.QkvHidden, m = c.MlpDim, g = c.InitPosGrid, p = c.PatchSize, mm = c.MergedDim, t = c.TextHidden;
        List<(string Name, long[] Dims)> list =
        [
            ("vision_tower.patch_embed.proj.weight", [h, 3, p, p]),
            ("vision_tower.patch_embed.pos_emb.weight", [g, g, h]),
            ("vision_tower.encoder.final_layernorm.weight", [h]),
            ("mm_projector.proj.0.weight", [mm, mm]),
            ("mm_projector.proj.2.weight", [t, mm]),
            ("mm_projector.post_norm.weight", [t]),
        ];
        for (int i = 0; i < c.Layers; i++)
        {
            string b = $"vision_tower.encoder.blocks.{i}.";
            list.Add((b + "wqkv.weight", [3 * q, h]));
            list.Add((b + "wo.weight", [h, q]));
            list.Add((b + "norm0.weight", [h]));
            list.Add((b + "norm1.weight", [h]));
            list.Add((b + "mlp.fc0.weight", [m, h]));
            list.Add((b + "mlp.fc1.weight", [h, m]));
        }
        return list;
    }

    /// <summary>Seeded F32 values for each tensor; norm weights sit near one. Tensors named in <paramref name="bf16Names"/> are returned as the
    /// BF16 values a BF16 shard stores (truncated to 16 bits).</summary>
    internal static Dictionary<string, float[]> Values(KimiVisionConfig c, int seed, params string[] bf16Names)
    {
        Random rng = new(seed);
        Dictionary<string, float[]> values = new(StringComparer.Ordinal);
        foreach ((string name, long[] dims) in TensorNames(c))
        {
            long count = 1;
            foreach (long d in dims) count *= d;
            bool norm = name.Contains("norm", StringComparison.Ordinal);
            float[] v = new float[count];
            for (int i = 0; i < v.Length; i++)
            {
                float u = (float)(rng.NextDouble() * 0.2 - 0.1);
                v[i] = norm ? 1f + u : u;
            }
            if (bf16Names.Contains(name))
                for (int i = 0; i < v.Length; i++) v[i] = Bf16ToF32(TensorCasts.F32ToBf16Bits(v[i]));
            values[name] = v;
        }
        return values;
    }

    /// <summary>Writes <paramref name="values"/> as one shard with its index. Tensors named in <paramref name="bf16Names"/> are stored BF16.</summary>
    internal static void Write(string directory, KimiVisionConfig c, IReadOnlyDictionary<string, float[]> values, params string[] bf16Names)
    {
        Directory.CreateDirectory(directory);
        Dictionary<string, object> header = new(StringComparer.Ordinal);
        List<byte[]> payload = [];
        Dictionary<string, object> weightMap = new(StringComparer.Ordinal);
        long offset = 0;
        foreach ((string name, long[] dims) in TensorNames(c))
        {
            float[] v = values[name];
            bool bf16 = bf16Names.Contains(name);
            byte[] bytes = bf16 ? Bf16Bytes(v) : MemoryMarshal.AsBytes(v.AsSpan()).ToArray();
            header[name] = new Dictionary<string, object>
            {
                ["dtype"] = bf16 ? "BF16" : "F32",
                ["shape"] = dims,
                ["data_offsets"] = new long[] { offset, offset + bytes.Length },
            };
            payload.Add(bytes);
            weightMap[name] = "model.safetensors";
            offset += bytes.Length;
        }

        byte[] headerBytes = JsonSerializer.SerializeToUtf8Bytes(header);
        using (FileStream stream = File.Create(Path.Combine(directory, "model.safetensors")))
        using (BinaryWriter writer = new(stream))
        {
            writer.Write((long)headerBytes.Length);
            writer.Write(headerBytes);
            foreach (byte[] bytes in payload) writer.Write(bytes);
        }
        File.WriteAllText(Path.Combine(directory, "model.safetensors.index.json"),
            JsonSerializer.Serialize(new Dictionary<string, object> { ["weight_map"] = weightMap }));
    }

    internal static float Bf16ToF32(ushort bits) => BitConverter.Int32BitsToSingle((int)bits << 16);

    private static byte[] Bf16Bytes(float[] values)
    {
        byte[] bytes = new byte[values.Length * 2];
        for (int i = 0; i < values.Length; i++)
        {
            ushort bits = TensorCasts.F32ToBf16Bits(values[i]);
            bytes[2 * i] = (byte)bits;
            bytes[2 * i + 1] = (byte)(bits >> 8);
        }
        return bytes;
    }
}
