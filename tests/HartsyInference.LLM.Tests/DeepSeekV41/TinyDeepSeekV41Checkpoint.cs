using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using HartsyInference.ModelAssets.Quant;

namespace HartsyInference.LLM.Tests.DeepSeekV41;

/// <summary>Writes a miniature but structurally faithful DeepSeek-V4.1 checkpoint directory: 4 backbone layers, one draft
/// layer, an Engram table on layer 1, and the same key names, dtypes and companion layout as the real producers.</summary>
/// <remarks>Synthetic by design: it proves naming, classification, binding and draft-completeness logic on files small
/// enough for CI, not real weight values. The real shard headers are covered by the header-template tests.</remarks>
internal static class TinyDeepSeekV41Checkpoint
{
    public const int BackboneLayers = 4;
    public const int RoutedExperts = 4;
    public const int DraftExperts = 3;
    public const int EngramLayer = 1;
    public const int EngramRows = 8;

    /// <summary>Creates the directory contents and returns the total tensor bytes written (the index's total_size).</summary>
    /// <param name="directory">An existing empty directory.</param>
    /// <param name="flavor">Official (fp4 experts + .scale) or Mlx (affine int4 + .scales/.biases).</param>
    /// <param name="fullDraftExperts">Draft experts written whole; the rest are absent (MLX gap shape). Null writes all.</param>
    /// <param name="partialDraftExpert">A draft expert written as bare weights without companions, like MLX's expert 87.</param>
    /// <param name="includeDraft">False drops every mtp tensor, as the DwarfStar and some derivatives do.</param>
    public static long Write(string directory, QuantFlavor flavor = QuantFlavor.Official,
        IReadOnlyCollection<int>? fullDraftExperts = null, int? partialDraftExpert = null, bool includeDraft = true)
    {
        List<Tensor> tensors = new();
        AddDense(tensors, includeDraft);
        for (int layer = 0; layer < BackboneLayers; layer++)
        {
            for (int expert = 0; expert < RoutedExperts; expert++)
                AddExpert(tensors, flavor, $"layers.{layer}.ffn.experts.{expert}", full: true);
        }
        AddEngram(tensors);
        if (includeDraft)
        {
            for (int expert = 0; expert < DraftExperts; expert++)
            {
                if (fullDraftExperts is null || fullDraftExperts.Contains(expert))
                    AddExpert(tensors, flavor, $"mtp.0.ffn.experts.{expert}", full: true);
            }
            if (partialDraftExpert is { } partial)
                AddExpert(tensors, flavor, $"mtp.0.ffn.experts.{partial}", full: false);
        }

        string[] shardNames = ["model-00001-of-00003.safetensors", "model-00002-of-00003.safetensors", "model-00003-of-00003.safetensors"];
        Dictionary<string, string> weightMap = new();
        for (int shard = 0; shard < shardNames.Length; shard++)
        {
            Tensor[] inShard = tensors.Where(tensor => ShardOf(tensor.Name) == shard).ToArray();
            WriteShard(Path.Combine(directory, shardNames[shard]), inShard);
            foreach (Tensor tensor in inShard)
                weightMap[tensor.Name] = shardNames[shard];
        }
        long total = tensors.Sum(static tensor => (long)tensor.Data.Length);
        File.WriteAllText(Path.Combine(directory, "model.safetensors.index.json"), JsonSerializer.Serialize(new
        {
            metadata = new { total_size = total },
            weight_map = weightMap,
        }));
        File.WriteAllText(Path.Combine(directory, "config.json"), Config(flavor));
        return total;
    }

    /// <summary>The config the checkpoint is written against.</summary>
    public static string Config(QuantFlavor flavor)
    {
        JsonObject root = JsonNode.Parse(DeepSeekV41Fixtures.OfficialConfig(text =>
        {
            text["num_hidden_layers"] = BackboneLayers;
            text["num_nextn_predict_layers"] = 1;
            text["compress_ratios"] = new JsonArray(0, 0, 2, 2, 0);
            text["engram_layer_ids"] = new JsonArray(EngramLayer);
            text["engram_num_embeddings"] = new JsonArray(EngramRows);
            text["kv_source_layer_ids"] = new JsonArray(2);
            text["index_source_layer_ids"] = new JsonArray(2);
            text["dspark_target_layer_ids"] = new JsonArray(1, 2, 3);
            text["n_routed_experts"] = RoutedExperts;
            text["num_experts_per_tok"] = 2;
            text["dspark_n_routed_experts"] = DraftExperts;
            text["dspark_num_experts_per_tok"] = 2;
        }))!.AsObject();
        if (flavor == QuantFlavor.Mlx)
        {
            root["quantization_config"] = new JsonObject { ["group_size"] = 64, ["bits"] = 4, ["mode"] = "affine" };
        }
        return root.ToJsonString();
    }

    private sealed record Tensor(string Name, string DType, long[] Shape, byte[] Data);

    private static int ShardOf(string name) => name.Contains(".engram.embed.", StringComparison.Ordinal) ? 2
        : name.Contains(".ffn.experts.", StringComparison.Ordinal) ? 1 : 0;

    private static void AddDense(List<Tensor> tensors, bool includeDraft)
    {
        tensors.Add(Bf16("embed.weight", 4, 2));
        tensors.Add(Bf16("head.weight", 4, 2));
        tensors.Add(Bf16("norm.weight", 2));
        tensors.Add(Bf16("image_start", 2));
        tensors.Add(Bf16("vision.patch.weight", 2, 2));
        tensors.Add(Bf16("aligner.w1.weight", 2, 2));
        for (int layer = 0; layer < BackboneLayers; layer++)
            tensors.Add(Bf16($"layers.{layer}.attn.wq_a.weight", 2, 2));
        tensors.Add(Bf16($"layers.{EngramLayer}.engram.wkv.weight", 2, 2));
        if (includeDraft)
        {
            tensors.Add(Bf16("mtp.0.main_proj.weight", 2, 2));
            tensors.Add(Bf16("mtp.0.attn.wq_a.weight", 2, 2));
        }
    }

    private static void AddExpert(List<Tensor> tensors, QuantFlavor flavor, string prefix, bool full)
    {
        foreach (string projection in new[] { "w1", "w2", "w3" })
        {
            string name = $"{prefix}.{projection}";
            if (flavor == QuantFlavor.Mlx)
            {
                tensors.Add(new Tensor($"{name}.weight", "U32", [2, 8], new byte[64]));
                if (full)
                {
                    tensors.Add(new Tensor($"{name}.scales", "BF16", [2, 1], new byte[4]));
                    tensors.Add(new Tensor($"{name}.biases", "BF16", [2, 1], new byte[4]));
                }
            }
            else
            {
                tensors.Add(new Tensor($"{name}.weight", "I8", [2, 32], new byte[64]));
                tensors.Add(new Tensor($"{name}.scale", "F8_E8M0", [2, 2], new byte[4]));
            }
        }
    }

    private static void AddEngram(List<Tensor> tensors)
    {
        string prefix = $"layers.{EngramLayer}.engram.embed";
        tensors.Add(new Tensor($"{prefix}.weight", "F8_E4M3", [EngramRows, 256], new byte[EngramRows * 256]));
        tensors.Add(new Tensor($"{prefix}.scale", "F8_E8M0", [EngramRows, 8], new byte[EngramRows * 8]));
    }

    private static Tensor Bf16(string name, params long[] shape) =>
        new(name, "BF16", shape, new byte[shape.Aggregate(2L, static (a, b) => a * b)]);

    private static void WriteShard(string path, Tensor[] tensors)
    {
        Dictionary<string, object> header = new();
        long offset = 0;
        foreach (Tensor tensor in tensors)
        {
            header[tensor.Name] = new Dictionary<string, object>
            {
                ["dtype"] = tensor.DType,
                ["shape"] = tensor.Shape,
                ["data_offsets"] = new long[] { offset, offset + tensor.Data.Length },
            };
            offset += tensor.Data.Length;
        }
        byte[] json = Encoding.UTF8.GetBytes(JsonSerializer.Serialize(header));
        using FileStream stream = new(path, FileMode.Create, FileAccess.Write);
        stream.Write(BitConverter.GetBytes((ulong)json.Length));
        stream.Write(json);
        foreach (Tensor tensor in tensors)
            stream.Write(tensor.Data);
    }
}
