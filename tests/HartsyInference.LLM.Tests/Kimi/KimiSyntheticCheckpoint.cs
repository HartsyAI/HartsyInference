using System.Globalization;
using System.Text;
using System.Text.Json;
using HartsyInference.LLM.Kimi;

namespace HartsyInference.LLM.Tests.Kimi;

/// <summary>Writes tiny Kimi K3.1 checkpoint directories under the published tensor names, with seeded random weights: <c>config.json</c>, one or
/// more safetensors shards and their <c>model.safetensors.index.json</c>. The default decoder has KDA, MLA, KDA, MLA layers with a dense first
/// layer and latent MoE after it. <see cref="WriteSharded"/> splits the tensors over several shard files, and can store the routed experts
/// MXFP4-packed and the dense weights BF16.</summary>
internal static class KimiSyntheticCheckpoint
{
    /// <summary>The decoder <see cref="Write"/> writes: four layers, two of them MLA, one dense layer.</summary>
    internal static readonly KimiK31Config DefaultConfig = new()
    {
        Hidden = 16, Vocab = 20, Layers = 4, DenseLayers = 1, DenseIntermediate = 24, Heads = 2, KdaHeadDim = 8, KdaForgetRank = 128,
        ConvKernel = 4, FullAttentionLayers = new HashSet<int> { 1, 3 }, QLoraRank = 8, KvLoraRank = 6, QkNopeDim = 4, QkRopeDim = 4, VDim = 4,
        IndexHeads = 2, IndexHeadDim = 8, IndexTopK = 16, Experts = 4, ExpertsPerToken = 2, MoeIntermediate = 8, SharedExperts = 1,
        LatentSize = 8, AttnResBlock = 2, RmsNormEps = 1e-5f,
    };

    internal static readonly int Hidden = DefaultConfig.Hidden, Vocab = DefaultConfig.Vocab, Layers = DefaultConfig.Layers;

    /// <summary>Divides by 2 and by 4 (4 heads, 8 experts, 32-wide experts and shared expert): the 2- and 4-rank checkpoints. The packed MXFP4
    /// experts need their K dimension in 32-column blocks, hence the 32-wide latent and expert widths.</summary>
    internal static readonly KimiK31Config ShardedConfig = DefaultConfig with
    {
        Heads = 4, Experts = 8, ExpertsPerToken = 2, MoeIntermediate = 32, LatentSize = 32,
    };

    /// <summary>Divides by 8 as well (8 heads, 24 vocabulary rows): the 8-rank checkpoint.</summary>
    internal static readonly KimiK31Config ShardedEightConfig = DefaultConfig with
    {
        Heads = 8, Vocab = 24, Experts = 8, ExpertsPerToken = 2, MoeIntermediate = 32, LatentSize = 32,
    };

    /// <summary>One tensor of a synthetic checkpoint: its name, stored dtype (<c>F32</c>, <c>BF16</c> or <c>U8</c>) and shape.</summary>
    internal sealed record TensorSpec(string Name, string Dtype, long[] Shape);

    internal static void Write(string directory, int seed = 7) => WriteSharded(directory, DefaultConfig, shardFiles: 1, packedExperts: false, bf16Dense: false, seed);

    /// <summary>Writes <paramref name="config"/>'s checkpoint over <paramref name="shardFiles"/> shard files, each holding a contiguous run of the tensors,
    /// and an index that maps every tensor to its file.</summary>
    /// <param name="packedExperts">Store the routed experts as MXFP4 (<c>weight_packed</c> U8 with E8M0 <c>weight_scale</c>) instead of F32 or BF16.</param>
    /// <param name="bf16Dense">Store the dense and projection weights as BF16 (the values are rounded to BF16 on write); norms and small tensors stay F32.</param>
    internal static void WriteSharded(string directory, KimiK31Config config, int shardFiles, bool packedExperts, bool bf16Dense, int seed = 7)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(shardFiles, 1);
        Directory.CreateDirectory(directory);
        File.WriteAllText(Path.Combine(directory, "config.json"), ConfigJson(config));

        List<TensorSpec> specs = Specs(config, packedExperts, bf16Dense);
        Random rng = new(seed);
        Dictionary<string, string> weightMap = new(StringComparer.Ordinal);
        for (int s = 0; s < shardFiles; s++)
        {
            string file = $"model-{s + 1:00000}-of-{shardFiles:00000}.safetensors";
            int from = (int)((long)s * specs.Count / shardFiles), to = (int)((long)(s + 1) * specs.Count / shardFiles);
            List<TensorSpec> inShard = specs.GetRange(from, to - from);
            WriteShard(Path.Combine(directory, file), inShard, rng);
            foreach (TensorSpec spec in inShard) weightMap[spec.Name] = file;
        }
        File.WriteAllText(Path.Combine(directory, "model.safetensors.index.json"),
            JsonSerializer.Serialize(new Dictionary<string, object> { ["weight_map"] = weightMap }));
    }

    /// <summary>The tensors of <paramref name="c"/>'s checkpoint, in the published names.</summary>
    internal static List<TensorSpec> Specs(KimiK31Config c, bool packedExperts, bool bf16Dense)
    {
        const string root = "language_model.model.";
        string lin = bf16Dense ? "BF16" : "F32";
        int key = c.KeyDim, qk = c.QkNopeDim + c.QkRopeDim, h = c.Hidden, sh = c.SharedExperts * c.MoeIntermediate;
        List<TensorSpec> t =
        [
            S(root + "embed_tokens.weight", lin, c.Vocab, h),
            S(root + "norm.weight", "F32", h),
            S("language_model.lm_head.weight", lin, c.Vocab, h),
            S(root + "output_attn_res_proj.weight", "F32", 1, h),
            S(root + "output_attn_res_norm.weight", "F32", h),
        ];
        for (int i = 0; i < c.Layers; i++)
        {
            string p = $"{root}layers.{i}.", a = p + "self_attn.";
            t.Add(S(p + "input_layernorm.weight", "F32", h));
            t.Add(S(p + "post_attention_layernorm.weight", "F32", h));
            t.Add(S(p + "self_attention_res_proj.weight", "F32", 1, h));
            t.Add(S(p + "self_attention_res_norm.weight", "F32", h));
            t.Add(S(p + "mlp_res_proj.weight", "F32", 1, h));
            t.Add(S(p + "mlp_res_norm.weight", "F32", h));
            if (c.IsMla(i))
            {
                t.Add(S(a + "kv_a_proj_with_mqa.weight", lin, c.KvLoraRank + c.QkRopeDim, h));
                t.Add(S(a + "q_a_proj.weight", lin, c.QLoraRank, h));
                t.Add(S(a + "q_b_proj.weight", lin, c.Heads * qk, c.QLoraRank));
                t.Add(S(a + "kv_b_proj.weight", lin, c.Heads * (c.QkNopeDim + c.VDim), c.KvLoraRank));
                t.Add(S(a + "o_proj.weight", lin, h, c.Heads * c.VDim));
                t.Add(S(a + "g_proj.weight", lin, c.Heads * c.VDim, h));
                t.Add(S(a + "q_a_layernorm.weight", "F32", c.QLoraRank));
                t.Add(S(a + "kv_a_layernorm.weight", "F32", c.KvLoraRank));
                t.Add(S(a + "indexer.wq_b.weight", lin, c.IndexHeads * c.IndexHeadDim, c.QLoraRank));
                t.Add(S(a + "indexer.wk.weight", lin, c.IndexHeadDim, h));
                t.Add(S(a + "indexer.k_norm.weight", "F32", c.IndexHeadDim));
                t.Add(S(a + "indexer.k_norm.bias", "F32", c.IndexHeadDim));
                t.Add(S(a + "indexer.weights_proj.weight", lin, c.IndexHeads, h));
            }
            else
            {
                foreach (string proj in new[] { "q_proj", "k_proj", "v_proj" }) t.Add(S(a + proj + ".weight", lin, key, h));
                foreach (string conv in new[] { "q_conv1d", "k_conv1d", "v_conv1d" }) t.Add(S(a + conv + ".weight", "F32", key, 1, c.ConvKernel));
                t.Add(S(a + "A_log", "F32", c.Heads));
                t.Add(S(a + "dt_bias", "F32", key));
                t.Add(S(a + "f_a_proj.weight", lin, c.KdaForgetRank, h));
                t.Add(S(a + "f_b_proj.weight", lin, key, c.KdaForgetRank));
                t.Add(S(a + "b_proj.weight", lin, c.Heads, h));
                t.Add(S(a + "g_proj.weight", lin, key, h));
                t.Add(S(a + "o_norm.weight", "F32", c.KdaHeadDim));
                t.Add(S(a + "o_proj.weight", lin, h, key));
            }
            if (c.IsDense(i))
            {
                t.Add(S(p + "mlp.gate_proj.weight", lin, c.DenseIntermediate, h));
                t.Add(S(p + "mlp.up_proj.weight", lin, c.DenseIntermediate, h));
                t.Add(S(p + "mlp.down_proj.weight", lin, h, c.DenseIntermediate));
                continue;
            }
            string m = p + "block_sparse_moe.";
            t.Add(S(m + "gate.weight", "F32", c.Experts, h));
            t.Add(S(m + "gate.e_score_correction_bias", "F32", c.Experts));
            t.Add(S(m + "routed_expert_down_proj.weight", lin, c.LatentSize, h));
            t.Add(S(m + "routed_expert_norm.weight", "F32", c.LatentSize));
            t.Add(S(m + "routed_expert_up_proj.weight", lin, h, c.LatentSize));
            t.Add(S(m + "shared_experts.gate_proj.weight", lin, sh, h));
            t.Add(S(m + "shared_experts.up_proj.weight", lin, sh, h));
            t.Add(S(m + "shared_experts.down_proj.weight", lin, h, sh));
            for (int e = 0; e < c.Experts; e++)
            {
                string x = $"{m}experts.{e}";
                if (packedExperts)
                {
                    foreach ((string proj, int rows, int cols) in new[]
                    {
                        ("w1", c.MoeIntermediate, c.LatentSize), ("w3", c.MoeIntermediate, c.LatentSize), ("w2", c.LatentSize, c.MoeIntermediate),
                    })
                    {
                        t.Add(S($"{x}.{proj}.weight_packed", "U8", rows, cols / 2));
                        t.Add(S($"{x}.{proj}.weight_scale", "U8", rows, cols / 32));
                    }
                }
                else
                {
                    t.Add(S($"{x}.w1.weight", lin, c.MoeIntermediate, c.LatentSize));
                    t.Add(S($"{x}.w3.weight", lin, c.MoeIntermediate, c.LatentSize));
                    t.Add(S($"{x}.w2.weight", lin, c.LatentSize, c.MoeIntermediate));
                }
            }
        }
        return t;
    }

    internal static string ConfigJson(KimiK31Config c)
    {
        string oneBased = string.Join(", ", c.FullAttentionLayers.Select(i => i + 1).OrderBy(i => i));
        string Float(float v) => v.ToString("R", CultureInfo.InvariantCulture);
        return "{"
            + "\"model_type\": \"kimi_k31\", \"text_config\": {"
            + "\"model_type\": \"kimi_linear\", \"hidden_act\": \"situ\", "
            + $"\"activation_situ_beta\": {Float(c.SituBeta)}, \"activation_situ_linear_beta\": {Float(c.SituLinearBeta)},"
            + $"\"attn_res_block_size\": {c.AttnResBlock}, \"dsa_config\": {{\"index_num_heads\": {c.IndexHeads}, \"index_head_dim\": {c.IndexHeadDim}, \"index_topk\": {c.IndexTopK}}},"
            + $"\"first_k_dense_replace\": {c.DenseLayers}, \"hidden_size\": {c.Hidden}, \"intermediate_size\": {c.DenseIntermediate}, \"kv_lora_rank\": {c.KvLoraRank},"
            + $"\"linear_attn_config\": {{\"full_attn_layers\": [{oneBased}], \"gate_lower_bound\": {Float(c.GateLowerBound)}, \"head_dim\": {c.KdaHeadDim}, \"num_heads\": {c.Heads}, \"short_conv_kernel_size\": {c.ConvKernel}}},"
            + $"\"moe_intermediate_size\": {c.MoeIntermediate}, \"num_attention_heads\": {c.Heads}, \"num_experts\": {c.Experts}, \"num_experts_per_token\": {c.ExpertsPerToken},"
            + $"\"num_hidden_layers\": {c.Layers}, \"num_key_value_heads\": {c.Heads}, \"num_shared_experts\": {c.SharedExperts}, \"q_lora_rank\": {c.QLoraRank},"
            + $"\"qk_nope_head_dim\": {c.QkNopeDim}, \"qk_rope_head_dim\": {c.QkRopeDim}, \"rms_norm_eps\": {Float(c.RmsNormEps)}, \"routed_expert_hidden_size\": {c.LatentSize},"
            + $"\"v_head_dim\": {c.VDim}, \"vocab_size\": {c.Vocab}"
            + "}}";
    }

    /// <summary>Bytes one element of <paramref name="dtype"/> takes.</summary>
    internal static int ElementBytes(string dtype) => dtype switch
    {
        "F32" => 4,
        "BF16" => 2,
        "U8" => 1,
        _ => throw new ArgumentException($"Unsupported synthetic dtype '{dtype}'.", nameof(dtype)),
    };

    private static TensorSpec S(string name, string dtype, params long[] shape) => new(name, dtype, shape);

    // A safetensors shard: an 8-byte little-endian header length, the JSON header of {name: {dtype, shape, data_offsets}}, then the data.
    private static void WriteShard(string path, List<TensorSpec> tensors, Random rng)
    {
        Dictionary<string, object> header = new(StringComparer.Ordinal);
        long offset = 0;
        List<byte[]> data = new(tensors.Count);
        foreach (TensorSpec spec in tensors)
        {
            byte[] bytes = Values(spec, rng);
            header[spec.Name] = new Dictionary<string, object>
            {
                ["dtype"] = spec.Dtype,
                ["shape"] = spec.Shape,
                ["data_offsets"] = new long[] { offset, offset + bytes.Length },
            };
            offset += bytes.Length;
            data.Add(bytes);
        }
        byte[] headerBytes = Encoding.UTF8.GetBytes(JsonSerializer.Serialize(header));
        using FileStream file = File.Create(path);
        using BinaryWriter writer = new(file);
        writer.Write((long)headerBytes.Length);
        writer.Write(headerBytes);
        foreach (byte[] bytes in data) writer.Write(bytes);
    }

    private static byte[] Values(TensorSpec spec, Random rng)
    {
        long count = 1;
        foreach (long d in spec.Shape) count *= d;
        bool norm = spec.Name.EndsWith("norm.weight", StringComparison.Ordinal);
        switch (spec.Dtype)
        {
            case "U8":
            {
                // Packed nibbles are random; an E8M0 scale stays near 2^0.
                byte[] packed = new byte[count];
                if (spec.Name.EndsWith("weight_scale", StringComparison.Ordinal))
                {
                    for (long i = 0; i < count; i++) packed[i] = (byte)(124 + rng.Next(8));
                }
                else
                {
                    rng.NextBytes(packed);
                }
                return packed;
            }
            case "F32":
            {
                float[] values = new float[count];
                for (long i = 0; i < count; i++) values[i] = norm ? 1f : (float)(rng.NextDouble() * 0.2 - 0.1);
                byte[] bytes = new byte[count * 4];
                Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
                return bytes;
            }
            case "BF16":
            {
                byte[] bytes = new byte[count * 2];
                for (long i = 0; i < count; i++)
                {
                    float v = (float)(rng.NextDouble() * 0.2 - 0.1);
                    ushort bf16 = (ushort)(BitConverter.SingleToUInt32Bits(v) >> 16);
                    bytes[i * 2] = (byte)(bf16 & 0xFF);
                    bytes[i * 2 + 1] = (byte)(bf16 >> 8);
                }
                return bytes;
            }
            default:
                throw new ArgumentException($"Unsupported synthetic dtype '{spec.Dtype}'.", nameof(spec));
        }
    }
}
