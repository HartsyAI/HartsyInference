using System.Globalization;
using System.Text;
using HartsyInference.LLM.Ernie5;

namespace HartsyInference.Engine.Tests.Services;

/// <summary>Writes a tiny ERNIE-5.5 checkpoint directory (config, F32 safetensors with every key the model maps, byte-level tokenizer and chat
/// template) with seeded random weights. Layer 0 is an HCA attention with a dense FFN; layer 1 is a DSA attention with a latent MoE; layer 2 is
/// the multi-token-prediction layer, a DSA attention with a latent MoE like the real checkpoint's layer 43.</summary>
internal static class Ernie5SyntheticCheckpoint
{
    internal const int Vocab = 264, Hidden = 16, Hc = 2, Layers = 2, Heads = 2, HeadDim = 8, QLora = 8, OGroups = 1, OLoraRank = 8;
    internal const int MlaHeads = 2, MlaQLora = 8, MlaKvLora = 8, MlaNope = 4, MlaRope = 4, MlaV = 4, IndexHeads = 2, IndexHeadDim = 128;
    internal const int Dense = 12, MoeInter = 8, Latent = 6, Experts = 4, SharedExperts = 1, ExpertsPerTok = 2;
    internal const int BosId = 256, EosId = 257, MtpLayers = 1;

    /// <summary>The config as a checkpoint ships it; <paramref name="overrides"/> replaces or adds top-level fields (raw JSON values).</summary>
    internal static string ConfigJson(IReadOnlyDictionary<string, string>? overrides = null)
    {
        Dictionary<string, string> fields = new(StringComparer.Ordinal)
        {
            ["model_type"] = "\"ernie5_5_moe\"",
            ["vocab_size"] = Vocab.ToString(CultureInfo.InvariantCulture),
            ["hidden_size"] = Hidden.ToString(CultureInfo.InvariantCulture),
            ["num_hidden_layers"] = Layers.ToString(CultureInfo.InvariantCulture),
            ["num_nextn_predict_layers"] = MtpLayers.ToString(CultureInfo.InvariantCulture),
            ["num_attention_heads"] = Heads.ToString(CultureInfo.InvariantCulture),
            ["num_key_value_heads"] = "1",
            ["head_dim"] = HeadDim.ToString(CultureInfo.InvariantCulture),
            ["qk_nope_head_dim"] = "4",
            ["qk_rope_head_dim"] = "4",
            ["v_head_dim"] = HeadDim.ToString(CultureInfo.InvariantCulture),
            ["q_lora_rank"] = QLora.ToString(CultureInfo.InvariantCulture),
            ["kv_lora_rank"] = "8",
            ["o_lora_rank"] = OLoraRank.ToString(CultureInfo.InvariantCulture),
            ["o_groups"] = OGroups.ToString(CultureInfo.InvariantCulture),
            ["hybrid_mla_num_attention_heads"] = MlaHeads.ToString(CultureInfo.InvariantCulture),
            ["hybrid_mla_q_lora_rank"] = MlaQLora.ToString(CultureInfo.InvariantCulture),
            ["hybrid_mla_kv_lora_rank"] = MlaKvLora.ToString(CultureInfo.InvariantCulture),
            ["hybrid_mla_qk_nope_head_dim"] = MlaNope.ToString(CultureInfo.InvariantCulture),
            ["hybrid_mla_qk_rope_head_dim"] = MlaRope.ToString(CultureInfo.InvariantCulture),
            ["hybrid_mla_v_head_dim"] = MlaV.ToString(CultureInfo.InvariantCulture),
            ["intermediate_size"] = Dense.ToString(CultureInfo.InvariantCulture),
            ["moe_intermediate_size"] = MoeInter.ToString(CultureInfo.InvariantCulture),
            ["moe_latent_size"] = Latent.ToString(CultureInfo.InvariantCulture),
            ["n_routed_experts"] = Experts.ToString(CultureInfo.InvariantCulture),
            ["n_shared_experts"] = SharedExperts.ToString(CultureInfo.InvariantCulture),
            ["num_experts_per_tok"] = ExpertsPerTok.ToString(CultureInfo.InvariantCulture),
            ["first_k_dense_replace"] = "1",
            ["routed_scaling_factor"] = "1.0",
            ["routed_scaling_factor_learnable"] = "false",
            ["moe_split_feature_routing"] = "false",
            ["scoring_func"] = "\"sigmoid\"",
            ["norm_topk_prob"] = "true",
            ["n_group"] = "1",
            ["topk_group"] = "1",
            ["moe_shared_expert_gate"] = "false",
            ["latent_moe_use_norm"] = "true",
            ["hidden_act"] = "\"situ\"",
            ["activation_situ_beta"] = "4.0",
            ["activation_situ_linear_beta"] = "25.0",
            ["compress_ratios"] = "[2, -2, -2]",
            ["experimental_attention_variant"] = "\"dsv4_hybrid\"",
            ["use_gated_attn"] = "false",
            ["use_vha_attention"] = "false",
            ["rope_interleave"] = "true",
            ["enable_hyper_connections"] = "true",
            ["hc_mult"] = Hc.ToString(CultureInfo.InvariantCulture),
            ["hc_sinkhorn_iters"] = "3",
            ["hc_eps"] = "1e-6",
            ["index_n_heads"] = IndexHeads.ToString(CultureInfo.InvariantCulture),
            ["index_head_dim"] = IndexHeadDim.ToString(CultureInfo.InvariantCulture),
            ["index_topk"] = "4",
            ["sliding_window"] = "4",
            ["rope_theta"] = "10000.0",
            ["compress_rope_theta"] = "10000.0",
            ["rms_norm_eps"] = "1e-6",
            ["max_position_embeddings"] = "64",
            ["tie_word_embeddings"] = "false",
            ["bos_token_id"] = BosId.ToString(CultureInfo.InvariantCulture),
            ["eos_token_id"] = EosId.ToString(CultureInfo.InvariantCulture),
        };
        if (overrides is not null)
            foreach (KeyValuePair<string, string> kv in overrides) fields[kv.Key] = kv.Value;
        return "{" + string.Join(",\n", fields.Select(kv => $"\"{kv.Key}\": {kv.Value}")) + "}";
    }

    /// <summary>How the weights are stored: all F32; the ERNIE FP8 layout, where each linear is E4M3 with a <c>weight_scale_inv</c> grid; or the
    /// same values as <see cref="Fp8Dequantized"/>, every linear decoded to F32, the reference an FP8 run is compared with.</summary>
    internal enum Storage { F32, Fp8, Fp8Dequantized }

    /// <summary>The 128x128 block-FP8 <c>quantization_config</c> of the real checkpoint.</summary>
    internal static readonly IReadOnlyDictionary<string, string> Fp8Overrides = new Dictionary<string, string>(StringComparer.Ordinal)
    {
        ["quantization_config"] = "{\"quant_method\": \"fp8\", \"fmt\": \"e4m3\", \"weight_block_size\": [128, 128]}",
    };

    /// <summary>The FP8 block edge, as the real checkpoint stores it.</summary>
    internal const int Fp8Block = 128;

    /// <summary>The real checkpoint's dimensions: hidden 4096, 64 heads of 512 with the real MLA and indexer sizes, a 14336 dense FFN, 2048-wide
    /// latent experts, 128-wide FP8 blocks and the real attention flags. Three layers (HCA, DSA, HCA) with one dense layer first, 8 routed
    /// experts instead of 512 and a 4096-token vocabulary, so one forward fits on a local GPU. Every linear's shape follows from these, the
    /// same as the real index.</summary>
    internal static readonly IReadOnlyDictionary<string, string> RealDimensionOverrides = new Dictionary<string, string>(StringComparer.Ordinal)
    {
        ["hidden_size"] = "4096", ["num_hidden_layers"] = "3", ["num_nextn_predict_layers"] = "0",
        ["num_attention_heads"] = "64", ["num_key_value_heads"] = "1", ["head_dim"] = "512",
        ["qk_nope_head_dim"] = "448", ["qk_rope_head_dim"] = "64", ["v_head_dim"] = "512",
        ["q_lora_rank"] = "1024", ["kv_lora_rank"] = "512", ["o_lora_rank"] = "1024", ["o_groups"] = "8",
        ["hybrid_mla_num_attention_heads"] = "64", ["hybrid_mla_q_lora_rank"] = "1024", ["hybrid_mla_kv_lora_rank"] = "512",
        ["hybrid_mla_qk_nope_head_dim"] = "192", ["hybrid_mla_qk_rope_head_dim"] = "64", ["hybrid_mla_v_head_dim"] = "256",
        ["intermediate_size"] = "14336", ["moe_intermediate_size"] = "2048", ["moe_latent_size"] = "2048",
        ["n_routed_experts"] = "8", ["n_shared_experts"] = "1", ["num_experts_per_tok"] = "2", ["first_k_dense_replace"] = "1",
        ["compress_ratios"] = "[128, -2, 128]", ["hc_mult"] = "4", ["hc_sinkhorn_iters"] = "20",
        ["index_n_heads"] = "64", ["index_head_dim"] = "128", ["index_topk"] = "64", ["sliding_window"] = "128",
        ["max_position_embeddings"] = "4096", ["vocab_size"] = "4096", ["bos_token_id"] = "1", ["eos_token_id"] = "2",
        ["use_gated_attn"] = "true", ["use_vha_attention"] = "true", ["moe_split_feature_routing"] = "true",
        ["routed_scaling_factor"] = "3.0", ["routed_scaling_factor_learnable"] = "true", ["scoring_func"] = "\"sigmoid\"",
        ["rope_theta"] = "8000000.0", ["compress_rope_theta"] = "160000.0",
        ["activation_situ_beta"] = "4.0", ["activation_situ_linear_beta"] = "25.0",
    };

    /// <summary>Writes a complete checkpoint into <paramref name="directory"/> (created if absent) with weights from <paramref name="seed"/>.
    /// <paramref name="configOverrides"/> replaces or adds config fields on top of the storage's own.</summary>
    internal static void Write(string directory, int seed = 7, Storage storage = Storage.F32, IReadOnlyDictionary<string, string>? configOverrides = null)
    {
        Directory.CreateDirectory(directory);
        Dictionary<string, string> merged = new(StringComparer.Ordinal);
        if (storage == Storage.Fp8) foreach (KeyValuePair<string, string> kv in Fp8Overrides) merged[kv.Key] = kv.Value;
        if (configOverrides is not null) foreach (KeyValuePair<string, string> kv in configOverrides) merged[kv.Key] = kv.Value;
        string configJson = ConfigJson(merged.Count == 0 ? null : merged);
        File.WriteAllText(Path.Combine(directory, "config.json"), configJson);
        File.WriteAllText(Path.Combine(directory, "tokenizer.json"), TokenizerJson());
        File.WriteAllText(Path.Combine(directory, "tokenizer_config.json"), "{\"bos_token\": \"<s>\", \"eos_token\": \"</s>\"}");
        File.WriteAllText(Path.Combine(directory, "chat_template.jinja"), "{{ bos_token }}{% for message in messages %}{{ message['content'] }}{% endfor %}");
        Ernie5Config config = Ernie5Config.Parse(configJson);
        WriteSafetensors(Path.Combine(directory, "model.safetensors"), config, seed, storage);
    }

    /// <summary>A byte-level BPE tokenizer: ids 0..255 are the bytes, then the begin and end specials.</summary>
    internal static string TokenizerJson()
    {
        Dictionary<int, char> map = ByteToUnicode();
        StringBuilder vocab = new();
        for (int b = 0; b < 256; b++)
        {
            if (b > 0) vocab.Append(',');
            vocab.Append(System.Text.Json.JsonSerializer.Serialize(map[b].ToString())).Append(':').Append(b);
        }
        // one merge, so the tokenizer is a BPE and not a bare alphabet: "h" + "i" -> "hi" (id 258)
        vocab.Append(",\"hi\":258");
        return "{\"version\":\"1.0\",\"model\":{\"type\":\"BPE\",\"vocab\":{" + vocab + "},\"merges\":[\"h i\"]},"
            + "\"added_tokens\":["
            + $"{{\"id\":{BosId},\"content\":\"<s>\",\"special\":true}},{{\"id\":{EosId},\"content\":\"</s>\",\"special\":true}}]}}";
    }

    // The GPT-2 byte-level alphabet: printable bytes map to themselves, the rest to code points above 255.
    private static Dictionary<int, char> ByteToUnicode()
    {
        List<int> bytes = [];
        for (int b = 33; b <= 126; b++) bytes.Add(b);
        for (int b = 161; b <= 172; b++) bytes.Add(b);
        for (int b = 174; b <= 255; b++) bytes.Add(b);
        List<int> codes = [.. bytes];
        int extra = 0;
        for (int b = 0; b < 256; b++)
        {
            if (bytes.Contains(b)) continue;
            bytes.Add(b);
            codes.Add(256 + extra++);
        }
        Dictionary<int, char> map = new();
        for (int i = 0; i < bytes.Count; i++) map[bytes[i]] = (char)codes[i];
        return map;
    }

    private sealed record Entry(string Name, string DType, int[] Shape, byte[] Data);

    private static void WriteSafetensors(string path, Ernie5Config config, int seed, Storage storage)
    {
        Random rng = new(seed);
        List<Entry> entries = [];
        foreach (string key in Ernie5WeightMap.ExpectedKeys(config).Concat(Ernie5WeightMap.ExpectedMtpKeys(config)))
        {
            int[] shape = ShapeOf(key, config);
            int count = shape.Aggregate(1, (a, n) => a * n);
            if (storage != Storage.F32 && IsBf16Tensor(key))
            {
                // The real checkpoint stores these BF16, rounded to nearest even; the twin holds the same rounded values in F32.
                float[] rounded = ValuesOf(key, count, rng).Select(RoundToBf16).ToArray();
                entries.Add(storage == Storage.Fp8
                    ? new Entry(key, "BF16", shape, Bf16Bytes(rounded))
                    : new Entry(key, "F32", shape, FloatBytes(rounded)));
                continue;
            }
            if (storage == Storage.F32 || !IsFp8Linear(key, shape))
            {
                entries.Add(new Entry(key, "F32", shape, FloatBytes(ValuesOf(key, count, rng))));
                continue;
            }
            // Same draw order for both FP8 storages, so the dequantized twin holds exactly the values the FP8 file decodes to.
            byte[] raw = Fp8BytesOf(count, rng);
            float[] scale = ScaleValues(shape, rng);
            if (storage == Storage.Fp8)
            {
                entries.Add(new Entry(key, "F8_E4M3", shape, raw));
                entries.Add(new Entry(key + Ernie5WeightNames.ScaleSuffix, "F32", ScaleShape(shape), FloatBytes(scale)));
            }
            else
            {
                entries.Add(new Entry(key, "F32", shape, FloatBytes(DequantizedValues(raw, scale, shape))));
            }
        }
        entries.Sort(static (a, b) => string.CompareOrdinal(a.Name, b.Name));

        StringBuilder header = new("{");
        long offset = 0;
        for (int i = 0; i < entries.Count; i++)
        {
            Entry e = entries[i];
            long bytes = e.Data.LongLength;
            if (i > 0) header.Append(',');
            header.Append($"\"{e.Name}\":{{\"dtype\":\"{e.DType}\",\"shape\":[{string.Join(",", e.Shape)}],\"data_offsets\":[{offset},{offset + bytes}]}}");
            offset += bytes;
        }
        header.Append('}');
        // the header is padded with spaces to an 8-byte boundary, as the format recommends
        while (header.Length % 8 != 0) header.Append(' ');
        byte[] headerBytes = Encoding.UTF8.GetBytes(header.ToString());

        using FileStream stream = File.Create(path);
        using BinaryWriter writer = new(stream);
        writer.Write((long)headerBytes.Length);
        writer.Write(headerBytes);
        foreach (Entry e in entries) writer.Write(e.Data);
    }

    private static byte[] FloatBytes(float[] values)
    {
        byte[] bytes = new byte[values.Length * sizeof(float)];
        Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
        return bytes;
    }

    // The tensors the real checkpoint stores BF16: the embedding and LM head, the routers, the compressor and the indexer projections.
    private static bool IsBf16Tensor(string key) =>
        // "block_sparse_moe.gate.weight" also matches the split-feature second router, gate.weight_1, which must stay BF16 too.
        key == Ernie5WeightNames.Embedding || key == Ernie5WeightNames.LmHead || key.Contains("block_sparse_moe.gate.weight", StringComparison.Ordinal)
        || key.Contains("compressor.linear_w", StringComparison.Ordinal) || key.Contains("indexer.wk.", StringComparison.Ordinal)
        || key.Contains("indexer.weights_proj", StringComparison.Ordinal);

    // Round to nearest, ties to even, keeping the top 16 bits: the BF16 value of a float.
    private static float RoundToBf16(float v)
    {
        uint bits = BitConverter.SingleToUInt32Bits(v);
        bits = (bits + 0x7FFFu + ((bits >> 16) & 1u)) & 0xFFFF0000u;
        return BitConverter.UInt32BitsToSingle(bits);
    }

    private static byte[] Bf16Bytes(float[] values)
    {
        byte[] bytes = new byte[values.Length * 2];
        for (int i = 0; i < values.Length; i++)
        {
            ushort top = (ushort)(BitConverter.SingleToUInt32Bits(values[i]) >> 16);
            bytes[2 * i] = (byte)top;
            bytes[2 * i + 1] = (byte)(top >> 8);
        }
        return bytes;
    }

    // The FP8 linears: the 2-D projections the real checkpoint stores as E4M3. The hyper-connection mappings and the other F32 tensors stay F32;
    // the tensors listed in IsBf16Tensor (embedding, LM head, routers, compressor and indexer projections) are BF16 and are handled first.
    private static bool IsFp8Linear(string key, int[] shape)
    {
        if (shape.Length != 2) return false;
        if (!(key.EndsWith(".weight", StringComparison.Ordinal) || key.EndsWith("linear_o_group_proj", StringComparison.Ordinal))) return false;
        return !(key.Contains("embed_tokens", StringComparison.Ordinal) || key.Contains("lm_head", StringComparison.Ordinal)
            || key.Contains("gate.weight", StringComparison.Ordinal) || key.Contains("mapping_proj", StringComparison.Ordinal)
            || key.Contains("compressor.linear_w", StringComparison.Ordinal) || key.Contains("indexer.wk.", StringComparison.Ordinal)
            || key.Contains("weights_proj", StringComparison.Ordinal));
    }

    // Any byte is a finite E4M3 value except the two NaN codes, 0x7F and 0xFF, which map to their neighbours.
    private static byte[] Fp8BytesOf(int count, Random rng)
    {
        byte[] bytes = new byte[count];
        for (int i = 0; i < count; i++)
        {
            byte v = (byte)rng.Next(256);
            if ((v & 0x7F) == 0x7F) v &= 0xFE;
            bytes[i] = v;
        }
        return bytes;
    }

    private static int[] ScaleShape(int[] shape) => [(shape[0] + Fp8Block - 1) / Fp8Block, (shape[1] + Fp8Block - 1) / Fp8Block];

    private static float[] ScaleValues(int[] shape, Random rng)
    {
        int[] grid = ScaleShape(shape);
        float[] scale = new float[grid[0] * grid[1]];
        for (int i = 0; i < scale.Length; i++) scale[i] = (float)(0.001 + rng.NextDouble() * 0.009);
        return scale;
    }

    // Decodes the bytes with their block scales: W[r, c] = e4m3(raw[r, c]) * scale[r / 128, c / 128], the F32 weight the FP8 file stands for.
    private static float[] DequantizedValues(byte[] raw, float[] scale, int[] shape)
    {
        int rows = shape[0], cols = shape[1], gridCols = ScaleShape(shape)[1];
        float[] values = new float[rows * cols];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                values[r * cols + c] = E4m3ToFloat(raw[r * cols + c]) * scale[(r / Fp8Block) * gridCols + c / Fp8Block];
        return values;
    }

    // IEEE-style E4M3: 1 sign, 4 exponent bits (bias 7), 3 mantissa bits; exponent 0 is subnormal.
    private static float E4m3ToFloat(byte b)
    {
        int sign = b >> 7, exponent = (b >> 3) & 0xF, mantissa = b & 0x7;
        float magnitude = exponent == 0
            ? mantissa / 8f * MathF.Pow(2, -6)
            : (1 + mantissa / 8f) * MathF.Pow(2, exponent - 7);
        return sign == 1 ? -magnitude : magnitude;
    }

    private static float[] ValuesOf(string key, int count, Random rng)
    {
        float[] values = new float[count];
        for (int i = 0; i < count; i++)
        {
            if (key.EndsWith("norm.weight", StringComparison.Ordinal)) values[i] = 1f;
            else if (key.EndsWith("norm.bias", StringComparison.Ordinal) || key.EndsWith(".bias", StringComparison.Ordinal)
                || key.EndsWith("attn_sink", StringComparison.Ordinal) || key.EndsWith("softmax_offset", StringComparison.Ordinal)
                || key.EndsWith("e_score_correction_bias", StringComparison.Ordinal) || key.EndsWith("hc_head_base", StringComparison.Ordinal)) values[i] = 0f;
            else if (key.EndsWith("hc_head_scale", StringComparison.Ordinal)) values[i] = 1f;
            else if (key.Contains("_hyper_connection.alpha_", StringComparison.Ordinal)) values[i] = 0.05f;
            else values[i] = (float)((rng.NextDouble() * 2 - 1) * 0.05);
        }
        return values;
    }

    private static int[] ShapeOf(string key, Ernie5Config c)
    {
        int hc = c.HcMult, mixes = (2 + hc) * hc, flat = hc * c.HiddenSize, h = c.HiddenSize;
        if (key == Ernie5WeightNames.Embedding || key == Ernie5WeightNames.LmHead) return [c.VocabSize, h];
        if (key == Ernie5WeightNames.FinalNorm) return [h];
        // the main head and each MTP layer's own head share the hc_head_* leaf names
        if (key.EndsWith(Ernie5WeightNames.MtpHeadFn, StringComparison.Ordinal)) return [flat, hc];
        if (key.EndsWith(Ernie5WeightNames.MtpHeadBase, StringComparison.Ordinal)) return [hc];
        if (key.EndsWith(Ernie5WeightNames.MtpHeadScale, StringComparison.Ordinal)) return [1];
        if (key.Contains("_hyper_connection.mapping_proj.weight", StringComparison.Ordinal)) return [flat, mixes];
        if (key.Contains("_hyper_connection.bias", StringComparison.Ordinal)) return [mixes];
        if (key.Contains("_hyper_connection.alpha_", StringComparison.Ordinal)) return [1];
        if (key.EndsWith("input_layernorm.weight", StringComparison.Ordinal) || key.EndsWith("post_attention_layernorm.weight", StringComparison.Ordinal))
            return [h];

        int layer = LayerOf(key);
        string leaf = key[(key.IndexOf('.', "model.layers.".Length) + 1)..];
        if (leaf is Ernie5WeightNames.MtpEmbedNorm or Ernie5WeightNames.MtpHiddenNorm or Ernie5WeightNames.MtpFinalNorm) return [h];
        if (leaf is Ernie5WeightNames.MtpEmbedProj or Ernie5WeightNames.MtpHiddenProj) return [h, h];
        // the attention output gate reads the q LoRA; its width is the grouped o-projection's (HCA) or the heads' value width (DSA)
        if (c.UseGatedAttn && leaf == "self_attn." + Ernie5WeightNames.AttnGate)
            return c.IsDsaLayer(layer) ? [c.MlaHeads * c.MlaVHeadDim, c.MlaQLoraRank] : [c.OGroups * c.OLoraRank, c.QLoraRank];
        // the VHA post-mix matrices are [heads, 16], with the head count Ernie5Layer folds them with: MLA heads on DSA layers, attention heads on HCA
        if (c.UseVhaAttention && Ernie5WeightNames.VhaLeaves.Any(v => leaf == "self_attn." + v))
            return [c.IsDsaLayer(layer) ? c.MlaHeads : c.NumAttentionHeads, 16];
        if (leaf.StartsWith("self_attn.", StringComparison.Ordinal))
            return c.IsDsaLayer(layer) ? DsaShape(leaf, c) : HcaShape(leaf, c);
        if (leaf.StartsWith("mlp.", StringComparison.Ordinal) && layer < c.FirstKDenseReplace)
            return leaf.StartsWith("mlp.down_proj", StringComparison.Ordinal) ? [h, c.IntermediateSize] : [c.IntermediateSize, h];
        return MoeShape(leaf, c);
    }

    private static int[] HcaShape(string leaf, Ernie5Config c)
    {
        int grouped = c.OGroups * c.OLoraRank, groupDim = c.NumAttentionHeads * c.HeadDim / c.OGroups, ratio = c.CompressRatios[0];
        return leaf switch
        {
            "self_attn.linear_q_down_proj.weight" => [c.QLoraRank, c.HiddenSize],
            "self_attn.q_layernorm.weight" => [c.QLoraRank],
            "self_attn.linear_q_up_proj.weight" => [c.NumAttentionHeads * c.HeadDim, c.QLoraRank],
            "self_attn.linear_kv_proj.weight" => [c.HeadDim, c.HiddenSize],
            "self_attn.kv_layernorm.weight" => [c.HeadDim],
            "self_attn.linear_o_group_proj" => [grouped, groupDim],
            "self_attn.o_proj.weight" => [c.HiddenSize, grouped],
            "self_attn.core_attention.attn_sink" => [c.NumAttentionHeads],
            "self_attn.core_attention.compressor.linear_wkv.weight" => [c.HeadDim, c.HiddenSize],
            "self_attn.core_attention.compressor.linear_wgate.weight" => [c.HeadDim, c.HiddenSize],
            "self_attn.core_attention.compressor.ape" => [ratio * c.HeadDim],
            "self_attn.core_attention.compressor.norm.weight" => [c.HeadDim],
            _ => throw new InvalidOperationException($"No HCA shape for '{leaf}'."),
        };
    }

    private static int[] DsaShape(string leaf, Ernie5Config c)
    {
        int heads = c.MlaHeads, qk = c.MlaQkNopeHeadDim + c.MlaQkRopeHeadDim, latent = c.MlaKvLoraRank + c.MlaQkRopeHeadDim;
        return leaf switch
        {
            "self_attn.q_a_proj.weight" => [c.MlaQLoraRank, c.HiddenSize],
            "self_attn.q_a_layernorm.weight" => [c.MlaQLoraRank],
            "self_attn.q_b_proj.weight" => [heads * qk, c.MlaQLoraRank],
            "self_attn.kv_a_proj_with_mqa.weight" => [latent, c.HiddenSize],
            "self_attn.kv_a_layernorm.weight" => [c.MlaKvLoraRank],
            "self_attn.kv_b_proj.weight" => [heads * (c.MlaQkNopeHeadDim + c.MlaVHeadDim), c.MlaKvLoraRank],
            "self_attn.o_proj.weight" => [c.HiddenSize, heads * c.MlaVHeadDim],
            "self_attn.core_attention.softmax_offset" => [heads],
            "self_attn.core_attention.indexer.wq_b.weight" => [c.IndexNHeads * c.IndexHeadDim, c.MlaQLoraRank],
            "self_attn.core_attention.indexer.wk.weight" => [c.IndexHeadDim, c.HiddenSize],
            "self_attn.core_attention.indexer.k_norm.weight" => [c.IndexHeadDim],
            "self_attn.core_attention.indexer.k_norm.bias" => [c.IndexHeadDim],
            "self_attn.core_attention.indexer.weights_proj.weight" => [c.IndexNHeads, c.HiddenSize],
            _ => throw new InvalidOperationException($"No DSA shape for '{leaf}'."),
        };
    }

    private static int[] MoeShape(string leaf, Ernie5Config c)
    {
        int shared = c.NSharedExperts * c.MoeIntermediateSize, latent = c.MoeLatentSize;
        if (leaf is "block_sparse_moe.gate.weight" or "block_sparse_moe.gate.weight_1") return [c.NRoutedExperts, c.HiddenSize];
        if (leaf == "block_sparse_moe.gate.routed_scaling_factor_param") return [c.NRoutedExperts];
        if (leaf == "block_sparse_moe.e_score_correction_bias") return [c.NRoutedExperts];
        if (leaf == "block_sparse_moe.fc1_latent_proj.weight") return [latent, c.HiddenSize];
        if (leaf == "block_sparse_moe.latent_norm.weight") return [latent];
        if (leaf == "block_sparse_moe.fc2_latent_proj.weight") return [c.HiddenSize, latent];
        if (leaf.StartsWith("block_sparse_moe.shared_experts.", StringComparison.Ordinal))
            return leaf.Contains(".w2.", StringComparison.Ordinal) ? [c.HiddenSize, shared] : [shared, c.HiddenSize];
        if (leaf.StartsWith("block_sparse_moe.experts.", StringComparison.Ordinal))
            return leaf.Contains(".w2.", StringComparison.Ordinal) ? [latent, c.MoeIntermediateSize] : [c.MoeIntermediateSize, latent];
        throw new InvalidOperationException($"No MoE shape for '{leaf}'.");
    }

    private static int LayerOf(string key)
    {
        const string prefix = "model.layers.";
        int end = key.IndexOf('.', prefix.Length);
        return int.Parse(key.AsSpan(prefix.Length, end - prefix.Length), NumberStyles.None, CultureInfo.InvariantCulture);
    }
}
