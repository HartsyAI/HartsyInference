using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.ModelAssets.CheckpointConverters;

/// <summary>Splits the Comfy-Org YuE2 single-file checkpoint (<c>checkpoints/yue2_3b_*.safetensors</c>) into the
/// three stacks the engine loads separately: the autoregressive planner/semantic LM, the non-autoregressive acoustic
/// transformer, and the Oobleck VAE.</summary>
/// <remarks><para>The repack carries three prefixes — <c>text_encoders.</c>, <c>model.diffusion_model.</c> and
/// <c>vae.</c> — and merges each attention block's Q/K/V into one <c>qkv_proj</c> and each MLP's gate/up into one
/// <c>gate_up_proj</c>. <see cref="GenericTransformer"/> wants them split, so this undoes the merge.</para>
///
/// <para>Both the split order (Q ‖ K ‖ V row-wise, gate ‖ up row-wise) and the fact that Comfy <b>duplicates</b> the
/// original checkpoint's single shared <c>model.norm</c> into both stacks were proven by diffing every tensor of
/// <c>Comfy-Org/YuE2</c> against <c>m-a-p/YuE2-3B</c>: 461/461 bit-identical. The original interleaves the two
/// stacks in one layer list (<c>self_attn</c>/<c>mlp</c> against <c>nar_self_attn</c>/<c>nar_mlp</c>), which is
/// where the NAR stack's <c>post_attention_layernorm</c> gets its odd upstream name,
/// <c>nar_pre_mlp_layernorm</c>.</para></remarks>
public static class Yue2CheckpointConverter
{
    private const string ArPrefix = "text_encoders.";
    private const string NarPrefix = "model.diffusion_model.";
    private const string VaePrefix = "vae.";

    /// <summary>The embedded HuggingFace <c>tokenizers</c> JSON. YuE2 ships no side tokenizer file — the whole
    /// 8 MB document rides in the checkpoint as a U8 tensor.</summary>
    public const string TokenizerKey = "text_encoders.yue2_tokenizer_json";

    /// <summary>True when the header is a Comfy-Org YuE2 single file.</summary>
    public static bool IsComfyCheckpoint(IReadOnlyDictionary<string, SafeTensorDescriptor> descriptors)
    {
        ArgumentNullException.ThrowIfNull(descriptors);
        return descriptors.ContainsKey(TokenizerKey)
            && descriptors.ContainsKey(NarPrefix + "vae2llm.weight")
            && descriptors.ContainsKey(NarPrefix + "llm2vae.weight")
            && descriptors.ContainsKey(NarPrefix + "latent_pos_embed.pe")
            && descriptors.ContainsKey(NarPrefix + "time_embedder.mlp.0.weight");
    }

    /// <summary>The three weight sets, in the naming <see cref="GenericTransformer"/> and the Oobleck codec expect.
    /// An unrecognised key throws: a silently dropped weight is a wrong song, not a warning.</summary>
    public static Yue2Weights Convert(IReadOnlyDictionary<string, Tensor> raw, int layers = 28)
    {
        ArgumentNullException.ThrowIfNull(raw);
        Dictionary<string, Tensor> ar = new(StringComparer.Ordinal);
        Dictionary<string, Tensor> nar = new(StringComparer.Ordinal);
        Dictionary<string, Tensor> vae = new(StringComparer.Ordinal);
        byte[]? tokenizerJson = null;

        foreach ((string key, Tensor tensor) in raw)
        {
            if (key == TokenizerKey) { tokenizerJson = ReadBytes(tensor); continue; }

            if (key.StartsWith(VaePrefix, StringComparison.Ordinal))
            {
                // The Oobleck codec host-reads its weight-norm pairs and Snake alpha/beta as F32 spans.
                vae[key[VaePrefix.Length..]] = CastToF32IfNeeded(tensor);
                continue;
            }
            if (key.StartsWith(ArPrefix, StringComparison.Ordinal)) { MapBody(key[ArPrefix.Length..], tensor, ar, isAr: true); continue; }
            if (key.StartsWith(NarPrefix, StringComparison.Ordinal)) { MapNar(key[NarPrefix.Length..], tensor, nar); continue; }

            throw new InvalidOperationException($"YuE2 checkpoint carries an unrecognised key '{key}'.");
        }

        if (tokenizerJson is null)
            throw new InvalidOperationException($"YuE2 checkpoint has no '{TokenizerKey}' tensor; it is the only tokenizer the model ships.");

        // Comfy duplicates the shared final norm; the NAR stack reads it through llm2vae, so both need it.
        if (!nar.ContainsKey("model.norm.weight") && ar.TryGetValue("model.norm.weight", out Tensor? shared))
            nar["model.norm.weight"] = shared;

        RequireComplete(ar, layers, "autoregressive", lmHead: true);
        RequireComplete(nar, layers, "acoustic", lmHead: false);
        foreach (string required in (string[])["vae2llm.weight", "vae2llm.bias", "llm2vae.weight", "llm2vae.bias",
                                               "latent_pos_embed.pe", "time_embedder.mlp.0.weight", "time_embedder.mlp.2.weight"])
        {
            if (!nar.ContainsKey(required))
                throw new InvalidOperationException($"YuE2 acoustic stack is missing '{required}'.");
        }

        return new Yue2Weights(ar, nar, vae, tokenizerJson);
    }

    private static void MapNar(string sub, Tensor tensor, Dictionary<string, Tensor> nar)
    {
        switch (sub)
        {
            // Host-read by the projection and the sinusoidal timestep embedding.
            case "vae2llm.weight" or "vae2llm.bias" or "llm2vae.weight" or "llm2vae.bias"
                 or "time_embedder.mlp.0.weight" or "time_embedder.mlp.0.bias"
                 or "time_embedder.mlp.2.weight" or "time_embedder.mlp.2.bias":
                nar[sub] = CastToF32IfNeeded(tensor);
                return;
            case "latent_pos_embed.pe":
                nar[sub] = CastToF32IfNeeded(tensor);
                return;
        }
        MapBody(sub, tensor, nar, isAr: false);
    }

    private static void MapBody(string sub, Tensor tensor, Dictionary<string, Tensor> output, bool isAr)
    {
        if (sub is "model.embed_tokens.weight" or "model.lm_head.weight" or "model.norm.weight")
        {
            // GenericTransformer takes lm_head unprefixed; embed/norm keep their model-relative names.
            output[sub == "model.lm_head.weight" ? "lm_head.weight" : sub] = tensor;
            return;
        }
        if (!sub.StartsWith("model.layers.", StringComparison.Ordinal))
            throw new InvalidOperationException($"YuE2 {(isAr ? "AR" : "acoustic")} stack carries an unrecognised key '{sub}'.");

        int dot = sub.IndexOf('.', "model.layers.".Length);
        string layer = sub[..dot];
        string leaf = sub[(dot + 1)..];
        switch (leaf)
        {
            case "self_attn.qkv_proj.weight":
            {
                // Row blocks are Q ‖ K ‖ V, sized (heads·head_dim, kv_heads·head_dim, kv_heads·head_dim).
                long rows = tensor.Shape[0];
                long kv = rows / 4;             // 2048 q + 1024 k + 1024 v for the released 16/8-head geometry
                output[$"{layer}.self_attn.q_proj.weight"] = RowSlice(tensor, 0, rows - 2 * kv);
                output[$"{layer}.self_attn.k_proj.weight"] = RowSlice(tensor, rows - 2 * kv, kv);
                output[$"{layer}.self_attn.v_proj.weight"] = RowSlice(tensor, rows - kv, kv);
                return;
            }
            case "mlp.gate_up_proj.weight":
            {
                long half = tensor.Shape[0] / 2;
                output[$"{layer}.mlp.gate_proj.weight"] = RowSlice(tensor, 0, half);
                output[$"{layer}.mlp.up_proj.weight"] = RowSlice(tensor, half, half);
                return;
            }
            case "self_attn.o_proj.weight" or "mlp.down_proj.weight"
                 or "input_layernorm.weight" or "post_attention_layernorm.weight":
                output[$"{layer}.{leaf}"] = tensor;
                return;
            case "self_attn.q_norm.weight" or "self_attn.k_norm.weight":
                output[$"{layer}.{leaf}"] = CastToF32IfNeeded(tensor);
                return;
            default:
                throw new InvalidOperationException($"YuE2 {(isAr ? "AR" : "acoustic")} layer carries an unrecognised leaf '{leaf}'.");
        }
    }

    private static void RequireComplete(Dictionary<string, Tensor> weights, int layers, string what, bool lmHead)
    {
        for (int i = 0; i < layers; i++)
        {
            foreach (string leaf in (string[])["self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
                                               "self_attn.o_proj.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight",
                                               "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
                                               "input_layernorm.weight", "post_attention_layernorm.weight"])
            {
                if (!weights.ContainsKey($"model.layers.{i}.{leaf}"))
                    throw new InvalidOperationException($"YuE2 {what} stack is missing 'model.layers.{i}.{leaf}'.");
            }
        }
        if (!weights.ContainsKey("model.norm.weight"))
            throw new InvalidOperationException($"YuE2 {what} stack is missing 'model.norm.weight'.");
        if (lmHead && (!weights.ContainsKey("lm_head.weight") || !weights.ContainsKey("model.embed_tokens.weight")))
            throw new InvalidOperationException($"YuE2 {what} stack is missing its embedding or output head.");
    }

    private static unsafe byte[] ReadBytes(Tensor tensor)
    {
        byte[] bytes = new byte[tensor.ElementCount * tensor.DType.SizeInBytes];
        fixed (byte* dst = bytes)
            Buffer.MemoryCopy((void*)tensor.DataPointer, dst, bytes.Length, bytes.Length);
        return bytes;
    }

    /// <summary>Copies rows <c>[startRow, startRow+numRows)</c> into a new owned tensor of the same dtype.</summary>
    private static unsafe Tensor RowSlice(Tensor src, long startRow, long numRows)
    {
        long cols = src.Shape.Rank == 1 ? 1 : src.ElementCount / src.Shape[0];
        long rowBytes = cols * src.DType.SizeInBytes;
        Tensor dst = new(new TensorShape(numRows, cols), src.DType);
        byte* sp = (byte*)src.DataPointer + startRow * rowBytes;
        Buffer.MemoryCopy(sp, (void*)dst.DataPointer, numRows * rowBytes, numRows * rowBytes);
        return dst;
    }

    private static Tensor CastToF32IfNeeded(Tensor tensor)
        => tensor.DType == DType.F32 ? tensor : tensor.CastTo(DType.F32);
}

/// <summary>The three weight sets a YuE2 checkpoint splits into, plus the tokenizer document it embeds.</summary>
public sealed record Yue2Weights(
    Dictionary<string, Tensor> Ar,
    Dictionary<string, Tensor> Nar,
    Dictionary<string, Tensor> Vae,
    byte[] TokenizerJson);
