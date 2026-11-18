using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelAssets.CheckpointConverters;

/// <summary>Converts the Comfy-Org MiniMax-Music-3 single-file DiT checkpoint (<c>diffusion_models/minimax_music3_dit_*.safetensors</c>: <c>diffusion_transformer.*</c> with fused QKV, plus the four condition-encoder tensors under Comfy naming) into the engine's diffusers-derived naming (<c>transformer_blocks.{i}.attn.to_q/…</c> and the <c>condition_encoder</c> keys). The Comfy file carries ONLY the flow DiT + condition encoder — the language model, depth decoder, vocoder, and tokenizer are not in it and load from the official repo.</summary>
public static class MiniMaxMusic3CheckpointConverter
{
    private static readonly string[] ConditionKeys = ["layer_weight_logits", "layer_scale", "proj.weight", "proj.bias"];

    private static readonly string[] TopLevelDitKeys =
    [
        "preprocess_conv.weight", "postprocess_conv.weight", "time_proj.weight", "time_embed.linear_1.weight",
        "time_embed.linear_1.bias", "time_embed.linear_2.weight", "time_embed.linear_2.bias", "proj_in.weight", "proj_out.weight",
    ];

    private static readonly string[] BlockKeys =
    [
        "norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias", "attn.to_q.weight", "attn.to_k.weight", "attn.to_v.weight",
        "attn.to_out.0.weight", "ff_in.weight", "ff_in.bias", "ff_out.weight", "ff_out.bias",
    ];

    /// <summary>Maps a diffusers-named flow transformer — the official <c>transformer/</c> weights, the Hartsy repack, a fp16/bf16 copy or a GGUF build that kept the names — into the engine's dictionaries. The condition encoder comes back only when the file carries it. Unknown keys throw, as in <see cref="ConvertComfyDit"/>.</summary>
    /// <remarks>Host-read tensors widen to F32 (an F16 or BF16 view would silently misread as floats). A GGUF kernel-1 convolution arrives in ggml order <c>[1, in, out]</c> and is relabelled <c>[out, in, 1]</c>.</remarks>
    public static (Dictionary<string, Tensor> Transformer, Dictionary<string, Tensor>? ConditionEncoder)
        ConvertDiffusersDit(IReadOnlyDictionary<string, Tensor> raw)
    {
        Dictionary<string, Tensor> transformer = new Dictionary<string, Tensor>(StringComparer.Ordinal);
        Dictionary<string, Tensor> conditionEncoder = new Dictionary<string, Tensor>(StringComparer.Ordinal);
        foreach ((string key, Tensor source) in raw)
        {
            Tensor tensor = Dequantized(source);
            if (Array.IndexOf(ConditionKeys, key) >= 0)
            {
                conditionEncoder[key] = CastToF32IfNeeded(tensor);
            }
            else if (key == "time_proj.weight")
            {
                transformer[key] = CastToF32IfNeeded(tensor);
            }
            else if (key is "preprocess_conv.weight" or "postprocess_conv.weight")
            {
                transformer[key] = KernelOneAsOutInOne(tensor);
            }
            else if (IsDiffusersDitKey(key))
            {
                transformer[key] = tensor;
            }
            else if (!key.EndsWith(".inv_freq", StringComparison.Ordinal))
            {
                throw new NotSupportedException($"Unrecognized MiniMax-Music-3 DiT key '{key}' — refusing a partial conversion.");
            }
        }
        WidenVectorsToF32(transformer);
        return (transformer, conditionEncoder.Count == 0 ? null : conditionEncoder);
    }

    private static bool IsDiffusersDitKey(string key)
    {
        if (Array.IndexOf(TopLevelDitKeys, key) >= 0)
        {
            return true;
        }
        const string blocks = "transformer_blocks.";
        if (!key.StartsWith(blocks, StringComparison.Ordinal))
        {
            return false;
        }
        int dot = key.IndexOf('.', blocks.Length);
        return dot > blocks.Length && int.TryParse(key.AsSpan(blocks.Length, dot - blocks.Length), out _)
            && Array.IndexOf(BlockKeys, key[(dot + 1)..]) >= 0;
    }

    /// <summary>A kernel-1 convolution stored in ggml order <c>[1, in, out]</c> viewed as <c>[out, in, 1]</c>; any other rank-3 shape is returned as is.</summary>
    private static Tensor KernelOneAsOutInOne(Tensor kernel) =>
        kernel.Shape.Rank == 3 && kernel.Shape[0] == 1 && kernel.Shape[2] != 1
            ? kernel.Reshape(new TensorShape((int)kernel.Shape[2], (int)kernel.Shape[1], 1)) : kernel;

    /// <summary>An int8 (ConvRot) weight widened to BF16 so the splits and the transformer see a dense matrix; any other tensor is returned as is.</summary>
    private static Tensor Dequantized(Tensor tensor) =>
        tensor.DType == DType.I8 && tensor.QuantInfo?.RowScale is { } rowScale
            ? Int8ConvRotCodec.DequantToBf16(tensor, rowScale, tensor.QuantInfo.ConvRotGroupSize) : tensor;

    /// <summary>Maps the Comfy DiT file into the engine's transformer + condition-encoder dictionaries. Fused QKV rows split q-first (concat order proven by tensor diff against the official per-projection weights). Unmapped keys throw — a silently dropped weight is a wrong generation, not a warning.</summary>
    public static (Dictionary<string, Tensor> Transformer, Dictionary<string, Tensor> ConditionEncoder)
        ConvertComfyDit(IReadOnlyDictionary<string, Tensor> raw)
    {
        Dictionary<string, Tensor> transformer = new Dictionary<string, Tensor>(StringComparer.Ordinal);
        Dictionary<string, Tensor> conditionEncoder = new Dictionary<string, Tensor>(StringComparer.Ordinal);
        foreach ((string key, Tensor source) in raw)
        {
            Tensor tensor = Dequantized(source);
            switch (key)
            {
                // F32 casts: the condition encoder host-reads the logit/scale values via AsReadOnlySpan<float>
                // (an F16 view would silently halve the span), and the official subfolder ships these F32.
                case "cond_layer_logits":
                    conditionEncoder["layer_weight_logits"] = CastToF32IfNeeded(tensor);
                    continue;
                case "cond_layer_scale":
                    conditionEncoder["layer_scale"] = CastToF32IfNeeded(tensor);
                    continue;
                case "latent_conditioners.0.weight":
                    conditionEncoder["proj.weight"] = CastToF32IfNeeded(tensor);
                    continue;
                case "latent_conditioners.0.bias":
                    conditionEncoder["proj.bias"] = CastToF32IfNeeded(tensor);
                    continue;
                case "diffusion_transformer.preprocess_conv.weight":
                    transformer["preprocess_conv.weight"] = tensor;
                    continue;
                case "diffusion_transformer.postprocess_conv.weight":
                    transformer["postprocess_conv.weight"] = tensor;
                    continue;
                case "diffusion_transformer.timestep_features.weight":
                    // Host-read by the Fourier embedding (same AsReadOnlySpan<float> constraint as the
                    // condition encoder's logits) — the official subfolder ships it F32.
                    transformer["time_proj.weight"] = CastToF32IfNeeded(tensor);
                    continue;
                case "diffusion_transformer.to_timestep_embed.0.weight":
                    transformer["time_embed.linear_1.weight"] = tensor;
                    continue;
                case "diffusion_transformer.to_timestep_embed.0.bias":
                    transformer["time_embed.linear_1.bias"] = tensor;
                    continue;
                case "diffusion_transformer.to_timestep_embed.2.weight":
                    transformer["time_embed.linear_2.weight"] = tensor;
                    continue;
                case "diffusion_transformer.to_timestep_embed.2.bias":
                    transformer["time_embed.linear_2.bias"] = tensor;
                    continue;
                case "diffusion_transformer.transformer.project_in.weight":
                    transformer["proj_in.weight"] = tensor;
                    continue;
                case "diffusion_transformer.transformer.project_out.weight":
                    transformer["proj_out.weight"] = tensor;
                    continue;
                case "diffusion_transformer.transformer.rotary_pos_emb.inv_freq":
                    continue;
            }

            const string layerRoot = "diffusion_transformer.transformer.layers.";
            if (!key.StartsWith(layerRoot, StringComparison.Ordinal))
            {
                throw new NotSupportedException($"Unrecognized MiniMax-Music-3 Comfy DiT key '{key}' — refusing a partial conversion.");
            }
            string rest = key[layerRoot.Length..];
            int dot = rest.IndexOf('.', StringComparison.Ordinal);
            string layer = rest[..dot];
            string sub = rest[(dot + 1)..];
            string prefix = $"transformer_blocks.{layer}";
            switch (sub)
            {
                case "pre_norm.gamma": transformer[$"{prefix}.norm1.weight"] = tensor; break;
                case "pre_norm.beta": transformer[$"{prefix}.norm1.bias"] = tensor; break;
                case "ff_norm.gamma": transformer[$"{prefix}.norm2.weight"] = tensor; break;
                case "ff_norm.beta": transformer[$"{prefix}.norm2.bias"] = tensor; break;
                case "self_attn.to_out.weight": transformer[$"{prefix}.attn.to_out.0.weight"] = tensor; break;
                case "ff.ff.0.proj.weight": transformer[$"{prefix}.ff_in.weight"] = tensor; break;
                case "ff.ff.0.proj.bias": transformer[$"{prefix}.ff_in.bias"] = tensor; break;
                case "ff.ff.2.weight": transformer[$"{prefix}.ff_out.weight"] = tensor; break;
                case "ff.ff.2.bias": transformer[$"{prefix}.ff_out.bias"] = tensor; break;
                case "self_attn.to_qkv.weight":
                    SplitQkvRows(tensor, prefix, transformer);
                    break;
                default:
                    throw new NotSupportedException($"Unrecognized MiniMax-Music-3 Comfy DiT layer key '{key}' — refusing a partial conversion.");
            }
        }
        WidenVectorsToF32(transformer);
        return (transformer, conditionEncoder);
    }

    /// <summary>Norm scales and biases back to F32 — the precision the official weights ship them in. An F16 vector (fp16 and GGUF builds store them so) corrupts the DiT once its weights are resident on the device.</summary>
    private static void WidenVectorsToF32(Dictionary<string, Tensor> weights)
    {
        foreach ((string key, Tensor tensor) in weights.ToList())
        {
            if (tensor.Shape.Rank == 1)
            {
                weights[key] = CastToF32IfNeeded(tensor);
            }
        }
    }

    private static Tensor CastToF32IfNeeded(Tensor t) => t.DType == DType.F32 ? t : t.CastTo(DType.F32);

    /// <summary>Splits fused QKV rows <c>[3·hidden, inDim]</c> into to_q / to_k / to_v (q-first concat).</summary>
    private static unsafe void SplitQkvRows(Tensor fused, string prefix, Dictionary<string, Tensor> output)
    {
        int hidden = (int)fused.Shape[0] / 3;
        int inDim = (int)fused.Shape[1];
        long chunkBytes = (long)hidden * fused.DType.ComputeByteCount(inDim);
        TensorShape splitShape = new TensorShape(hidden, inDim);
        Tensor q = new Tensor(splitShape, fused.DType);
        Tensor k = new Tensor(splitShape, fused.DType);
        Tensor v = new Tensor(splitShape, fused.DType);
        byte* src = (byte*)fused.DataPointer;
        Buffer.MemoryCopy(src, (void*)q.DataPointer, chunkBytes, chunkBytes);
        Buffer.MemoryCopy(src + chunkBytes, (void*)k.DataPointer, chunkBytes, chunkBytes);
        Buffer.MemoryCopy(src + 2 * chunkBytes, (void*)v.DataPointer, chunkBytes, chunkBytes);
        output[$"{prefix}.attn.to_q.weight"] = q;
        output[$"{prefix}.attn.to_k.weight"] = k;
        output[$"{prefix}.attn.to_v.weight"] = v;
    }
}
