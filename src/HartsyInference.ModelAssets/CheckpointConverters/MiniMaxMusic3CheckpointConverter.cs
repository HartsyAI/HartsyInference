using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.ModelAssets.CheckpointConverters;

/// <summary>Converts the Comfy-Org MiniMax-Music-3 single-file DiT checkpoint
/// (<c>diffusion_models/minimax_music3_dit_*.safetensors</c>: <c>diffusion_transformer.*</c> with fused QKV,
/// plus the four condition-encoder tensors under Comfy naming) into the engine's diffusers-derived naming
/// (<c>transformer_blocks.{i}.attn.to_q/…</c> and the <c>condition_encoder</c> keys). The Comfy file carries
/// ONLY the flow DiT + condition encoder — the language model, depth decoder, vocoder, and tokenizer are not
/// in it and load from the official repo.</summary>
public static class MiniMaxMusic3CheckpointConverter
{
    /// <summary>True when the header is a Comfy-Org MiniMax-Music-3 DiT single file.</summary>
    public static bool IsComfyDit(IReadOnlyDictionary<string, SafeTensorDescriptor> descriptors) =>
        descriptors.ContainsKey("diffusion_transformer.transformer.layers.0.self_attn.to_qkv.weight")
        && descriptors.ContainsKey("cond_layer_logits");

    /// <summary>Maps the Comfy DiT file into the engine's transformer + condition-encoder dictionaries. Fused
    /// QKV rows split q-first (concat order proven by tensor diff against the official per-projection
    /// weights). Unmapped keys throw — a silently dropped weight is a wrong generation, not a warning.</summary>
    public static (Dictionary<string, Tensor> Transformer, Dictionary<string, Tensor> ConditionEncoder)
        ConvertComfyDit(IReadOnlyDictionary<string, Tensor> raw)
    {
        Dictionary<string, Tensor> transformer = new Dictionary<string, Tensor>(StringComparer.Ordinal);
        Dictionary<string, Tensor> conditionEncoder = new Dictionary<string, Tensor>(StringComparer.Ordinal);
        foreach ((string key, Tensor tensor) in raw)
        {
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
        return (transformer, conditionEncoder);
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
