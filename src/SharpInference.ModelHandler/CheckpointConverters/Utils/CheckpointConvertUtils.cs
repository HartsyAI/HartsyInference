using SharpInference.Core.Tensors;

namespace SharpInference.ModelHandler.CheckpointConverters.Utils;

/// <summary>Shared utilities for converting LDM/CompVis checkpoint keys to diffusers format. Used by model-specific converters.</summary>
public static unsafe class CheckpointConvertUtils
{
    // ── UNet Shared ──────────────────────────────────────────

    /// <summary>Converts LDM ResNet sub-keys to diffusers format. Shared across all Stable Diffusion variants.</summary>
    /// <remarks>
    /// in_layers.0→norm1, in_layers.2→conv1, emb_layers.1→time_emb_proj,
    /// out_layers.0→norm2, out_layers.3→conv2, skip_connection→conv_shortcut.
    /// </remarks>
    public static string ConvertResNetSubKey(string ldmSubKey)
    {
        if (ldmSubKey.StartsWith("in_layers.0."))
            return "norm1." + ldmSubKey["in_layers.0.".Length..];
        if (ldmSubKey.StartsWith("in_layers.2."))
            return "conv1." + ldmSubKey["in_layers.2.".Length..];
        if (ldmSubKey.StartsWith("emb_layers.1."))
            return "time_emb_proj." + ldmSubKey["emb_layers.1.".Length..];
        if (ldmSubKey.StartsWith("out_layers.0."))
            return "norm2." + ldmSubKey["out_layers.0.".Length..];
        if (ldmSubKey.StartsWith("out_layers.3."))
            return "conv2." + ldmSubKey["out_layers.3.".Length..];
        if (ldmSubKey.StartsWith("skip_connection."))
            return "conv_shortcut." + ldmSubKey["skip_connection.".Length..];
        return ldmSubKey;
    }

    /// <summary>Converts LDM time_embed keys to diffusers time_embedding format.</summary>
    /// <remarks>time_embed.0→time_embedding.linear_1, time_embed.2→time_embedding.linear_2.</remarks>
    public static string ConvertTimeEmbedKey(string ldmKey)
    {
        string rest = ldmKey["time_embed.".Length..];
        if (rest.StartsWith("0."))
            return "time_embedding.linear_1." + rest[2..];
        if (rest.StartsWith("2."))
            return "time_embedding.linear_2." + rest[2..];
        return "time_embedding." + rest;
    }

    /// <summary>Converts LDM out keys to diffusers conv_norm_out/conv_out format.</summary>
    /// <remarks>out.0→conv_norm_out, out.2→conv_out.</remarks>
    public static string ConvertOutKey(string ldmKey)
    {
        string rest = ldmKey["out.".Length..];
        if (rest.StartsWith("0."))
            return "conv_norm_out." + rest[2..];
        if (rest.StartsWith("2."))
            return "conv_out." + rest[2..];
        return "conv_out." + rest;
    }

    /// <summary>Converts LDM middle_block keys to diffusers mid_block format.</summary>
    /// <remarks>middle_block.0→mid_block.resnets.0, middle_block.1→mid_block.attentions.0, middle_block.2→mid_block.resnets.1.</remarks>
    public static string? ConvertMiddleBlockKey(string ldmKey)
    {
        string afterPrefix = ldmKey["middle_block.".Length..];
        int firstDot = afterPrefix.IndexOf('.');
        if (firstDot < 0) return null;

        int subIdx = int.Parse(afterPrefix[..firstDot]);
        string rest = afterPrefix[(firstDot + 1)..];

        return subIdx switch
        {
            0 => "mid_block.resnets.0." + ConvertResNetSubKey(rest),
            1 => "mid_block.attentions.0." + rest,
            2 => "mid_block.resnets.1." + ConvertResNetSubKey(rest),
            _ => null,
        };
    }


    // ── VAE Shared ──────────────────────────────────────────

    /// <summary>Converts LDM VAE keys (after stripping first_stage_model. prefix) to diffusers format. Shared across all Stable Diffusion variants.</summary>
    /// <param name="ldmKey">Key after stripping "first_stage_model." prefix.</param>
    /// <param name="numUpLevels">Number of up block levels (4 for SD1.5/SDXL).</param>
    public static string? ConvertVaeKey(string ldmKey, int numUpLevels = 4)
    {
        // post_quant_conv / quant_conv — keep as-is
        if (ldmKey.StartsWith("post_quant_conv.") || ldmKey.StartsWith("quant_conv."))
            return ldmKey;

        if (ldmKey.StartsWith("encoder."))
            return ConvertVaeEncoderKey(ldmKey["encoder.".Length..], numUpLevels);

        if (ldmKey.StartsWith("decoder."))
            return ConvertVaeDecoderKey(ldmKey["decoder.".Length..], numUpLevels);

        return null;
    }

    private static string? ConvertVaeDecoderKey(string decoderKey, int numUpLevels)
    {
        if (decoderKey.StartsWith("conv_in."))
            return "decoder.conv_in." + decoderKey["conv_in.".Length..];
        if (decoderKey.StartsWith("conv_out."))
            return "decoder.conv_out." + decoderKey["conv_out.".Length..];
        if (decoderKey.StartsWith("norm_out."))
            return "decoder.conv_norm_out." + decoderKey["norm_out.".Length..];

        if (decoderKey.StartsWith("mid."))
            return ConvertVaeMidKey("decoder", decoderKey["mid.".Length..]);

        if (decoderKey.StartsWith("up."))
            return ConvertVaeUpKey(decoderKey["up.".Length..], numUpLevels);

        return "decoder." + decoderKey;
    }

    private static string? ConvertVaeEncoderKey(string encoderKey, int numDownLevels)
    {
        if (encoderKey.StartsWith("conv_in."))
            return "encoder.conv_in." + encoderKey["conv_in.".Length..];
        if (encoderKey.StartsWith("conv_out."))
            return "encoder.conv_out." + encoderKey["conv_out.".Length..];
        if (encoderKey.StartsWith("norm_out."))
            return "encoder.conv_norm_out." + encoderKey["norm_out.".Length..];

        if (encoderKey.StartsWith("mid."))
            return ConvertVaeMidKey("encoder", encoderKey["mid.".Length..]);

        if (encoderKey.StartsWith("down."))
            return ConvertVaeDownKey(encoderKey["down.".Length..], numDownLevels);

        return "encoder." + encoderKey;
    }

    // Mid block layout is identical between encoder and decoder.
    private static string? ConvertVaeMidKey(string section, string midKey)
    {
        if (midKey.StartsWith("block_1."))
            return $"{section}.mid_block.resnets.0." + midKey["block_1.".Length..];
        if (midKey.StartsWith("block_2."))
            return $"{section}.mid_block.resnets.1." + midKey["block_2.".Length..];
        if (midKey.StartsWith("attn_1."))
            return ConvertVaeAttentionKey($"{section}.mid_block.attentions.0", midKey["attn_1.".Length..]);
        return null;
    }

    private static string? ConvertVaeUpKey(string upKey, int numUpLevels)
    {
        int firstDot = upKey.IndexOf('.');
        if (firstDot < 0) return null;

        int ldmLevel = int.Parse(upKey[..firstDot]);
        // Diffusers indexes up_blocks deepest-first; LDM indexes shallowest-first. Reverse.
        int diffusersLevel = (numUpLevels - 1) - ldmLevel;
        string rest = upKey[(firstDot + 1)..];

        if (rest.StartsWith("block."))
        {
            string blockRest = rest["block.".Length..];
            int nextDot = blockRest.IndexOf('.');
            if (nextDot < 0) return null;
            string resIdxStr = blockRest[..nextDot];
            string param = blockRest[(nextDot + 1)..];

            if (param.StartsWith("nin_shortcut."))
                param = "conv_shortcut." + param["nin_shortcut.".Length..];

            return $"decoder.up_blocks.{diffusersLevel}.resnets.{resIdxStr}.{param}";
        }

        if (rest.StartsWith("upsample.conv."))
            return $"decoder.up_blocks.{diffusersLevel}.upsamplers.0.conv." + rest["upsample.conv.".Length..];

        return null;
    }

    private static string? ConvertVaeDownKey(string downKey, int numDownLevels)
    {
        int firstDot = downKey.IndexOf('.');
        if (firstDot < 0) return null;

        // Encoder down levels run shallow → deep in both LDM and diffusers, so no reversal.
        int level = int.Parse(downKey[..firstDot]);
        if (level < 0 || level >= numDownLevels) return null;
        string rest = downKey[(firstDot + 1)..];

        if (rest.StartsWith("block."))
        {
            string blockRest = rest["block.".Length..];
            int nextDot = blockRest.IndexOf('.');
            if (nextDot < 0) return null;
            string resIdxStr = blockRest[..nextDot];
            string param = blockRest[(nextDot + 1)..];

            if (param.StartsWith("nin_shortcut."))
                param = "conv_shortcut." + param["nin_shortcut.".Length..];

            return $"encoder.down_blocks.{level}.resnets.{resIdxStr}.{param}";
        }

        if (rest.StartsWith("downsample.conv."))
            return $"encoder.down_blocks.{level}.downsamplers.0.conv." + rest["downsample.conv.".Length..];

        return null;
    }

    /// <summary>Converts LDM VAE attention keys to diffusers format (norm→group_norm, q→to_q, etc.).</summary>
    public static string ConvertVaeAttentionKey(string prefix, string attnKey)
    {
        if (attnKey.StartsWith("norm."))
            return $"{prefix}.group_norm.{attnKey["norm.".Length..]}";
        if (attnKey.StartsWith("q."))
            return $"{prefix}.to_q.{attnKey["q.".Length..]}";
        if (attnKey.StartsWith("k."))
            return $"{prefix}.to_k.{attnKey["k.".Length..]}";
        if (attnKey.StartsWith("v."))
            return $"{prefix}.to_v.{attnKey["v.".Length..]}";
        if (attnKey.StartsWith("proj_out."))
            return $"{prefix}.to_out.0.{attnKey["proj_out.".Length..]}";
        return $"{prefix}.{attnKey}";
    }


    // ── Tensor Splitting ──────────────────────────────────────────

    /// <summary>Splits a fused in_proj_weight [3*H, H] into separate q_proj, k_proj, v_proj weights [H, H] each.</summary>
    public static void SplitInProjWeight(Tensor inProj, int hiddenSize, string layerPrefix, Dictionary<string, Tensor> output)
    {
        long rowBytes = hiddenSize * inProj.DType.SizeInBytes;
        TensorShape splitShape = new TensorShape(hiddenSize, hiddenSize);

        Tensor qWeight = new Tensor(splitShape, inProj.DType);
        Tensor kWeight = new Tensor(splitShape, inProj.DType);
        Tensor vWeight = new Tensor(splitShape, inProj.DType);

        byte* src = (byte*)inProj.DataPointer;
        long chunkBytes = hiddenSize * rowBytes;

        Buffer.MemoryCopy(src, (void*)qWeight.DataPointer, chunkBytes, chunkBytes);
        Buffer.MemoryCopy(src + chunkBytes, (void*)kWeight.DataPointer, chunkBytes, chunkBytes);
        Buffer.MemoryCopy(src + 2 * chunkBytes, (void*)vWeight.DataPointer, chunkBytes, chunkBytes);

        output[$"{layerPrefix}.self_attn.q_proj.weight"] = qWeight;
        output[$"{layerPrefix}.self_attn.k_proj.weight"] = kWeight;
        output[$"{layerPrefix}.self_attn.v_proj.weight"] = vWeight;
    }

    /// <summary>Splits a fused in_proj_bias [3*H] into separate q_proj, k_proj, v_proj biases [H] each.</summary>
    public static void SplitInProjBias(Tensor inProj, int hiddenSize, string layerPrefix, Dictionary<string, Tensor> output)
    {
        long elemBytes = inProj.DType.SizeInBytes;
        long chunkBytes = hiddenSize * elemBytes;
        TensorShape splitShape = new TensorShape(hiddenSize);

        Tensor qBias = new Tensor(splitShape, inProj.DType);
        Tensor kBias = new Tensor(splitShape, inProj.DType);
        Tensor vBias = new Tensor(splitShape, inProj.DType);

        byte* src = (byte*)inProj.DataPointer;

        Buffer.MemoryCopy(src, (void*)qBias.DataPointer, chunkBytes, chunkBytes);
        Buffer.MemoryCopy(src + chunkBytes, (void*)kBias.DataPointer, chunkBytes, chunkBytes);
        Buffer.MemoryCopy(src + 2 * chunkBytes, (void*)vBias.DataPointer, chunkBytes, chunkBytes);

        output[$"{layerPrefix}.self_attn.q_proj.bias"] = qBias;
        output[$"{layerPrefix}.self_attn.k_proj.bias"] = kBias;
        output[$"{layerPrefix}.self_attn.v_proj.bias"] = vBias;
    }


    // ── FP8 Scaled ──────────────────────────────────────────

    /// <summary>Folds per-tensor FP8 scale companions into <see cref="Tensor.Fp8ScaleFactor"/> on the matching weight tensors and drops the companions. Supports ComfyUI fp8_scaled (.scale_weight/.scale_input) and BFL Mistral / Flux.2 Dev mixed-fp8 (.weight_scale/.input_scale). The input-side scale is dropped — we run F32 activations and use alpha=weight_scale at GEMM time. Marker tensors like <c>scaled_fp8</c> are also dropped.</summary>
    /// <param name="source">Raw checkpoint dictionary (mutated; companion keys removed).</param>
    /// <returns>A new dictionary without companion keys, with <c>Fp8ScaleFactor</c> populated on FP8 weights.</returns>
    public static unsafe Dictionary<string, Tensor> ApplyFp8ScaledDequant(Dictionary<string, Tensor> source)
    {
        // First pass: gather scale companions keyed by the base name (the part before the suffix).
        Dictionary<string, Tensor> weightScales = new();
        bool sawAnyScale = false;
        foreach (KeyValuePair<string, Tensor> kvp in source)
        {
            string key = kvp.Key;
            string? baseKey = null;
            if (key.EndsWith(".scale_weight", StringComparison.Ordinal))
                baseKey = key[..^".scale_weight".Length];
            else if (key.EndsWith(".weight_scale", StringComparison.Ordinal))
                baseKey = key[..^".weight_scale".Length];
            if (baseKey is not null)
            {
                weightScales[baseKey] = kvp.Value;
                sawAnyScale = true;
            }
            else if (key.EndsWith(".scale_input", StringComparison.Ordinal) ||
                     key.EndsWith(".input_scale", StringComparison.Ordinal) ||
                     key == "scaled_fp8")
            {
                sawAnyScale = true; // these are dropped but flag the format as scaled
            }
        }
        if (!sawAnyScale)
            return source;

        Dictionary<string, Tensor> result = new(source.Count);
        foreach (KeyValuePair<string, Tensor> kvp in source)
        {
            string key = kvp.Key;
            // Drop companions and format-flag tensors.
            if (key.EndsWith(".scale_weight", StringComparison.Ordinal) ||
                key.EndsWith(".scale_input", StringComparison.Ordinal) ||
                key.EndsWith(".weight_scale", StringComparison.Ordinal) ||
                key.EndsWith(".input_scale", StringComparison.Ordinal) ||
                key == "scaled_fp8")
            {
                continue;
            }

            // For FP8 weight tensors with a matching scalar scale, fold into Fp8ScaleFactor so
            // CudaBackend can apply it via cuBLAS alpha at GEMM time.
            if (kvp.Value.DType.IsFp8 && key.EndsWith(".weight", StringComparison.Ordinal))
            {
                string baseKey = key[..^".weight".Length];
                if (weightScales.TryGetValue(baseKey, out Tensor? scaleT) && scaleT.DType == DType.F32)
                {
                    float scale = ((float*)scaleT.DataPointer)[0];
                    kvp.Value.Fp8ScaleFactor = scale;
                }
            }

            result[key] = kvp.Value;
        }
        return result;
    }

}
