using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelHandler.CheckpointConverters.Utils;

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
        // Propagate fp8_scaled per-tensor scale — the raw fp8 bytes are real_value/scale; splits keep the fused tensor's factor.
        qWeight.Fp8ScaleFactor = inProj.Fp8ScaleFactor;
        kWeight.Fp8ScaleFactor = inProj.Fp8ScaleFactor;
        vWeight.Fp8ScaleFactor = inProj.Fp8ScaleFactor;

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
        // Propagate fp8_scaled per-tensor scale — biases aren't fp8-scaled in practice, but a non-1 factor must follow the bytes.
        qBias.Fp8ScaleFactor = inProj.Fp8ScaleFactor;
        kBias.Fp8ScaleFactor = inProj.Fp8ScaleFactor;
        vBias.Fp8ScaleFactor = inProj.Fp8ScaleFactor;

        byte* src = (byte*)inProj.DataPointer;

        Buffer.MemoryCopy(src, (void*)qBias.DataPointer, chunkBytes, chunkBytes);
        Buffer.MemoryCopy(src + chunkBytes, (void*)kBias.DataPointer, chunkBytes, chunkBytes);
        Buffer.MemoryCopy(src + 2 * chunkBytes, (void*)vBias.DataPointer, chunkBytes, chunkBytes);

        output[$"{layerPrefix}.self_attn.q_proj.bias"] = qBias;
        output[$"{layerPrefix}.self_attn.k_proj.bias"] = kBias;
        output[$"{layerPrefix}.self_attn.v_proj.bias"] = vBias;
    }


    // ── FP8 Scaled ──────────────────────────────────────────

    /// <summary>Folds per-tensor FP8 scale companions into <see cref="Tensor.Fp8ScaleFactor"/> on the matching weight tensors and drops the companions. Supports three companion formats:
    /// <list type="bullet">
    ///   <item>ComfyUI <c>fp8_scaled</c>: <c>.scale_weight</c>/<c>.scale_input</c> (F32 scalar).</item>
    ///   <item>BFL Mistral / Flux.2 Dev mixed-fp8: <c>.weight_scale</c>/<c>.input_scale</c> (F32 scalar).</item>
    ///   <item>ComfyUI <c>comfy_quant</c>: <c>.comfy_quant</c> (U8 JSON blob like <c>{"format":"float8_e4m3fn"}</c>) — newer format used by Chroma1-HD-fp8mixed and similar. There's no separately-stored scalar; fp8 values are used at identity scale (the model is trained with the natural fp8 dynamic range). The companion is purely a format declaration and must still be dropped or it pollutes the weight dictionary.</item>
    /// </list>
    /// The input-side scale is dropped — we run F32 activations and use alpha=weight_scale at GEMM time. Marker tensors like <c>scaled_fp8</c> are also dropped.</summary>
    /// <param name="source">Raw checkpoint dictionary (mutated; companion keys removed).</param>
    /// <returns>A new dictionary without companion keys, with <c>Fp8ScaleFactor</c> populated on FP8 weights.</returns>
    public static unsafe Dictionary<string, Tensor> ApplyFp8ScaledDequant(Dictionary<string, Tensor> source)
    {
        // First pass: gather scale companions keyed by the base name (the part before the suffix).
        // `.weight_scale_2` is nvfp4's global scalar (block scales live in `.weight_scale`); note that
        // ".weight_scale_2".EndsWith(".weight_scale") is FALSE, so the two never collide in the buckets.
        Dictionary<string, Tensor> weightScales = new();
        Dictionary<string, Tensor> weightScale2s = new();
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
            else if (key.EndsWith(".weight_scale_2", StringComparison.Ordinal))
            {
                weightScale2s[key[..^".weight_scale_2".Length]] = kvp.Value;
                sawAnyScale = true;
            }
            else if (key.EndsWith(".scale_input", StringComparison.Ordinal) ||
                     key.EndsWith(".input_scale", StringComparison.Ordinal) ||
                     key.EndsWith(".comfy_quant", StringComparison.Ordinal) ||
                     key == "scaled_fp8")
            {
                sawAnyScale = true; // these are dropped but flag the format as scaled (or as fp8-declared in comfy_quant's case)
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
                key.EndsWith(".weight_scale_2", StringComparison.Ordinal) ||
                key.EndsWith(".input_scale", StringComparison.Ordinal) ||
                key.EndsWith(".comfy_quant", StringComparison.Ordinal) ||
                key == "scaled_fp8")
            {
                continue;
            }

            // For FP8 weight tensors with a matching scalar scale, fold into Fp8ScaleFactor so
            // CudaBackend can apply it via cuBLAS alpha at GEMM time. Weights without a scale
            // companion (e.g. comfy_quant format) keep the default Fp8ScaleFactor of 1.0 — the
            // model was trained with fp8's natural dynamic range and no per-tensor rescaling.
            if (kvp.Value.DType.IsFp8 && key.EndsWith(".weight", StringComparison.Ordinal))
            {
                string baseKey = key[..^".weight".Length];
                if (weightScales.TryGetValue(baseKey, out Tensor? scaleT) && scaleT.DType == DType.F32)
                {
                    float scale = ((float*)scaleT.DataPointer)[0];
                    kvp.Value.Fp8ScaleFactor = scale;
                }
            }

            // NVFP4 (ComfyUI comfy_quant "nvfp4"): U8 weight packing two e2m1 values per byte, with per-16-element
            // F8-E4M3 block scales in `.weight_scale` and a global F32 scalar in `.weight_scale_2`. Identified
            // structurally (that companion combination exists for no other format) and dequantized to F16 host-side
            // at load — there is no per-block-scale GEMM path in the backends.
            if (kvp.Value.DType == DType.U8 && key.EndsWith(".weight", StringComparison.Ordinal))
            {
                string baseKey = key[..^".weight".Length];
                if (weightScales.TryGetValue(baseKey, out Tensor? blockScales) && blockScales.DType == DType.F8E4M3
                    && blockScales.Shape.Rank == 2
                    && weightScale2s.TryGetValue(baseKey, out Tensor? scale2T) && scale2T.DType == DType.F32)
                {
                    float globalScale = ((float*)scale2T.DataPointer)[0];
                    result[key] = DequantNvfp4ToF16(kvp.Value, blockScales, globalScale);
                    continue;
                }
            }

            result[key] = kvp.Value;
        }
        return result;
    }

    /// <summary>The 8 magnitudes representable by FP4 E2M1, indexed by bits [e1 e0 m]: exp==0 → {0, 0.5}
    /// (subnormal), exp>0 → 2^(exp-1) · (1 + m/2). Bit 3 of the nibble is the sign.</summary>
    private static readonly float[] E2M1Magnitudes = [0f, 0.5f, 1f, 1.5f, 2f, 3f, 4f, 6f];

    /// <summary>256-entry FP8-E4M3FN decode table (built once). Index = raw byte. E4M3FN: bias 7, no infinities,
    /// exp=15/man=7 is NaN, max ±448.</summary>
    private static readonly float[] E4M3Table = BuildE4M3Table();

    private static float[] BuildE4M3Table()
    {
        float[] table = new float[256];
        for (int b = 0; b < 256; b++)
        {
            int sign = (b >> 7) & 1;
            int exp = (b >> 3) & 0xF;
            int man = b & 7;
            float value;
            if (exp == 15 && man == 7)
                value = float.NaN;
            else if (exp == 0)
                value = man / 8f * MathF.Pow(2f, -6f);          // subnormal
            else
                value = MathF.Pow(2f, exp - 7) * (1f + man / 8f); // normal
            table[b] = sign == 1 ? -value : value;
        }
        return table;
    }

    /// <summary>Dequantizes an NVFP4 weight to F16. <paramref name="packed"/> is U8 <c>[rows, cols/2]</c> with two
    /// e2m1 values per byte — element <c>2j</c> in the HIGH nibble, <c>2j+1</c> in the LOW nibble (comfy_kitchen
    /// <c>hi_first=True</c>, its default for both quantize and dequantize; verified against
    /// <c>float_utils.unpack_uint4</c>). <paramref name="blockScales"/> is F8-E4M3 <c>[rows, cols/16]</c> — one
    /// scale per 16 consecutive input-dim elements. The full value is <c>e2m1 · block_scale · globalScale</c>.
    /// Rows are dequantized in parallel (pure per-row writes).</summary>
    public static Tensor DequantNvfp4ToF16(Tensor packed, Tensor blockScales, float globalScale)
    {
        if (packed.DType != DType.U8 || packed.Shape.Rank != 2)
            throw new ArgumentException($"NVFP4 packed weight must be U8 rank-2; got {packed.DType} {packed.Shape}.");
        long rows = packed.Shape[0];
        long packedCols = packed.Shape[1];
        long cols = packedCols * 2;
        long scaleCols = blockScales.Shape[1];
        if (blockScales.Shape[0] != rows || scaleCols * 16 != cols)
            throw new ArgumentException(
                $"NVFP4 block-scale shape [{blockScales.Shape[0]}, {scaleCols}] does not match packed weight [{rows}, {packedCols}] (expect [rows, cols/16]).");

        Tensor result = new Tensor(new TensorShape(rows, cols), DType.F16);
        unsafe
        {
            byte* src = (byte*)packed.DataPointer;
            byte* scales = (byte*)blockScales.DataPointer;
            Half* dst = (Half*)result.DataPointer;
            float[] e2m1 = E2M1Magnitudes;
            float[] e4m3 = E4M3Table;
            Parallel.For(0, (int)rows, r =>
            {
                byte* rowSrc = src + (long)r * packedCols;
                byte* rowScales = scales + (long)r * scaleCols;
                Half* rowDst = dst + (long)r * cols;
                for (long j = 0; j < packedCols; j++)
                {
                    // 8 packed bytes per 16-element scale block → scale index = j/8.
                    float scale = e4m3[rowScales[j >> 3]] * globalScale;
                    byte b = rowSrc[j];
                    int hi = b >> 4, lo = b & 0xF;
                    float hiVal = e2m1[hi & 7] * scale;
                    float loVal = e2m1[lo & 7] * scale;
                    rowDst[j * 2] = (Half)((hi & 8) != 0 ? -hiVal : hiVal);
                    rowDst[j * 2 + 1] = (Half)((lo & 8) != 0 ? -loVal : loVal);
                }
            });
        }
        return result;
    }
}
