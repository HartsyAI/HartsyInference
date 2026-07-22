using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelAssets.CheckpointConverters.Utils;

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
    public static string? ConvertVaeKey(string ldmKey, int numUpLevels = 4, bool reverseUpIndices = true)
    {
        // post_quant_conv / quant_conv — keep as-is
        if (ldmKey.StartsWith("post_quant_conv.") || ldmKey.StartsWith("quant_conv."))
            return ldmKey;

        if (ldmKey.StartsWith("encoder."))
            return ConvertVaeEncoderKey(ldmKey["encoder.".Length..], numUpLevels);

        if (ldmKey.StartsWith("decoder."))
            return ConvertVaeDecoderKey(ldmKey["decoder.".Length..], numUpLevels, reverseUpIndices);

        return null;
    }

    private static string? ConvertVaeDecoderKey(string decoderKey, int numUpLevels, bool reverseUpIndices = true)
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
            return ConvertVaeUpKey(decoderKey["up.".Length..], numUpLevels, reverseUpIndices);

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

    private static string? ConvertVaeUpKey(string upKey, int numUpLevels, bool reverseUpIndices = true)
    {
        int firstDot = upKey.IndexOf('.');
        if (firstDot < 0) return null;

        int ldmLevel = int.Parse(upKey[..firstDot]);
        // Diffusers indexes up_blocks deepest-first; classic SD-LDM indexes shallowest-first (reverse).
        // HunyuanImage 2.1's VAE stores up.0 = deepest already — pass reverseUpIndices=false there.
        int diffusersLevel = reverseUpIndices ? (numUpLevels - 1) - ldmLevel : ldmLevel;
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
    /// <param name="nvfp4ToFp8">When true, NVFP4 weights are dequantized to <b>fp8 (1 byte/param)</b> with the block
    /// scale folded into the value and the global scale carried on <see cref="Tensor.Fp8ScaleFactor"/>, instead of
    /// the default F16 (2 byte/param). Halves the resident footprint of an all-nvfp4 model (Ideogram 4: two 9.3B
    /// DiTs, 35.9 GB at F16 → 18.6 GB at fp8, the difference between "won't fit a 24 GB card" and "fits"). Leave
    /// false for small nvfp4 text encoders (Z-Image's Qwen3-4B) where F16 is free and avoids fp8's smaller range.</param>
    /// <returns>A new dictionary without companion keys, with <c>Fp8ScaleFactor</c> populated on FP8 weights.</returns>
    public static unsafe Dictionary<string, Tensor> ApplyFp8ScaledDequant(Dictionary<string, Tensor> source, bool nvfp4ToFp8 = false)
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
                // The scalar scale companion is F32 in the original Wan/Comfy fp8_scaled files, but BF16 in some
                // repackages (Kijai WanVideo_comfy_fp8_scaled v2 — e.g. wan2.2_animate_14B). Reading a BF16 [1] scalar
                // through (float*) reinterprets two BF16 halves as one F32 = garbage, so the `== F32` guard used to
                // silently DROP the scale → the raw fp8 weight (±448) ran ~5× hot → the block activations exploded
                // and every token collapsed to the dominant direction (the Wan-Animate checkerboard). Read the scalar
                // as F32 for any real-valued scalar companion (F32 pass-through, else cast).
                if (weightScales.TryGetValue(baseKey, out Tensor? scaleT) && scaleT.Shape.ElementCount == 1
                    && (scaleT.DType == DType.F32 || scaleT.DType == DType.F16 || scaleT.DType == DType.BF16))
                {
                    if (scaleT.DType == DType.F32)
                        kvp.Value.Fp8ScaleFactor = ((float*)scaleT.DataPointer)[0];
                    else
                    {
                        using Tensor scaleF32 = scaleT.CastTo(DType.F32);
                        kvp.Value.Fp8ScaleFactor = ((float*)scaleF32.DataPointer)[0];
                    }
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
                    result[key] = nvfp4ToFp8
                        ? DequantNvfp4ToFp8(kvp.Value, blockScales, globalScale)
                        : DequantNvfp4ToF16(kvp.Value, blockScales, globalScale);
                    continue;
                }
            }

            result[key] = kvp.Value;
        }
        return result;
    }

    /// <summary>In-place: quantizes the large 2-D Linear weights under <paramref name="blockKeyMarker"/> from
    /// BF16/F16 to fp8 e4m3, ADDING a <c>.scale_weight</c> [1] F32 companion for each (the ComfyUI
    /// <c>fp8_scaled</c> layout). Because the scale rides in a companion tensor — not the un-persisted
    /// <see cref="Tensor.Fp8ScaleFactor"/> — the result round-trips through safetensors: write it once with
    /// <c>SafeTensorsWriter</c> to build a persistent fp8 <b>repack</b>, then every future load reads it back and
    /// <see cref="ApplyFp8ScaledDequant"/> folds the companions in. This halves the DiT's resident footprint
    /// (2 → 1 byte/param; the LTX-2.3 22B goes ~35 → ~18 GB, fitting a 24 GB card fully resident and killing the
    /// per-step streaming). Only weights with ≥ <paramref name="minElements"/> elements are touched (norms,
    /// scale-shift tables, timestep-MLP and projections stay BF16); the VAE/TE and already-fp8 tensors are skipped.</summary>
    public static unsafe void QuantizeDitBlocksToFp8(Dictionary<string, Tensor> weights, string blockKeyMarker,
        long minElements = 1L << 20)
    {
        List<(string Key, Tensor Scale)> companions = new();
        foreach (string key in weights.Keys.ToList())
        {
            Tensor t = weights[key];
            bool quantize = key.Contains(blockKeyMarker, StringComparison.Ordinal)
                && key.EndsWith(".weight", StringComparison.Ordinal)
                && t.Shape.Rank == 2 && t.ElementCount >= minElements
                && (t.DType == DType.BF16 || t.DType == DType.F16);
            if (!quantize) continue;
            Tensor fp8 = QuantizeToFp8Scaled(t);
            Tensor scale = new Tensor(new TensorShape(1), DType.F32);
            ((float*)scale.DataPointer)[0] = fp8.Fp8ScaleFactor;
            weights[key] = fp8;
            companions.Add((key[..^".weight".Length] + ".scale_weight", scale));
        }
        foreach ((string k, Tensor s) in companions) weights[k] = s;
    }

    /// <summary>BF16/F16 weight → fp8 e4m3 with a per-tensor scale = absmax/448 (E4M3's max magnitude), folded onto
    /// <see cref="Tensor.Fp8ScaleFactor"/> so the GEMM computes <c>fp8_decoded · scale ≈ w</c> via cuBLAS alpha.
    /// Uses the engine's tested F32→F8E4M3 cast on a transient F32 copy (the source is left untouched).</summary>
    private static unsafe Tensor QuantizeToFp8Scaled(Tensor w)
    {
        Tensor f32 = w.CastTo(DType.F32);        // BF16/F16 → fresh F32 (source unmodified)
        long n = f32.ElementCount;
        float* p = (float*)f32.DataPointer;
        float absmax = 0f;
        for (long i = 0; i < n; i++) { float a = MathF.Abs(p[i]); if (a > absmax) absmax = a; }
        float scale = absmax > 0f ? absmax / 448f : 1f;
        float inv = 1f / scale;
        for (long i = 0; i < n; i++) p[i] *= inv;
        Tensor fp8 = f32.CastTo(DType.F8E4M3);
        f32.Dispose();
        fp8.Fp8ScaleFactor = scale;
        return fp8;
    }

    /// <summary>The 8 magnitudes representable by FP4 E2M1, indexed by bits [e1 e0 m]: exp==0 → {0, 0.5}
    /// (subnormal), exp>0 → 2^(exp-1) · (1 + m/2). Bit 3 of the nibble is the sign.</summary>
    private static readonly float[] E2M1Magnitudes = [0f, 0.5f, 1f, 1.5f, 2f, 3f, 4f, 6f];

    /// <summary>256-entry FP8-E4M3FN decode table (built once). Index = raw byte. E4M3FN: bias 7, no infinities,
    /// exp=15/man=7 is NaN, max ±448. Shared with <see cref="HartsyInference.ModelAssets.Nvfp4.Nvfp4Codec"/>.</summary>
    internal static readonly float[] E4M3Table = BuildE4M3Table();

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

    /// <summary>Dequantizes an NVFP4 weight to <b>fp8 e4m3 (1 byte/param)</b> instead of F16, folding the per-16-element
    /// block scale into the stored value and returning the <b>global</b> scale on <see cref="Tensor.Fp8ScaleFactor"/>.
    /// The GEMM then computes <c>fp8_value · globalScale ≈ e2m1 · block_scale · globalScale</c> (the same real weight),
    /// reusing the existing fp8 alpha path — so an all-nvfp4 model keeps its footprint at fp8 size. fp8 e4m3 (3-bit
    /// mantissa) is finer than the 4-bit nvfp4 source, so folding block_scale into the value adds negligible rounding
    /// vs the F16 path; the only real difference is fp8's smaller range — a block whose <c>e2m1·block_scale</c> falls
    /// below fp8's min subnormal (~2^-9) flushes to 0 (F16 wouldn't). Acceptable for weights; used for the Ideogram-4
    /// DiTs to fit a 24 GB card. Nibble order and block layout match <see cref="DequantNvfp4ToF16"/>.</summary>
    public static Tensor DequantNvfp4ToFp8(Tensor packed, Tensor blockScales, float globalScale)
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

        // Build the dequantized values (e2m1 · block_scale) as F32, then cast to fp8 via the engine's TESTED
        // F32→F8E4M3 path (Tensor.CastTo) rather than a hand-rolled quantizer. The F32 intermediate is per-tensor
        // transient (freed below). Global scale rides on Fp8ScaleFactor, applied as the GEMM alpha.
        Tensor f32 = new Tensor(new TensorShape(rows, cols), DType.F32);
        unsafe
        {
            byte* src = (byte*)packed.DataPointer;
            byte* scales = (byte*)blockScales.DataPointer;
            float* dst = (float*)f32.DataPointer;
            float[] e2m1 = E2M1Magnitudes;
            float[] e4m3 = E4M3Table;
            Parallel.For(0, (int)rows, r =>
            {
                byte* rowSrc = src + (long)r * packedCols;
                byte* rowScales = scales + (long)r * scaleCols;
                float* rowDst = dst + (long)r * cols;
                for (long j = 0; j < packedCols; j++)
                {
                    // Fold ONLY the block scale into the value (not the global scale — that rides on Fp8ScaleFactor).
                    float scale = e4m3[rowScales[j >> 3]];
                    byte b = rowSrc[j];
                    int hi = b >> 4, lo = b & 0xF;
                    float hiVal = e2m1[hi & 7] * scale;
                    float loVal = e2m1[lo & 7] * scale;
                    rowDst[j * 2] = (hi & 8) != 0 ? -hiVal : hiVal;
                    rowDst[j * 2 + 1] = (lo & 8) != 0 ? -loVal : loVal;
                }
            });
        }
        Tensor result = f32.CastTo(DType.F8E4M3);
        f32.Dispose();
        result.Fp8ScaleFactor = globalScale;
        return result;
    }

    /// <summary>Requantizes a group of fp8_scaled (E4M3) tensors IN PLACE to one common per-tensor scale —
    /// the enabler for concat-fusing separately-scaled projections (Q/K/V, FFN w1/w3) into a single GEMM,
    /// which was previously ruled out because one cuBLAS <c>alpha</c> cannot represent N different
    /// <see cref="Tensor.Fp8ScaleFactor"/>s. Picks <c>s* = max(sᵢ)</c>, decodes each tensor's bytes at its own
    /// scale, and re-encodes at <c>s*</c> — values only shrink (<c>sᵢ ≤ s*</c>), so nothing saturates.
    ///
    /// <para><b>Precision:</b> E4M3 is floating-point, so relative error is scale-invariant for values that
    /// stay in the NORMAL range after rescaling — those weights round-trip within half an E4M3 ulp
    /// (rel ≤ 1/16). Only weights whose real magnitude falls below <c>s*·2⁻⁶</c> (E4M3 min normal) enter the
    /// subnormal range and lose precision progressively, bounded by the group's scale ratio; weights below
    /// <c>s*·2⁻¹⁰</c> (half min subnormal) flush to zero. Both populations carry negligible GEMM energy at
    /// typical fp8_scaled ratios (≤ ~8×), but callers fusing groups with extreme ratios should A/B output
    /// quality. Returns the largest introduced decode error across the group, normalized per tensor to its
    /// own amax (<c>max |Δreal| / amax(real)</c> — the metric that bounds the GEMM contribution); 0 when all
    /// scales already match (nothing rewritten).</para></summary>
    public static float RequantizeToCommonFp8Scale(params Tensor[] tensors)
    {
        if (tensors is null || tensors.Length < 2)
            throw new ArgumentException("RequantizeToCommonFp8Scale needs at least two tensors to unify.", nameof(tensors));
        float commonScale = 0f;
        foreach (Tensor t in tensors)
        {
            if (t.DType != DType.F8E4M3)
                throw new ArgumentException($"RequantizeToCommonFp8Scale requires F8E4M3 tensors; got {t.DType}.");
            commonScale = Math.Max(commonScale, t.Fp8ScaleFactor);
        }

        float maxError = 0f;
        foreach (Tensor t in tensors)
        {
            if (t.Fp8ScaleFactor == commonScale) continue;

            // CastTo(F32) folds Fp8ScaleFactor → true weight values. Dividing by s* and re-encoding puts the
            // bytes on the common scale; the rewrite is in place so weight-dictionary references stay valid.
            // Load-time-only cost.
            Tensor real = t.CastTo(DType.F32);
            long n = real.ElementCount;
            float* rp = (float*)real.DataPointer;

            Tensor scaled = new Tensor(real.Shape, DType.F32);
            float* sp = (float*)scaled.DataPointer;
            float inv = 1.0f / commonScale;
            float amax = 0f;
            for (long i = 0; i < n; i++)
            {
                sp[i] = rp[i] * inv;
                amax = Math.Max(amax, Math.Abs(rp[i]));
            }
            Tensor requantized = scaled.CastTo(DType.F8E4M3);
            scaled.Dispose();

            Buffer.MemoryCopy((byte*)requantized.DataPointer, (byte*)t.DataPointer, n, n);
            requantized.Dispose();
            t.Fp8ScaleFactor = commonScale;

            // Introduced error vs the original decode, normalized to this tensor's amax.
            if (amax > 0f)
            {
                Tensor newReal = t.CastTo(DType.F32);
                float* np = (float*)newReal.DataPointer;
                float invAmax = 1.0f / amax;
                for (long i = 0; i < n; i++)
                    maxError = Math.Max(maxError, Math.Abs(np[i] - rp[i]) * invAmax);
                newReal.Dispose();
            }
            real.Dispose();
        }
        return maxError;
    }

    /// <summary>Row-concatenates two rank-2 weight tensors <c>[r1, C]</c> + <c>[r2, C]</c> → <c>[r1+r2, C]</c>
    /// (host memcpy — row-major rows are contiguous). Same dtype and column count required; for fp8 tensors the
    /// caller must have unified the scales first (<see cref="RequantizeToCommonFp8Scale"/>) — the concat carries
    /// <paramref name="a"/>'s <see cref="Tensor.Fp8ScaleFactor"/> and throws if <paramref name="b"/>'s differs.</summary>
    public static unsafe Tensor ConcatRowsHost(Tensor a, Tensor b)
    {
        if (a.Shape.Rank != 2 || b.Shape.Rank != 2)
            throw new ArgumentException($"ConcatRowsHost needs rank-2 tensors; got {a.Shape} and {b.Shape}.");
        if (a.Shape[1] != b.Shape[1])
            throw new ArgumentException($"ConcatRowsHost column mismatch: {a.Shape} vs {b.Shape}.");
        if (a.DType != b.DType)
            throw new ArgumentException($"ConcatRowsHost dtype mismatch: {a.DType} vs {b.DType}.");
        if (a.DType == DType.F8E4M3 && a.Fp8ScaleFactor != b.Fp8ScaleFactor)
            throw new ArgumentException("ConcatRowsHost: fp8 scales differ — call RequantizeToCommonFp8Scale first.");

        Tensor fused = new Tensor(new TensorShape(a.Shape[0] + b.Shape[0], a.Shape[1]), a.DType);
        long aBytes = a.ElementCount * a.DType.SizeInBytes;
        long bBytes = b.ElementCount * b.DType.SizeInBytes;
        Buffer.MemoryCopy((void*)a.DataPointer, (void*)fused.DataPointer, aBytes + bBytes, aBytes);
        Buffer.MemoryCopy((void*)b.DataPointer, (byte*)fused.DataPointer + aBytes, bBytes, bBytes);
        fused.Fp8ScaleFactor = a.Fp8ScaleFactor;
        return fused;
    }

    /// <summary>Fuses SwiGLU FFN weight pairs (<c>…w1Suffix</c> gate + <c>…w3Suffix</c> up) into single
    /// row-concatenated <c>…fusedSuffix</c> tensors — one GEMM instead of two (INFERENCE_ACCEL_GRIND §H3).
    /// fp8_scaled pairs are first unified via <see cref="RequantizeToCommonFp8Scale"/>; a pair whose
    /// introduced requant error exceeds <paramref name="maxRequantError"/> is left UNFUSED (per-block
    /// fallback — extreme scale ratios would visibly perturb output). Mixed-dtype or mismatched-shape pairs
    /// are skipped. Returns (fused count, worst accepted requant error).</summary>
    public static (int Fused, float WorstError) FuseSwiGluPairs(Dictionary<string, Tensor> weights,
        string w1Suffix, string w3Suffix, string fusedSuffix, float maxRequantError = 1f / 16f)
    {
        List<string> w1Keys = new();
        foreach (string k in weights.Keys)
            if (k.EndsWith(w1Suffix, StringComparison.Ordinal)) w1Keys.Add(k);

        int fusedCount = 0;
        float worst = 0f;
        foreach (string w1Key in w1Keys)
        {
            string baseKey = w1Key.Substring(0, w1Key.Length - w1Suffix.Length);
            string w3Key = baseKey + w3Suffix;
            if (!weights.TryGetValue(w3Key, out Tensor? w3)) continue;
            Tensor w1 = weights[w1Key];
            if (w1.DType != w3.DType || w1.Shape.Rank != 2 || w3.Shape.Rank != 2 || w1.Shape[1] != w3.Shape[1]) continue;

            float err = 0f;
            if (w1.DType == DType.F8E4M3 && w1.Fp8ScaleFactor != w3.Fp8ScaleFactor)
            {
                err = RequantizeToCommonFp8Scale(w1, w3);
                if (err > maxRequantError) continue;   // leave this block unfused rather than degrade it
            }
            weights[baseKey + fusedSuffix] = ConcatRowsHost(w1, w3);
            weights.Remove(w1Key);
            weights.Remove(w3Key);
            fusedCount++;
            worst = Math.Max(worst, err);
        }
        return (fusedCount, worst);
    }
}
