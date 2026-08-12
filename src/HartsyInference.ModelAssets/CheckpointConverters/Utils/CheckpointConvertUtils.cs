using System.Runtime.CompilerServices;
using System.Text;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Nvfp4;
using HartsyInference.ModelAssets.Quant;

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


    // ── ComfyUI int8_tensorwise ──────────────────────────────────────────

    /// <summary>Suffix of ComfyUI's per-output-row int8 dequant scale (the same suffix fp8 builds use for their
    /// per-tensor scalar; the weight's dtype is what tells the two apart).</summary>
    private const string WeightScaleSuffix = ".weight_scale";

    /// <summary>Parses a <c>.comfy_quant</c> descriptor tensor, tolerating the trailing NUL padding a fixed-width U8
    /// tensor can carry. Returns null when the tensor is not a descriptor or its JSON is unreadable.</summary>
    public static ComfyQuantDescriptor? TryReadComfyQuant(Tensor blob)
    {
        ArgumentNullException.ThrowIfNull(blob);
        if (blob.DType != DType.U8 || blob.ElementCount is <= 0 or > 4096)
            return null;
        ReadOnlySpan<byte> bytes = blob.AsReadOnlySpan<byte>();
        int end = bytes.Length;
        while (end > 0 && bytes[end - 1] == 0) end--;
        return ComfyQuantDescriptor.TryParse(bytes[..end]);
    }

    /// <summary>Moves ComfyUI's <c>int8_tensorwise</c> companions — the <c>.weight_scale</c> per-output-row scale and
    /// the <c>.comfy_quant</c> descriptor — onto <see cref="Tensor.QuantInfo"/> of the I8 weight they belong to, and
    /// returns a dictionary without those companion keys. The weight itself is left <b>packed</b>: dequantizing here
    /// would turn LTX 2.5's 21.5 GB int8 DiT back into 42 GB, which is the whole reason the format exists.</summary>
    /// <remarks><para><see cref="ApplyFp8ScaledDequant"/> calls this first, because that pass drops
    /// <c>.weight_scale</c>/<c>.comfy_quant</c> unconditionally — an int8 weight that skipped this step would reach
    /// the backend as raw int8 with no scale at all. Every converter funnels through it, so no caller can lose the
    /// companions by forgetting a call.</para>
    /// <para>Idempotent: a weight that already carries <see cref="Tensor.QuantInfo"/> is skipped. The companion
    /// tensors are <b>borrowed</b> — the loader that produced them still owns their lifetime.</para></remarks>
    public static Dictionary<string, Tensor> AttachInt8QuantInfo(Dictionary<string, Tensor> source)
    {
        ArgumentNullException.ThrowIfNull(source);
        List<string>? int8Weights = null;
        foreach (KeyValuePair<string, Tensor> kvp in source)
        {
            if (kvp.Value.DType == DType.I8 && kvp.Value.QuantInfo is null
                && kvp.Key.EndsWith(".weight", StringComparison.Ordinal))
            {
                (int8Weights ??= new List<string>()).Add(kvp.Key);
            }
        }
        if (int8Weights is null)
            return source;

        HashSet<string> companions = new HashSet<string>(StringComparer.Ordinal);
        foreach (string key in int8Weights)
        {
            string baseKey = key[..^".weight".Length];
            string scaleKey = baseKey + WeightScaleSuffix;
            string descriptorKey = baseKey + ComfyQuantDescriptor.Suffix;
            Tensor weight = source[key];
            if (weight.Shape.Rank != 2)
            {
                throw new NotSupportedException(
                    $"int8 weight '{key}' has shape {weight.Shape}; only rank-2 int8_tensorwise Linear weights are supported.");
            }
            if (!source.TryGetValue(scaleKey, out Tensor? rowScale))
            {
                throw new NotSupportedException(
                    $"int8 weight '{key}' has no '{scaleKey}' companion, so its dequant scale is unrecoverable. Every "
                    + "ComfyUI int8_tensorwise build ships one per quantized Linear — re-download the checkpoint, or "
                    + "use a BF16/fp8_scaled build instead.");
            }
            // A BF16 scalar read through (float*) is garbage, and a silently wrong scale is far worse than a refusal;
            // no shipped int8 build stores anything but F32 here.
            if (rowScale.DType != DType.F32)
                throw new NotSupportedException($"'{scaleKey}' must be F32; got {rowScale.DType}.");
            long rows = weight.Shape[0];
            if (rowScale.ElementCount != rows && rowScale.ElementCount != 1)
            {
                throw new NotSupportedException(
                    $"'{scaleKey}' must hold {rows} per-row scales or a single per-tensor scale; got {rowScale.ElementCount}.");
            }

            ComfyQuantDescriptor? descriptor = source.TryGetValue(descriptorKey, out Tensor? blob)
                ? TryReadComfyQuant(blob) : null;
            // Only the per-layer descriptor says whether the rows were ConvRot-rotated (the file-level
            // __metadata__ mirror can't be trusted — re-quants skip different layers), and consuming a rotated
            // weight as unrotated produces plausible-looking garbage instead of an error.
            if (descriptor is null)
            {
                throw new NotSupportedException(
                    $"int8 weight '{key}' has no readable '{descriptorKey}' descriptor, so whether it was ConvRot-rotated "
                    + "is unknowable and the weight cannot be consumed safely.");
            }
            int groupSize = descriptor.ConvRotGroupSize;
            if (groupSize > 0)
            {
                if (!Int8ConvRotCodec.IsValidGroupSize(groupSize))
                {
                    throw new NotSupportedException(
                        $"'{descriptorKey}' declares ConvRot group size {groupSize}, which is not a power of four.");
                }
                if (weight.Shape[1] % groupSize != 0)
                {
                    throw new NotSupportedException(
                        $"'{key}' has in_features {weight.Shape[1]}, which ConvRot group size {groupSize} does not divide.");
                }
            }

            weight.QuantInfo = new QuantWeightInfo
            {
                Format = descriptor.Format,
                RowScale = rowScale,
                ConvRotGroupSize = groupSize,
                FullPrecisionMatMul = descriptor.FullPrecisionMatMul,
            };
            companions.Add(scaleKey);
            companions.Add(descriptorKey);
        }

        Dictionary<string, Tensor> result = new Dictionary<string, Tensor>(source.Count);
        foreach (KeyValuePair<string, Tensor> kvp in source)
        {
            if (!companions.Contains(kvp.Key))
                result[kvp.Key] = kvp.Value;
        }
        return result;
    }


    // ── FP8 Scaled ──────────────────────────────────────────

    /// <summary>Folds per-tensor FP8 scale companions into <see cref="Tensor.Fp8ScaleFactor"/> on the matching weight tensors and drops the companions. Supports three companion formats:
    /// <list type="bullet">
    ///   <item>ComfyUI <c>fp8_scaled</c>: <c>.scale_weight</c>/<c>.scale_input</c> (F32 scalar).</item>
    ///   <item>BFL Mistral / Flux.2 Dev mixed-fp8: <c>.weight_scale</c>/<c>.input_scale</c> (F32 scalar).</item>
    ///   <item>ComfyUI <c>comfy_quant</c>: <c>.comfy_quant</c> (U8 JSON blob like <c>{"format":"float8_e4m3fn"}</c>) — newer format used by Chroma1-HD-fp8mixed and similar. There's no separately-stored scalar; fp8 values are used at identity scale (the model is trained with the natural fp8 dynamic range). The companion is purely a format declaration and must still be dropped or it pollutes the weight dictionary.</item>
    /// </list>
    /// The input-side scale is folded into <see cref="Tensor.Fp8InputScaleFactor"/> so the backend can quantize the
    /// activation with a constant rather than a per-call absmax; weights without one keep 0 and take the dynamic path. Marker tensors like <c>scaled_fp8</c> are also dropped.</summary>
    /// <param name="source">Raw checkpoint dictionary (mutated; companion keys removed).</param>
    /// <param name="nvfp4ToFp8">When true, NVFP4 weights are dequantized to <b>fp8 (1 byte/param)</b> with the block
    /// scale folded into the value and the global scale carried on <see cref="Tensor.Fp8ScaleFactor"/>, instead of
    /// the default F16 (2 byte/param). Halves the resident footprint of an all-nvfp4 model (Ideogram 4: two 9.3B
    /// DiTs, 35.9 GB at F16 → 18.6 GB at fp8, the difference between "won't fit a 24 GB card" and "fits"). Leave
    /// false for small nvfp4 text encoders (Z-Image's Qwen3-4B) where F16 is free and avoids fp8's smaller range.</param>
    /// <returns>A new dictionary without companion keys, with <c>Fp8ScaleFactor</c> populated on FP8 weights.</returns>
    /// <param name="residentNvfp4">Keep nvfp4 weights PACKED — relabelled <c>F4E2M1 [N, K]</c> with their scales on
    /// <see cref="Tensor.QuantInfo"/> — instead of unpacking them here. Opt-in, unlike int8: the eager path works and
    /// is what the CPU/Vulkan backends need, so only a caller that knows a CUDA backend will consume the weights
    /// should ask for it. AWQ layers carrying <c>pre_quant_scale</c> refuse and take the eager path regardless.</param>
    public static unsafe Dictionary<string, Tensor> ApplyFp8ScaledDequant(Dictionary<string, Tensor> source,
        bool nvfp4ToFp8 = false, bool residentNvfp4 = false)
    {
        // int8_tensorwise uses the same companion suffixes this pass drops, so its scales have to move onto the
        // weight before anything here can strip them.
        source = AttachInt8QuantInfo(source);

        // First pass: gather scale companions keyed by the base name (the part before the suffix).
        // `.weight_scale_2` is nvfp4's global scalar (block scales live in `.weight_scale`); note that
        // ".weight_scale_2".EndsWith(".weight_scale") is FALSE, so the two never collide in the buckets.
        Dictionary<string, Tensor> weightScales = new();
        Dictionary<string, Tensor> weightScale2s = new();
        Dictionary<string, Tensor> inputScales = new();
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
            else if (key.EndsWith(".scale_input", StringComparison.Ordinal))
            {
                inputScales[key[..^".scale_input".Length]] = kvp.Value;
                sawAnyScale = true;
            }
            else if (key.EndsWith(".input_scale", StringComparison.Ordinal))
            {
                inputScales[key[..^".input_scale".Length]] = kvp.Value;
                sawAnyScale = true;
            }
            else if (key.EndsWith(".comfy_quant", StringComparison.Ordinal) || key == "scaled_fp8")
            {
                sawAnyScale = true; // dropped, but flags the format as scaled (or fp8-declared, for comfy_quant)
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
                // The activation-side scalar, when the file ships one. Carrying it lets the backend quantize with a
                // constant instead of computing a per-call absmax over the whole activation. Note not every fp8
                // Linear has one: MiniMax-H3's mlp.fc2 ships none, because its comfy_quant declares
                // "full_precision_matrix_mult" and its input is never quantized at all.
                if (inputScales.TryGetValue(baseKey, out Tensor? inScaleT) && inScaleT.Shape.ElementCount == 1
                    && (inScaleT.DType == DType.F32 || inScaleT.DType == DType.F16 || inScaleT.DType == DType.BF16))
                {
                    if (inScaleT.DType == DType.F32)
                        kvp.Value.Fp8InputScaleFactor = ((float*)inScaleT.DataPointer)[0];
                    else
                    {
                        using Tensor inScaleF32 = inScaleT.CastTo(DType.F32);
                        kvp.Value.Fp8InputScaleFactor = ((float*)inScaleF32.DataPointer)[0];
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
                    // Resident: relabel to F4E2M1 [N, K] and hang the scales off QuantInfo rather than unpacking.
                    // Refuses AWQ pre_quant_scale layers, which fall through to the eager dequant below.
                    if (residentNvfp4 && Nvfp4Codec.TryAttachResident(kvp.Value, blockScales, scale2T,
                            source.ContainsKey($"{baseKey}.pre_quant_scale"), out Tensor residentWeight))
                    {
                        result[key] = residentWeight;
                        continue;
                    }

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
    public static void QuantizeDitBlocksToFp8(Dictionary<string, Tensor> weights, string blockKeyMarker,
        long minElements = 1L << 20)
    {
        foreach (string key in weights.Keys.ToList())
        {
            if (!key.Contains(blockKeyMarker, StringComparison.Ordinal)) continue;
            TryQuantizeWeightToFp8(weights, key, weights[key], minElements);
        }
    }

    /// <summary>The per-tensor half of <see cref="QuantizeDitBlocksToFp8"/>: same eligibility gate (rank-2 BF16/F16
    /// with at least <paramref name="minElements"/> elements, key ending in <c>.weight</c>) and same emitted bytes,
    /// but for ONE tensor, so a converter can requantize each weight the moment it is produced and free the wide
    /// intermediate before taking the next. Writes the fp8 tensor plus its <c>.scale_weight</c> [1] F32 companion into
    /// <paramref name="output"/> and returns true; returns false (writing nothing) when the tensor is not eligible.
    /// Does NOT dispose <paramref name="tensor"/> — the caller owns it and knows whether it is a borrowed mmap view.</summary>
    public static unsafe bool TryQuantizeWeightToFp8(Dictionary<string, Tensor> output, string key, Tensor tensor,
        long minElements = 1L << 20)
    {
        if (!key.EndsWith(".weight", StringComparison.Ordinal) || tensor.Shape.Rank != 2
            || tensor.ElementCount < minElements || (tensor.DType != DType.BF16 && tensor.DType != DType.F16))
        {
            return false;
        }
        Tensor fp8 = QuantizeToFp8Scaled(tensor);
        Tensor scale = new Tensor(new TensorShape(1), DType.F32);
        ((float*)scale.DataPointer)[0] = fp8.Fp8ScaleFactor;
        output[key] = fp8;
        output[key[..^".weight".Length] + ".scale_weight"] = scale;
        return true;
    }

    /// <summary>BF16/F16 weight → fp8 e4m3 with a per-tensor scale = absmax/448 (E4M3's max magnitude), folded onto
    /// <see cref="Tensor.Fp8ScaleFactor"/> so the GEMM computes <c>fp8_decoded · scale ≈ w</c> via cuBLAS alpha.
    /// Uses the engine's tested F32→F8E4M3 cast on a transient F32 copy (the source is left untouched).</summary>
    private static unsafe Tensor QuantizeToFp8Scaled(Tensor w)
    {
        Tensor f32 = w.CastTo(DType.F32);        // BF16/F16 → fresh F32 (source unmodified)
        Tensor fp8 = QuantizeF32ToFp8Scaled(f32);
        f32.Dispose();
        return fp8;
    }

    /// <summary>The same <c>absmax/448</c> per-tensor scheme as <see cref="QuantizeDitBlocksToFp8"/>, but entered with
    /// the wide values already in hand: a LoRA merge dequantizes an fp8 weight, adds its delta in F32, and hands the
    /// accumulator straight back here, so no second quantizer exists. <b>Overwrites <paramref name="f32"/>'s contents</b>
    /// (scaled in place) but does NOT dispose it — the caller owns it. Emits no <c>.scale_weight</c> companion; the
    /// scale rides on <see cref="Tensor.Fp8ScaleFactor"/> only, which is what an in-memory weight dict consumes.</summary>
    /// <param name="stochasticSeedKey">Tensor key seeding stochastic rounding, mirroring ComfyUI's <c>string_to_seed</c>; null keeps the plain round-to-nearest cast.</param>
    public static unsafe Tensor QuantizeF32ToFp8Scaled(Tensor f32, string? stochasticSeedKey = null)
    {
        ArgumentNullException.ThrowIfNull(f32);
        if (f32.DType != DType.F32)
            throw new ArgumentException($"An F32 accumulator is required, got {f32.DType}.", nameof(f32));

        long n = f32.ElementCount;
        float* p = (float*)f32.DataPointer;
        float absmax = AbsMax(p, n);
        float scale = absmax > 0f ? absmax / 448f : 1f;
        ScaleInPlace(p, n, 1f / scale);
        if (stochasticSeedKey is not null)
        {
            StochasticPreRoundToFp8Grid(p, n, Crc32(stochasticSeedKey));
        }
        Tensor fp8 = f32.CastTo(DType.F8E4M3);
        fp8.Fp8ScaleFactor = scale;
        return fp8;
    }

    /// <summary>Snaps each value onto an exactly-representable e4m3 grid point, picking between the two neighbours with
    /// probability proportional to distance. The engine's F32→F8E4M3 cast is round-to-nearest, which biases a weight
    /// systematically when many tensors are requantized; ComfyUI rounds stochastically instead
    /// (<c>comfy/float.py</c> <c>manual_stochastic_round_to_float8</c>), and this reproduces that mantissa-domain
    /// <c>floor(scaled + U[0,1))</c>. Pre-rounding rather than writing a second encoder keeps the tested cast as the
    /// only encoder — the values it then sees are already on the grid, so it is lossless.</summary>
    private static unsafe void StochasticPreRoundToFp8Grid(float* p, long n, uint seed)
    {
        if (n < ParallelPassMinElements)
        {
            StochasticPreRoundRange(p, n, seed, 0);
            return;
        }
        long chunks = Math.Min(Environment.ProcessorCount, Math.Max(1L, n / ParallelPassChunkElements));
        long perChunk = (n + chunks - 1) / chunks;
        nint basePtr = (nint)p;
        Parallel.For(0, (int)chunks, c =>
        {
            long start = c * perChunk;
            long length = Math.Min(perChunk, n - start);
            if (length > 0)
                StochasticPreRoundRange((float*)(basePtr + (nint)(start * sizeof(float))), length, seed, start);
        });
    }

    /// <summary>Values are drawn from the global element index, so chunking cannot perturb the result.</summary>
    private static unsafe void StochasticPreRoundRange(float* p, long n, uint seed, long indexOffset)
    {
        for (long i = 0; i < n; i++)
        {
            uint bits = BitConverter.SingleToUInt32Bits(p[i]);
            float absX = BitConverter.UInt32BitsToSingle(bits & 0x7FFFFFFFu);
            // Zero, NaN and out-of-range already land on a grid point (the cast saturates at ±448).
            if (!(absX > 0f) || absX > 448f) continue;

            // The float exponent field IS floor(log2|x|), so no transcendental is needed; a subnormal F32 reads
            // as -127 and clamps into the e4m3 subnormal branch, which is correct.
            int e = (int)((bits >> 23) & 0xFFu) - 127 + 7;
            if (e < 0) e = 0;
            else if (e > 15) e = 15;
            float pw = BitConverter.UInt32BitsToSingle((uint)((e - 7 + 127) << 23));
            float mantissa = e != 0 ? (absX / pw - 1f) * 8f : absX * 512f;
            mantissa = MathF.Floor(mantissa + NextUniform(seed, indexOffset + i));
            float rounded = e != 0 ? pw * (1f + mantissa * 0.125f) : mantissa * (1f / 512f);
            if (rounded > 448f) rounded = 448f;
            p[i] = BitConverter.UInt32BitsToSingle(BitConverter.SingleToUInt32Bits(rounded) | (bits & 0x80000000u));
        }
    }

    /// <summary>Uniform in [0,1) from a counter-based splitmix64 mix, so a draw depends only on its element index.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float NextUniform(uint seed, long index)
    {
        ulong z = seed + (ulong)index * 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        z ^= z >> 31;
        return (z >> 40) * (1f / 16777216f);
    }

    /// <summary>ComfyUI's <c>string_to_seed</c> verbatim — CRC-32 (reflected, poly 0xEDB88320) over the key's bytes.</summary>
    private static uint Crc32(string text)
    {
        uint crc = 0xFFFFFFFFu;
        foreach (byte b in Encoding.UTF8.GetBytes(text))
        {
            crc ^= b;
            for (int bit = 0; bit < 8; bit++)
            {
                crc = (crc & 1u) != 0u ? (crc >> 1) ^ 0xEDB88320u : crc >> 1;
            }
        }
        return crc ^ 0xFFFFFFFFu;
    }

    /// <summary>Element count at or above which the fp8 absmax/scale passes fan out across cores. Once vectorized these
    /// passes are memory-bound, so a <see cref="Parallel.For"/> dispatch (measured at 22µs for 16 chunks on this box) only
    /// pays for itself on a fairly large buffer: the measured crossover is 2^19 elements (vector-serial 91µs vs parallel
    /// 84µs; at 2^18 serial still wins 43µs vs 66µs). Everything reaching <see cref="QuantizeToFp8Scaled"/> is already
    /// ≥ <c>minElements</c> (2^20), so this is a guard for any future caller with smaller tensors.</summary>
    private const long ParallelPassMinElements = 1L << 19;

    /// <summary>Minimum elements per chunk, so a barely-over-threshold pass fans out to a few fat slices instead of
    /// <see cref="Environment.ProcessorCount"/> slivers that cost more to dispatch than to run.</summary>
    private const long ParallelPassChunkElements = 1L << 17;

    /// <summary>Largest <c>|x|</c> over <paramref name="n"/> floats, ignoring NaN.</summary>
    /// <remarks>Bit-identical to the scalar <c>if (MathF.Abs(p[i]) &gt; absmax) absmax = …</c> it replaces, including
    /// under NaN: the lane update uses <c>GreaterThan</c> + <c>ConditionalSelect</c>, NOT <see cref="Vector.Max{T}"/>,
    /// which lowers to <c>maxps</c> and would let a NaN lane win. Max is order-independent, so chunking across cores
    /// cannot perturb the result the way a sum reduction would.</remarks>
    private static unsafe float AbsMax(float* p, long n)
    {
        if (n < ParallelPassMinElements)
            return AbsMaxRange(p, n);

        long chunks = Math.Min(Environment.ProcessorCount, Math.Max(1L, n / ParallelPassChunkElements));
        long perChunk = (n + chunks - 1) / chunks;
        float[] partials = new float[chunks];
        nint basePtr = (nint)p;
        Parallel.For(0, (int)chunks, c =>
        {
            long start = c * perChunk;
            long length = Math.Min(perChunk, n - start);
            partials[c] = length > 0 ? AbsMaxRange((float*)(basePtr + (nint)(start * sizeof(float))), length) : 0f;
        });
        float absmax = 0f;
        foreach (float partial in partials)
        {
            if (partial > absmax) absmax = partial;
        }
        return absmax;
    }

    private static unsafe float AbsMaxRange(float* p, long n)
    {
        int width = Vector<float>.Count;
        Vector<float> acc = Vector<float>.Zero;
        long i = 0;
        for (; i <= n - width; i += width)
        {
            Vector<float> a = Vector.Abs(Vector.Load(p + i));
            acc = Vector.ConditionalSelect(Vector.GreaterThan(a, acc), a, acc);
        }
        float absmax = 0f;
        for (int lane = 0; lane < width; lane++)
        {
            if (acc[lane] > absmax) absmax = acc[lane];
        }
        for (; i < n; i++)
        {
            float a = MathF.Abs(p[i]);
            if (a > absmax) absmax = a;
        }
        return absmax;
    }

    /// <summary>Multiplies <paramref name="n"/> floats in place by <paramref name="inv"/>.</summary>
    /// <remarks>Bit-identical to the scalar loop: IEEE multiply is exact per element and every element is independent,
    /// so neither vector width nor chunk boundaries can change a result.</remarks>
    private static unsafe void ScaleInPlace(float* p, long n, float inv)
    {
        if (n < ParallelPassMinElements)
        {
            ScaleRange(p, n, inv);
            return;
        }
        long chunks = Math.Min(Environment.ProcessorCount, Math.Max(1L, n / ParallelPassChunkElements));
        long perChunk = (n + chunks - 1) / chunks;
        nint basePtr = (nint)p;
        Parallel.For(0, (int)chunks, c =>
        {
            long start = c * perChunk;
            long length = Math.Min(perChunk, n - start);
            if (length > 0)
                ScaleRange((float*)(basePtr + (nint)(start * sizeof(float))), length, inv);
        });
    }

    private static unsafe void ScaleRange(float* p, long n, float inv)
    {
        int width = Vector<float>.Count;
        Vector<float> factor = new Vector<float>(inv);
        long i = 0;
        for (; i <= n - width; i += width)
        {
            Vector.Store(Vector.Load(p + i) * factor, p + i);
        }
        for (; i < n; i++)
        {
            p[i] *= inv;
        }
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
            // comfy_quant checkpoints carry per-key quant descriptors the fused tensor cannot inherit —
            // fusing them produced correct F32 output but DEGENERATE F16 output (Ideogram4, 2026-07-22).
            // Skip until the fusion understands the blocked-quant layout end to end.
            string w1Head = w1Key.Substring(0, w1Key.Length - ".weight".Length);
            if (weights.ContainsKey(w1Head + ".comfy_quant") || weights.ContainsKey(w1Head + ".weight_scale")) continue;
            Tensor w1 = weights[w1Key];
            // Once AttachInt8QuantInfo has consumed the companion KEYS the check above can't see an int8 pair any
            // more; the row scale and rotation now ride on the tensor, and a concat would silently drop both.
            if (w1.QuantInfo is not null || w3.QuantInfo is not null) continue;
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
