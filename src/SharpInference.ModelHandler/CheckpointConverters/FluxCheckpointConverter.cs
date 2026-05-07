using SharpInference.Core.Tensors;
using SharpInference.ModelHandler.CheckpointConverters.Utils;
using SharpInference.ModelHandler.SafeTensors;

namespace SharpInference.ModelHandler.CheckpointConverters;

/// <summary>Converts single-file Flux checkpoints (BFL / ComfyUI format) to diffusers-format weight dictionaries. Handles fused QKV splitting for double-stream blocks and fused linear1 splitting for single-stream blocks.</summary>
public sealed class FluxCheckpointConverter
{
    /// <summary>Result of converting a single-file Flux checkpoint into per-component weight dictionaries.</summary>
    public sealed class ConvertedWeights
    {
        /// <summary>Flux transformer weights in diffusers format.</summary>
        public required Dictionary<string, Tensor> Transformer { get; init; }

        /// <summary>CLIP-L text encoder weights.</summary>
        public required Dictionary<string, Tensor> ClipL { get; init; }

        /// <summary>T5-XXL text encoder weights.</summary>
        public required Dictionary<string, Tensor> T5 { get; init; }

        /// <summary>VAE weights in diffusers format.</summary>
        public required Dictionary<string, Tensor> Vae { get; init; }
    }

    /// <summary>Hidden size for Flux.1 models (3072). Used for QKV split sizing.</summary>
    private const int HiddenSize = 3072;

    /// <summary>MLP inner dimension (4 * 3072 = 12288).</summary>
    private const int MlpDim = 12288;

    /// <summary>Converts a single-file Flux checkpoint into separate per-component weight dictionaries. Auto-detects BFL vs diffusers format.</summary>
    public static ConvertedWeights Convert(Dictionary<string, Tensor> allWeights)
    {
        // Pre-process: detect ComfyUI fp8_scaled format (presence of `*.scale_weight` companion tensors)
        // and dequant matching FP8 weights to F16 by multiplying each value by its scalar scale.
        // Drops the `.scale_weight` and `.scale_input` metadata keys from the dict so the rest of
        // the converter can run unchanged on a "regular" fp8 weight set.
        allWeights = ApplyFp8ScaledDequant(allWeights);

        Dictionary<string, Tensor> transformer = new(4000);
        Dictionary<string, Tensor> clipL = new(200);
        Dictionary<string, Tensor> t5 = new(800);
        Dictionary<string, Tensor> vae = new(250);

        // Detect format: BFL uses "double_blocks", diffusers uses "transformer_blocks"
        bool isBflFormat = false;
        foreach (string key in allWeights.Keys)
        {
            if (key.StartsWith("double_blocks.", StringComparison.Ordinal) ||
                key.StartsWith("model.diffusion_model.double_blocks.", StringComparison.Ordinal))
            {
                isBflFormat = true;
                break;
            }
        }

        foreach (KeyValuePair<string, Tensor> kvp in allWeights)
        {
            string key = kvp.Key;
            Tensor tensor = kvp.Value;

            if (isBflFormat)
            {
                // BFL format: model.diffusion_model.* prefix or bare keys
                string bflKey = key;
                if (key.StartsWith("model.diffusion_model.", StringComparison.Ordinal))
                    bflKey = key["model.diffusion_model.".Length..];

                if (bflKey.StartsWith("double_blocks.", StringComparison.Ordinal) ||
                    bflKey.StartsWith("single_blocks.", StringComparison.Ordinal) ||
                    bflKey.StartsWith("img_in.", StringComparison.Ordinal) ||
                    bflKey.StartsWith("txt_in.", StringComparison.Ordinal) ||
                    bflKey.StartsWith("time_in.", StringComparison.Ordinal) ||
                    bflKey.StartsWith("vector_in.", StringComparison.Ordinal) ||
                    bflKey.StartsWith("guidance_in.", StringComparison.Ordinal) ||
                    bflKey.StartsWith("final_layer.", StringComparison.Ordinal))
                {
                    ConvertBflTransformerKey(bflKey, tensor, transformer);
                }
                else if (key.StartsWith("text_encoders.clip_l.", StringComparison.Ordinal) ||
                         key.StartsWith("conditioner.embedders.0.", StringComparison.Ordinal))
                {
                    ConvertClipL(key, tensor, clipL);
                }
                else if (key.StartsWith("text_encoders.t5xxl.", StringComparison.Ordinal))
                {
                    ConvertT5(key, tensor, t5);
                }
                else if (key.StartsWith("vae.", StringComparison.Ordinal) ||
                         key.StartsWith("first_stage_model.", StringComparison.Ordinal))
                {
                    ConvertVae(key, tensor, vae);
                }
            }
            else
            {
                // Diffusers format: keys are already in the expected naming
                if (key.StartsWith("transformer_blocks.", StringComparison.Ordinal) ||
                    key.StartsWith("single_transformer_blocks.", StringComparison.Ordinal) ||
                    key.StartsWith("x_embedder.", StringComparison.Ordinal) ||
                    key.StartsWith("context_embedder.", StringComparison.Ordinal) ||
                    key.StartsWith("time_text_embed.", StringComparison.Ordinal) ||
                    key.StartsWith("norm_out.", StringComparison.Ordinal) ||
                    key.StartsWith("proj_out.", StringComparison.Ordinal))
                {
                    transformer[key] = tensor;
                }
                else if (key.StartsWith("text_encoder.", StringComparison.Ordinal))
                {
                    clipL[key["text_encoder.".Length..]] = tensor;
                }
                else if (key.StartsWith("text_encoder_2.", StringComparison.Ordinal))
                {
                    t5[key["text_encoder_2.".Length..]] = tensor;
                }
            }
        }

        return new ConvertedWeights
        {
            Transformer = transformer,
            ClipL = clipL,
            T5 = t5,
            Vae = vae,
        };
    }

    /// <summary>Loads a single-file checkpoint and converts it in one step.</summary>
    public static (ConvertedWeights weights, SafeTensorsLoader loader) LoadAndConvert(string checkpointPath)
    {
        SafeTensorsLoader loader = new();
        loader.Load(checkpointPath);
        ConvertedWeights converted = Convert(loader.GetAllTensors());
        return (converted, loader);
    }

    /// <summary>
    /// Detects ComfyUI's <c>fp8_scaled</c> format and folds per-tensor <c>scale_weight</c> values
    /// into each FP8 weight tensor's <see cref="Tensor.Fp8ScaleFactor"/>. This keeps weights at
    /// native FP8 (1 byte/element) so a 12B-param transformer stays under 12GB RAM rather than
    /// ballooning to 24GB after F16 dequant. The scale is then applied for free at GEMM time via
    /// cuBLAS' <c>alpha</c> parameter — see <c>CudaBackend.Linear</c>.
    /// <para>The companion <c>scale_input</c> tensors (activation-quantization scales used only
    /// for true FP8 GEMM with FP8 activations) are dropped — we cast FP8→F16 and run F16 GEMM, so
    /// activations stay at full F16 magnitude.</para>
    /// Returns the original dict unchanged when no <c>scale_weight</c> keys are present.
    /// </summary>
    private static unsafe Dictionary<string, Tensor> ApplyFp8ScaledDequant(Dictionary<string, Tensor> source)
    {
        // Pre-scan for any scale_weight keys to decide whether this is the scaled format.
        Dictionary<string, Tensor> scaleWeights = new();
        foreach (KeyValuePair<string, Tensor> kvp in source)
        {
            if (kvp.Key.EndsWith(".scale_weight", StringComparison.Ordinal))
            {
                string baseKey = kvp.Key[..^".scale_weight".Length];
                scaleWeights[baseKey] = kvp.Value;
            }
        }
        if (scaleWeights.Count == 0)
            return source;

        Dictionary<string, Tensor> result = new(source.Count - 2 * scaleWeights.Count);
        foreach (KeyValuePair<string, Tensor> kvp in source)
        {
            // Drop the scale companion keys — they're consumed by being folded into Fp8ScaleFactor.
            if (kvp.Key.EndsWith(".scale_weight", StringComparison.Ordinal) ||
                kvp.Key.EndsWith(".scale_input", StringComparison.Ordinal))
            {
                continue;
            }

            // For an FP8 weight tensor with a matching scale companion, attach the scale to the
            // tensor's Fp8ScaleFactor so the GEMM call site can fold it into cuBLAS alpha.
            if (kvp.Value.DType == DType.F8E4M3 &&
                kvp.Key.EndsWith(".weight", StringComparison.Ordinal))
            {
                string baseKey = kvp.Key[..^".weight".Length];
                if (scaleWeights.TryGetValue(baseKey, out Tensor? scaleT) && scaleT.DType == DType.F32)
                {
                    float scale = ((float*)scaleT.DataPointer)[0];
                    kvp.Value.Fp8ScaleFactor = scale;
                }
            }

            result[kvp.Key] = kvp.Value;
        }
        return result;
    }

    #region BFL Transformer Key Conversion

    private static void ConvertBflTransformerKey(string bflKey, Tensor tensor, Dictionary<string, Tensor> output)
    {
        // Image input projection
        if (bflKey.StartsWith("img_in.", StringComparison.Ordinal))
        {
            output["x_embedder." + bflKey["img_in.".Length..]] = tensor;
            return;
        }

        // Text input projection
        if (bflKey.StartsWith("txt_in.", StringComparison.Ordinal))
        {
            output["context_embedder." + bflKey["txt_in.".Length..]] = tensor;
            return;
        }

        // Timestep embedder
        if (bflKey.StartsWith("time_in.", StringComparison.Ordinal))
        {
            ConvertEmbedderKey(bflKey["time_in.".Length..], "time_text_embed.timestep_embedder", tensor, output);
            return;
        }

        // Pooled text (CLIP) embedder
        if (bflKey.StartsWith("vector_in.", StringComparison.Ordinal))
        {
            ConvertEmbedderKey(bflKey["vector_in.".Length..], "time_text_embed.text_embedder", tensor, output);
            return;
        }

        // Guidance embedder (Dev only)
        if (bflKey.StartsWith("guidance_in.", StringComparison.Ordinal))
        {
            ConvertEmbedderKey(bflKey["guidance_in.".Length..], "time_text_embed.guidance_embedder", tensor, output);
            return;
        }

        // Final layer
        if (bflKey.StartsWith("final_layer.", StringComparison.Ordinal))
        {
            ConvertFinalLayerKey(bflKey["final_layer.".Length..], tensor, output);
            return;
        }

        // Double-stream blocks
        if (bflKey.StartsWith("double_blocks.", StringComparison.Ordinal))
        {
            ConvertDoubleBlockKey(bflKey["double_blocks.".Length..], tensor, output);
            return;
        }

        // Single-stream blocks
        if (bflKey.StartsWith("single_blocks.", StringComparison.Ordinal))
        {
            ConvertSingleBlockKey(bflKey["single_blocks.".Length..], tensor, output);
            return;
        }
    }

    /// <summary>Converts BFL embedder key (in_layer/out_layer) to diffusers format (linear_1/linear_2).</summary>
    private static void ConvertEmbedderKey(string rest, string diffusersPrefix, Tensor tensor, Dictionary<string, Tensor> output)
    {
        if (rest.StartsWith("in_layer.", StringComparison.Ordinal))
        {
            output[$"{diffusersPrefix}.linear_1.{rest["in_layer.".Length..]}"] = tensor;
        }
        else if (rest.StartsWith("out_layer.", StringComparison.Ordinal))
        {
            output[$"{diffusersPrefix}.linear_2.{rest["out_layer.".Length..]}"] = tensor;
        }
    }

    private static void ConvertFinalLayerKey(string rest, Tensor tensor, Dictionary<string, Tensor> output)
    {
        if (rest.StartsWith("adaLN_modulation.1.", StringComparison.Ordinal))
        {
            // BFL final-layer chunks the modulation as [shift, scale] (rows 0..H = shift, H..2H = scale)
            // and applies (1 + scale) * norm(x) + shift. Diffusers' AdaLayerNormContinuous chunks as
            // [scale, shift] instead. Our C# transformer's final layer matches the diffusers convention
            // (modParams[0..dim] = scale, modParams[dim..2*dim] = shift), so BFL weights need their
            // halves swapped along dim 0 to match. See diffusers' convert_flux_to_diffusers.py
            // `swap_scale_shift`.
            Tensor swapped = SwapScaleShiftHalves(tensor);
            output["norm_out.linear." + rest["adaLN_modulation.1.".Length..]] = swapped;
            return;
        }
        if (rest.StartsWith("linear.", StringComparison.Ordinal))
        {
            output["proj_out." + rest["linear.".Length..]] = tensor;
            return;
        }
    }

    /// <summary>Swaps the two halves of a tensor along dim 0. Input shape [2*H, ...] becomes [scale_half, shift_half] from BFL's [shift_half, scale_half] (or vice versa). Works for both 2D weights and 1D biases.</summary>
    private static unsafe Tensor SwapScaleShiftHalves(Tensor input)
    {
        long firstDim = input.Shape[0];
        if (firstDim % 2 != 0)
            throw new InvalidOperationException(
                $"SwapScaleShiftHalves: first dim must be even, got {firstDim}");

        long halfDim = firstDim / 2;
        long totalElements = input.ElementCount;
        long elemBytes = input.DType.SizeInBytes;
        long halfBytes = (totalElements / 2) * elemBytes;

        Tensor swapped = new Tensor(input.Shape, input.DType);
        // Propagate fp8_scaled per-tensor scale — the swap is a row-permutation, not a change of magnitudes.
        swapped.Fp8ScaleFactor = input.Fp8ScaleFactor;

        byte* src = (byte*)input.DataPointer;
        byte* dst = (byte*)swapped.DataPointer;

        // 2nd half of input → 1st half of swapped
        Buffer.MemoryCopy(src + halfBytes, dst, halfBytes, halfBytes);
        // 1st half of input → 2nd half of swapped
        Buffer.MemoryCopy(src, dst + halfBytes, halfBytes, halfBytes);

        return swapped;
    }

    private static void ConvertDoubleBlockKey(string rest, Tensor tensor, Dictionary<string, Tensor> output)
    {
        // Parse block index: "{i}.img_mod.lin.*" etc.
        int firstDot = rest.IndexOf('.');
        if (firstDot < 0) return;
        string blockIdxStr = rest[..firstDot];
        string afterBlockIdx = rest[(firstDot + 1)..];
        string prefix = $"transformer_blocks.{blockIdxStr}";

        // Image modulation
        if (afterBlockIdx.StartsWith("img_mod.lin.", StringComparison.Ordinal))
        {
            output[$"{prefix}.norm1.linear.{afterBlockIdx["img_mod.lin.".Length..]}"] = tensor;
            return;
        }

        // Text modulation
        if (afterBlockIdx.StartsWith("txt_mod.lin.", StringComparison.Ordinal))
        {
            output[$"{prefix}.norm1_context.linear.{afterBlockIdx["txt_mod.lin.".Length..]}"] = tensor;
            return;
        }

        // Image attention
        if (afterBlockIdx.StartsWith("img_attn.", StringComparison.Ordinal))
        {
            ConvertDoubleBlockImgAttn(prefix, afterBlockIdx["img_attn.".Length..], tensor, output);
            return;
        }

        // Text attention
        if (afterBlockIdx.StartsWith("txt_attn.", StringComparison.Ordinal))
        {
            ConvertDoubleBlockTxtAttn(prefix, afterBlockIdx["txt_attn.".Length..], tensor, output);
            return;
        }

        // Image MLP: img_mlp.0 → ff.net.0.proj, img_mlp.2 → ff.net.2
        if (afterBlockIdx.StartsWith("img_mlp.", StringComparison.Ordinal))
        {
            ConvertMlpKey(prefix, afterBlockIdx["img_mlp.".Length..], "ff", tensor, output);
            return;
        }

        // Text MLP: txt_mlp.0 → ff_context.net.0.proj, txt_mlp.2 → ff_context.net.2
        if (afterBlockIdx.StartsWith("txt_mlp.", StringComparison.Ordinal))
        {
            ConvertMlpKey(prefix, afterBlockIdx["txt_mlp.".Length..], "ff_context", tensor, output);
            return;
        }
    }

    private static void ConvertDoubleBlockImgAttn(string prefix, string rest, Tensor tensor, Dictionary<string, Tensor> output)
    {
        // Fused QKV → split
        if (rest == "qkv.weight")
        {
            SplitQkvWeight(tensor, HiddenSize, prefix, "attn.to_q", "attn.to_k", "attn.to_v", output);
            return;
        }
        if (rest == "qkv.bias")
        {
            SplitQkvBias(tensor, HiddenSize, prefix, "attn.to_q", "attn.to_k", "attn.to_v", output);
            return;
        }

        // Output projection
        if (rest.StartsWith("proj.", StringComparison.Ordinal))
        {
            output[$"{prefix}.attn.to_out.0.{rest["proj.".Length..]}"] = tensor;
            return;
        }

        // QK-norm (two naming conventions: BFL norm_q.weight vs FP8 norm.query_norm.scale)
        if (rest == "norm_q.weight" || rest == "norm.query_norm.scale")
        {
            output[$"{prefix}.attn.norm_q.weight"] = tensor;
            return;
        }
        if (rest == "norm_k.weight" || rest == "norm.key_norm.scale")
        {
            output[$"{prefix}.attn.norm_k.weight"] = tensor;
            return;
        }
    }

    private static void ConvertDoubleBlockTxtAttn(string prefix, string rest, Tensor tensor, Dictionary<string, Tensor> output)
    {
        // Fused QKV → split
        if (rest == "qkv.weight")
        {
            SplitQkvWeight(tensor, HiddenSize, prefix, "attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj", output);
            return;
        }
        if (rest == "qkv.bias")
        {
            SplitQkvBias(tensor, HiddenSize, prefix, "attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj", output);
            return;
        }

        // Output projection
        if (rest.StartsWith("proj.", StringComparison.Ordinal))
        {
            output[$"{prefix}.attn.to_add_out.{rest["proj.".Length..]}"] = tensor;
            return;
        }

        // QK-norm (two naming conventions: BFL norm_q.weight vs FP8 norm.query_norm.scale)
        if (rest == "norm_q.weight" || rest == "norm.query_norm.scale")
        {
            output[$"{prefix}.attn.norm_added_q.weight"] = tensor;
            return;
        }
        if (rest == "norm_k.weight" || rest == "norm.key_norm.scale")
        {
            output[$"{prefix}.attn.norm_added_k.weight"] = tensor;
            return;
        }
    }

    private static void ConvertSingleBlockKey(string rest, Tensor tensor, Dictionary<string, Tensor> output)
    {
        // Parse block index: "{i}.linear1.weight" etc.
        int firstDot = rest.IndexOf('.');
        if (firstDot < 0) return;
        string blockIdxStr = rest[..firstDot];
        string afterBlockIdx = rest[(firstDot + 1)..];
        string prefix = $"single_transformer_blocks.{blockIdxStr}";

        // Modulation
        if (afterBlockIdx.StartsWith("modulation.lin.", StringComparison.Ordinal))
        {
            output[$"{prefix}.norm.linear.{afterBlockIdx["modulation.lin.".Length..]}"] = tensor;
            return;
        }

        // Fused linear1: [3*hidden + mlpDim, hidden] = [21504, 3072]
        // Split into Q [3072, 3072], K [3072, 3072], V [3072, 3072], proj_mlp [12288, 3072]
        if (afterBlockIdx == "linear1.weight")
        {
            SplitSingleLinear1Weight(tensor, prefix, output);
            return;
        }
        if (afterBlockIdx == "linear1.bias")
        {
            SplitSingleLinear1Bias(tensor, prefix, output);
            return;
        }

        // linear2 → proj_out
        if (afterBlockIdx.StartsWith("linear2.", StringComparison.Ordinal))
        {
            output[$"{prefix}.proj_out.{afterBlockIdx["linear2.".Length..]}"] = tensor;
            return;
        }

        // QK-norm
        if (afterBlockIdx == "norm.query_norm.scale")
        {
            output[$"{prefix}.attn.norm_q.weight"] = tensor;
            return;
        }
        if (afterBlockIdx == "norm.key_norm.scale")
        {
            output[$"{prefix}.attn.norm_k.weight"] = tensor;
            return;
        }
    }

    /// <summary>Converts BFL MLP key format to diffusers. BFL: img_mlp.0.weight → ff.net.0.proj.weight, img_mlp.2.weight → ff.net.2.weight.</summary>
    private static void ConvertMlpKey(string prefix, string rest, string ffName, Tensor tensor, Dictionary<string, Tensor> output)
    {
        // rest is like "0.weight", "0.bias", "2.weight", "2.bias"
        if (rest.StartsWith("0.", StringComparison.Ordinal))
        {
            output[$"{prefix}.{ffName}.net.0.proj.{rest[2..]}"] = tensor;
        }
        else if (rest.StartsWith("2.", StringComparison.Ordinal))
        {
            output[$"{prefix}.{ffName}.net.2.{rest[2..]}"] = tensor;
        }
    }

    #endregion

    #region QKV Splitting

    /// <summary>Splits a fused QKV weight [3*innerDim, inDim] into three separate [innerDim, inDim] weights. Uses <see cref="DType.ComputeByteCount"/> rather than <c>SizeInBytes</c> so quantized fused-QKV tensors (Q4_K, Q5_K, Q8_0) split correctly — the row-aligned chunk size is computed from the per-row element count, which is block-aligned for ggml K-quants since Flux's hidden=3072 is a multiple of the 256-element super-block.</summary>
    private static unsafe void SplitQkvWeight(Tensor fused, int innerDim, string prefix,
        string qName, string kName, string vName, Dictionary<string, Tensor> output)
    {
        int inDim = (int)fused.Shape[1];
        long rowBytes = fused.DType.ComputeByteCount(inDim);
        long chunkBytes = (long)innerDim * rowBytes;
        TensorShape splitShape = new TensorShape(innerDim, inDim);

        Tensor qWeight = new Tensor(splitShape, fused.DType);
        Tensor kWeight = new Tensor(splitShape, fused.DType);
        Tensor vWeight = new Tensor(splitShape, fused.DType);

        // Propagate fp8_scaled per-tensor scale to all splits — they share the same quant scale.
        qWeight.Fp8ScaleFactor = fused.Fp8ScaleFactor;
        kWeight.Fp8ScaleFactor = fused.Fp8ScaleFactor;
        vWeight.Fp8ScaleFactor = fused.Fp8ScaleFactor;

        byte* src = (byte*)fused.DataPointer;
        Buffer.MemoryCopy(src, (void*)qWeight.DataPointer, chunkBytes, chunkBytes);
        Buffer.MemoryCopy(src + chunkBytes, (void*)kWeight.DataPointer, chunkBytes, chunkBytes);
        Buffer.MemoryCopy(src + 2 * chunkBytes, (void*)vWeight.DataPointer, chunkBytes, chunkBytes);

        output[$"{prefix}.{qName}.weight"] = qWeight;
        output[$"{prefix}.{kName}.weight"] = kWeight;
        output[$"{prefix}.{vName}.weight"] = vWeight;
    }

    /// <summary>Splits a fused QKV bias [3*innerDim] into three separate [innerDim] biases. ggml-quantized 1D biases would need block-alignment which 3072 satisfies (3072/256=12); but in practice ggml never quantizes biases (always F32/F16), so the quant path here is theoretical.</summary>
    private static unsafe void SplitQkvBias(Tensor fused, int innerDim, string prefix,
        string qName, string kName, string vName, Dictionary<string, Tensor> output)
    {
        long chunkBytes = fused.DType.ComputeByteCount(innerDim);
        TensorShape splitShape = new TensorShape(innerDim);

        Tensor qBias = new Tensor(splitShape, fused.DType);
        Tensor kBias = new Tensor(splitShape, fused.DType);
        Tensor vBias = new Tensor(splitShape, fused.DType);

        byte* src = (byte*)fused.DataPointer;
        Buffer.MemoryCopy(src, (void*)qBias.DataPointer, chunkBytes, chunkBytes);
        Buffer.MemoryCopy(src + chunkBytes, (void*)kBias.DataPointer, chunkBytes, chunkBytes);
        Buffer.MemoryCopy(src + 2 * chunkBytes, (void*)vBias.DataPointer, chunkBytes, chunkBytes);

        output[$"{prefix}.{qName}.bias"] = qBias;
        output[$"{prefix}.{kName}.bias"] = kBias;
        output[$"{prefix}.{vName}.bias"] = vBias;
    }

    /// <summary>Splits single-stream fused linear1 weight [3*hidden + mlpDim, hidden] = [21504, 3072] into Q, K, V, proj_mlp weights. Quantization-aware via <see cref="DType.ComputeByteCount"/>.</summary>
    private static unsafe void SplitSingleLinear1Weight(Tensor fused, string prefix, Dictionary<string, Tensor> output)
    {
        int inDim = (int)fused.Shape[1];
        long rowBytes = fused.DType.ComputeByteCount(inDim);

        // Split order: Q [HiddenSize rows], K [HiddenSize rows], V [HiddenSize rows], MLP [MlpDim rows]
        TensorShape qkvShape = new TensorShape(HiddenSize, inDim);
        TensorShape mlpShape = new TensorShape(MlpDim, inDim);

        Tensor qWeight = new Tensor(qkvShape, fused.DType);
        Tensor kWeight = new Tensor(qkvShape, fused.DType);
        Tensor vWeight = new Tensor(qkvShape, fused.DType);
        Tensor mlpWeight = new Tensor(mlpShape, fused.DType);

        // Propagate fp8_scaled per-tensor scale to all splits — they share the same quant scale.
        qWeight.Fp8ScaleFactor = fused.Fp8ScaleFactor;
        kWeight.Fp8ScaleFactor = fused.Fp8ScaleFactor;
        vWeight.Fp8ScaleFactor = fused.Fp8ScaleFactor;
        mlpWeight.Fp8ScaleFactor = fused.Fp8ScaleFactor;

        byte* src = (byte*)fused.DataPointer;
        long qkvChunkBytes = (long)HiddenSize * rowBytes;
        long mlpChunkBytes = (long)MlpDim * rowBytes;

        Buffer.MemoryCopy(src, (void*)qWeight.DataPointer, qkvChunkBytes, qkvChunkBytes);
        Buffer.MemoryCopy(src + qkvChunkBytes, (void*)kWeight.DataPointer, qkvChunkBytes, qkvChunkBytes);
        Buffer.MemoryCopy(src + 2 * qkvChunkBytes, (void*)vWeight.DataPointer, qkvChunkBytes, qkvChunkBytes);
        Buffer.MemoryCopy(src + 3 * qkvChunkBytes, (void*)mlpWeight.DataPointer, mlpChunkBytes, mlpChunkBytes);

        output[$"{prefix}.attn.to_q.weight"] = qWeight;
        output[$"{prefix}.attn.to_k.weight"] = kWeight;
        output[$"{prefix}.attn.to_v.weight"] = vWeight;
        output[$"{prefix}.proj_mlp.weight"] = mlpWeight;
    }

    /// <summary>Splits single-stream fused linear1 bias [3*hidden + mlpDim] = [21504] into Q, K, V, proj_mlp biases.</summary>
    private static unsafe void SplitSingleLinear1Bias(Tensor fused, string prefix, Dictionary<string, Tensor> output)
    {
        TensorShape qkvShape = new TensorShape(HiddenSize);
        TensorShape mlpShape = new TensorShape(MlpDim);

        Tensor qBias = new Tensor(qkvShape, fused.DType);
        Tensor kBias = new Tensor(qkvShape, fused.DType);
        Tensor vBias = new Tensor(qkvShape, fused.DType);
        Tensor mlpBias = new Tensor(mlpShape, fused.DType);

        byte* src = (byte*)fused.DataPointer;
        long qkvChunkBytes = fused.DType.ComputeByteCount(HiddenSize);
        long mlpChunkBytes = fused.DType.ComputeByteCount(MlpDim);

        Buffer.MemoryCopy(src, (void*)qBias.DataPointer, qkvChunkBytes, qkvChunkBytes);
        Buffer.MemoryCopy(src + qkvChunkBytes, (void*)kBias.DataPointer, qkvChunkBytes, qkvChunkBytes);
        Buffer.MemoryCopy(src + 2 * qkvChunkBytes, (void*)vBias.DataPointer, qkvChunkBytes, qkvChunkBytes);
        Buffer.MemoryCopy(src + 3 * qkvChunkBytes, (void*)mlpBias.DataPointer, mlpChunkBytes, mlpChunkBytes);

        output[$"{prefix}.attn.to_q.bias"] = qBias;
        output[$"{prefix}.attn.to_k.bias"] = kBias;
        output[$"{prefix}.attn.to_v.bias"] = vBias;
        output[$"{prefix}.proj_mlp.bias"] = mlpBias;
    }

    #endregion

    #region CLIP-L Key Conversion

    private static void ConvertClipL(string key, Tensor tensor, Dictionary<string, Tensor> clipL)
    {
        // ComfyUI: text_encoders.clip_l.transformer.text_model.* → text_model.*
        if (key.StartsWith("text_encoders.clip_l.transformer.", StringComparison.Ordinal))
        {
            string rest = key["text_encoders.clip_l.transformer.".Length..];
            if (rest.EndsWith("position_ids", StringComparison.Ordinal)) return;
            clipL[rest] = tensor;
            return;
        }

        // Stability: conditioner.embedders.0.transformer.text_model.* → text_model.*
        if (key.StartsWith("conditioner.embedders.0.transformer.", StringComparison.Ordinal))
        {
            string rest = key["conditioner.embedders.0.transformer.".Length..];
            if (rest.EndsWith("position_ids", StringComparison.Ordinal)) return;
            clipL[rest] = tensor;
        }
    }

    #endregion

    #region T5 Key Conversion

    private static void ConvertT5(string key, Tensor tensor, Dictionary<string, Tensor> t5)
    {
        // text_encoders.t5xxl.transformer.* → *
        string prefix = "text_encoders.t5xxl.transformer.";
        if (!key.StartsWith(prefix, StringComparison.Ordinal)) return;
        string rest = key[prefix.Length..];
        t5[rest] = tensor;
    }

    #endregion

    #region VAE Key Conversion

    private static void ConvertVae(string key, Tensor tensor, Dictionary<string, Tensor> vae)
    {
        string ldmKey;
        if (key.StartsWith("first_stage_model.", StringComparison.Ordinal))
            ldmKey = key["first_stage_model.".Length..];
        else if (key.StartsWith("vae.", StringComparison.Ordinal))
            ldmKey = key["vae.".Length..];
        else
            return;

        string? diffusersKey = CheckpointConvertUtils.ConvertVaeKey(ldmKey);
        if (diffusersKey is not null)
            vae[diffusersKey] = tensor;
    }

    #endregion

    /// <summary>Auto-detects model depth from transformer weights by counting block prefixes.</summary>
    public static (int doubleBlocks, int singleBlocks, bool hasGuidance) DetectArchitecture(
        Dictionary<string, Tensor> transformerWeights)
    {
        int maxDouble = -1;
        int maxSingle = -1;
        bool hasGuidance = false;

        foreach (string key in transformerWeights.Keys)
        {
            if (key.StartsWith("transformer_blocks.", StringComparison.Ordinal))
            {
                string afterPrefix = key["transformer_blocks.".Length..];
                int dot = afterPrefix.IndexOf('.');
                if (dot > 0 && int.TryParse(afterPrefix[..dot], out int blockIdx) && blockIdx > maxDouble)
                    maxDouble = blockIdx;
            }
            else if (key.StartsWith("single_transformer_blocks.", StringComparison.Ordinal))
            {
                string afterPrefix = key["single_transformer_blocks.".Length..];
                int dot = afterPrefix.IndexOf('.');
                if (dot > 0 && int.TryParse(afterPrefix[..dot], out int blockIdx) && blockIdx > maxSingle)
                    maxSingle = blockIdx;
            }
            else if (key.StartsWith("time_text_embed.guidance_embedder", StringComparison.Ordinal))
            {
                hasGuidance = true;
            }
        }

        return (maxDouble + 1, maxSingle + 1, hasGuidance);
    }
}
