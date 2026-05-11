using SharpInference.Core.Tensors;
using SharpInference.ModelHandler.CheckpointConverters.Utils;
using SharpInference.ModelHandler.SafeTensors;

namespace SharpInference.ModelHandler.CheckpointConverters;

/// <summary>Converts <c>fal/AuraFlow-v0.3</c> single-file safetensors (BFL-style native naming) to diffusers-format weight dictionaries for the C# AuraFlowTransformer. Mirrors <c>diffusers/loaders/single_file_utils.py:convert_auraflow_transformer_checkpoint_to_diffusers</c>. AuraFlow's BFL single-file is bias-free for attention and FFN (only timestep MLP has biases); Pile-T5-XL and the SDXL VAE ship separately in the diffusers folder layout.</summary>
public sealed class AuraFlowCheckpointConverter
{
    /// <summary>Result of converting a single-file AuraFlow checkpoint. The <c>calcuis/aura</c> FP8 single-file
    /// bundles transformer (<c>model.*</c>), Pile-T5-XL text encoder (<c>text_encoders.pile_t5xl.transformer.*</c>),
    /// and AuraFlow VAE (<c>vae.*</c>) all in one safetensors. The original <c>fal/AuraFlow-v0.3</c> BF16
    /// single-file ships transformer-only with no <c>model.</c> prefix; both formats are handled.</summary>
    public sealed class ConvertedWeights
    {
        /// <summary>AuraFlow MMDiT transformer weights in diffusers format.</summary>
        public required Dictionary<string, Tensor> Transformer { get; init; }

        /// <summary>Pile-T5-XL encoder weights (HuggingFace T5 naming, prefix stripped). Empty when the
        /// single-file doesn't bundle T5 (the BF16 <c>fal/AuraFlow-v0.3</c> release).</summary>
        public required Dictionary<string, Tensor> T5 { get; init; }

        /// <summary>AuraFlow VAE weights (diffusers naming). Empty when the single-file doesn't bundle the VAE
        /// (the BF16 release expects you to load <c>aura_vae.safetensors</c> separately).</summary>
        public required Dictionary<string, Tensor> Vae { get; init; }
    }

    /// <summary>Converts a flat AuraFlow single-file weight dictionary into transformer + T5 + VAE buckets.
    /// Routes by prefix: <c>model.*</c> → transformer (strip <c>model.</c>, then run BFL→diffusers rename);
    /// <c>text_encoders.pile_t5xl.transformer.*</c> → T5 (strip the prefix); <c>vae.*</c> → VAE (strip <c>vae.</c>).
    /// Folds ComfyUI <c>.scale_weight</c> FP8 companions into <c>Tensor.Fp8ScaleFactor</c> first.</summary>
    public static ConvertedWeights Convert(Dictionary<string, Tensor> allWeights)
    {
        allWeights = CheckpointConvertUtils.ApplyFp8ScaledDequant(allWeights);

        Dictionary<string, Tensor> transformer = new(800);
        Dictionary<string, Tensor> t5 = new(250);
        Dictionary<string, Tensor> vae = new(280);

        const string ModelPrefix = "model.";
        const string T5Prefix = "text_encoders.pile_t5xl.transformer.";
        const string VaePrefix = "vae.";

        foreach (KeyValuePair<string, Tensor> kvp in allWeights)
        {
            string key = kvp.Key;
            Tensor tensor = kvp.Value;

            if (key.EndsWith(".scaled_fp8") || key == "scaled_fp8")
                continue;

            if (key.StartsWith(T5Prefix, StringComparison.Ordinal))
            {
                t5[key[T5Prefix.Length..]] = tensor;
                continue;
            }
            if (key.StartsWith(VaePrefix, StringComparison.Ordinal))
            {
                vae[key[VaePrefix.Length..]] = tensor;
                continue;
            }

            // Strip "model." prefix (calcuis FP8 ships transformer keys this way; fal/AuraFlow BF16 ships flat).
            string trKey = key.StartsWith(ModelPrefix, StringComparison.Ordinal) ? key[ModelPrefix.Length..] : key;
            ConvertKey(trKey, tensor, transformer);
        }

        return new ConvertedWeights { Transformer = transformer, T5 = t5, Vae = vae };
    }

    /// <summary>Loads from disk and converts in one shot. Returns the converted weights plus the loader
    /// (the caller is responsible for disposing the loader once weights are no longer needed).</summary>
    public static (ConvertedWeights weights, SafeTensorsLoader loader) LoadAndConvert(string checkpointPath)
    {
        SafeTensorsLoader loader = new();
        loader.Load(checkpointPath);
        Dictionary<string, Tensor> raw = loader.GetAllTensors();
        ConvertedWeights converted = Convert(raw);
        return (converted, loader);
    }

    private static void ConvertKey(string key, Tensor tensor, Dictionary<string, Tensor> output)
    {
        // Top-level passthroughs and renames
        if (key == "register_tokens")
        {
            output["register_tokens"] = tensor;
            return;
        }
        if (key == "positional_encoding")
        {
            output["pos_embed.pos_embed"] = tensor;
            return;
        }
        if (key == "final_linear.weight" || key == "final_linear.bias")
        {
            output[$"proj_out.{key["final_linear.".Length..]}"] = tensor;
            return;
        }
        if (key == "modF.1.weight")
        {
            // BFL native [shift, scale] → diffusers [scale, shift] for AuraFlowPreFinalBlock.
            output["norm_out.linear.weight"] = SwapScaleShiftHalves(tensor);
            return;
        }
        if (key == "cond_seq_linear.weight")
        {
            output["context_embedder.weight"] = tensor;
            return;
        }
        // BFL `init_x_linear` is the patch projection — a Linear over flattened patches that
        // diffusers names `pos_embed.proj.{weight,bias}`. The diffusers single_file converter
        // omits this rename (it loads `pos_embed.pos_embed` from `positional_encoding` but
        // doesn't translate `init_x_linear`); we add it here so the C# transformer's `LoadWeights`
        // can find `pos_embed.proj.{weight,bias}`.
        if (key == "init_x_linear.weight" || key == "init_x_linear.bias")
        {
            output[$"pos_embed.proj.{key["init_x_linear.".Length..]}"] = tensor;
            return;
        }
        if (key.StartsWith("t_embedder.mlp.0.", StringComparison.Ordinal))
        {
            output["time_step_proj.linear_1." + key["t_embedder.mlp.0.".Length..]] = tensor;
            return;
        }
        if (key.StartsWith("t_embedder.mlp.2.", StringComparison.Ordinal))
        {
            output["time_step_proj.linear_2." + key["t_embedder.mlp.2.".Length..]] = tensor;
            return;
        }

        // Per-block paths
        if (key.StartsWith("double_layers.", StringComparison.Ordinal))
        {
            ConvertDoubleBlock(key["double_layers.".Length..], tensor, output);
            return;
        }
        if (key.StartsWith("single_layers.", StringComparison.Ordinal))
        {
            ConvertSingleBlock(key["single_layers.".Length..], tensor, output);
            return;
        }

        // Anything else (including pos_embed.pos_embed if it ships pre-converted, etc.) — pass through
        output[key] = tensor;
    }

    private static void ConvertDoubleBlock(string rest, Tensor tensor, Dictionary<string, Tensor> output)
    {
        int firstDot = rest.IndexOf('.');
        if (firstDot < 0) return;
        string idxStr = rest[..firstDot];
        string subKey = rest[(firstDot + 1)..];
        string prefix = $"joint_transformer_blocks.{idxStr}";

        // Modulation linears (no bias in AuraFlow)
        if (subKey == "modX.1.weight") { output[$"{prefix}.norm1.linear.weight"] = tensor; return; }
        if (subKey == "modC.1.weight") { output[$"{prefix}.norm1_context.linear.weight"] = tensor; return; }

        // Image FFN (mlpX → ff)
        if (subKey == "mlpX.c_fc1.weight") { output[$"{prefix}.ff.linear_1.weight"] = tensor; return; }
        if (subKey == "mlpX.c_fc2.weight") { output[$"{prefix}.ff.linear_2.weight"] = tensor; return; }
        if (subKey == "mlpX.c_proj.weight") { output[$"{prefix}.ff.out_projection.weight"] = tensor; return; }

        // Text FFN (mlpC → ff_context)
        if (subKey == "mlpC.c_fc1.weight") { output[$"{prefix}.ff_context.linear_1.weight"] = tensor; return; }
        if (subKey == "mlpC.c_fc2.weight") { output[$"{prefix}.ff_context.linear_2.weight"] = tensor; return; }
        if (subKey == "mlpC.c_proj.weight") { output[$"{prefix}.ff_context.out_projection.weight"] = tensor; return; }

        // Image attention (w2* → to_q/k/v/out.0)
        if (subKey == "attn.w2q.weight") { output[$"{prefix}.attn.to_q.weight"] = tensor; return; }
        if (subKey == "attn.w2k.weight") { output[$"{prefix}.attn.to_k.weight"] = tensor; return; }
        if (subKey == "attn.w2v.weight") { output[$"{prefix}.attn.to_v.weight"] = tensor; return; }
        if (subKey == "attn.w2o.weight") { output[$"{prefix}.attn.to_out.0.weight"] = tensor; return; }

        // Text attention (w1* → add_q/k/v_proj, to_add_out)
        if (subKey == "attn.w1q.weight") { output[$"{prefix}.attn.add_q_proj.weight"] = tensor; return; }
        if (subKey == "attn.w1k.weight") { output[$"{prefix}.attn.add_k_proj.weight"] = tensor; return; }
        if (subKey == "attn.w1v.weight") { output[$"{prefix}.attn.add_v_proj.weight"] = tensor; return; }
        if (subKey == "attn.w1o.weight") { output[$"{prefix}.attn.to_add_out.weight"] = tensor; return; }

        // QK-norm — diffusers reference doesn't list this in the single-file converter, but the
        // AuraFlow attention does use `qk_norm="fp32_layer_norm"` so the weight must come from somewhere.
        // The published v0.3 single-file appears to NOT have qk-norm weights (the FP32LayerNorm has
        // `elementwise_affine=False` in some configurations). If the C# transformer's `_normQ`/`_normK`
        // require weights and they're missing from a particular checkpoint, the loader will throw.
        // Pass through unrecognized keys so we surface them at load time.
        output[$"{prefix}.{subKey}"] = tensor;
    }

    private static void ConvertSingleBlock(string rest, Tensor tensor, Dictionary<string, Tensor> output)
    {
        int firstDot = rest.IndexOf('.');
        if (firstDot < 0) return;
        string idxStr = rest[..firstDot];
        string subKey = rest[(firstDot + 1)..];
        string prefix = $"single_transformer_blocks.{idxStr}";

        if (subKey == "modCX.1.weight") { output[$"{prefix}.norm1.linear.weight"] = tensor; return; }

        if (subKey == "mlp.c_fc1.weight") { output[$"{prefix}.ff.linear_1.weight"] = tensor; return; }
        if (subKey == "mlp.c_fc2.weight") { output[$"{prefix}.ff.linear_2.weight"] = tensor; return; }
        if (subKey == "mlp.c_proj.weight") { output[$"{prefix}.ff.out_projection.weight"] = tensor; return; }

        if (subKey == "attn.w1q.weight") { output[$"{prefix}.attn.to_q.weight"] = tensor; return; }
        if (subKey == "attn.w1k.weight") { output[$"{prefix}.attn.to_k.weight"] = tensor; return; }
        if (subKey == "attn.w1v.weight") { output[$"{prefix}.attn.to_v.weight"] = tensor; return; }
        if (subKey == "attn.w1o.weight") { output[$"{prefix}.attn.to_out.0.weight"] = tensor; return; }

        output[$"{prefix}.{subKey}"] = tensor;
    }

    /// <summary>Swaps the two halves of a tensor along dim 0 (BFL <c>[shift, scale]</c> ↔ diffusers <c>[scale, shift]</c>).
    /// Identical implementation to the one in <see cref="FluxCheckpointConverter"/>; kept inline rather than shared
    /// to avoid making this small utility part of a public API contract.</summary>
    private static unsafe Tensor SwapScaleShiftHalves(Tensor input)
    {
        long firstDim = input.Shape[0];
        if (firstDim % 2 != 0)
            throw new InvalidOperationException($"SwapScaleShiftHalves: first dim must be even, got {firstDim}");

        long halfBytes = (input.ElementCount / 2) * input.DType.SizeInBytes;

        Tensor swapped = new(input.Shape, input.DType)
        {
            Fp8ScaleFactor = input.Fp8ScaleFactor,
        };

        byte* src = (byte*)input.DataPointer;
        byte* dst = (byte*)swapped.DataPointer;

        Buffer.MemoryCopy(src + halfBytes, dst, halfBytes, halfBytes);              // 2nd half → 1st half
        Buffer.MemoryCopy(src, dst + halfBytes, halfBytes, halfBytes);              // 1st half → 2nd half
        return swapped;
    }
}
