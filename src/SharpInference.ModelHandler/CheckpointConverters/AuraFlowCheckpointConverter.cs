using SharpInference.Core.Tensors;
using SharpInference.ModelHandler.CheckpointConverters.Utils;
using SharpInference.ModelHandler.SafeTensors;

namespace SharpInference.ModelHandler.CheckpointConverters;

/// <summary>Converts <c>fal/AuraFlow-v0.3</c> single-file safetensors (BFL-style native naming) to
/// diffusers-format weight dictionaries for our C# <see cref="SharpInference.Diffusion.Models.Denoisers.AuraFlowTransformer"/>.
///
/// Mirrors <c>diffusers/loaders/single_file_utils.py:convert_auraflow_transformer_checkpoint_to_diffusers</c>:
/// <list type="bullet">
/// <item><c>double_layers.{i}.modX.1.weight</c> → <c>joint_transformer_blocks.{i}.norm1.linear.weight</c></item>
/// <item><c>double_layers.{i}.modC.1.weight</c> → <c>joint_transformer_blocks.{i}.norm1_context.linear.weight</c></item>
/// <item><c>double_layers.{i}.attn.w2{q,k,v,o}.weight</c> → image attention <c>attn.to_q/to_k/to_v/to_out.0</c></item>
/// <item><c>double_layers.{i}.attn.w1{q,k,v,o}.weight</c> → text attention <c>attn.add_q_proj/add_k_proj/add_v_proj/to_add_out</c></item>
/// <item><c>double_layers.{i}.mlpX.{c_fc1,c_fc2,c_proj}.weight</c> → image FFN <c>ff.linear_1/linear_2/out_projection</c></item>
/// <item><c>double_layers.{i}.mlpC.*</c> → text FFN <c>ff_context.*</c></item>
/// <item><c>single_layers.{i}.modCX.1.weight</c> → <c>single_transformer_blocks.{i}.norm1.linear.weight</c></item>
/// <item><c>single_layers.{i}.attn.w1{q,k,v,o}.weight</c> → single attention <c>attn.to_q/to_k/to_v/to_out.0</c></item>
/// <item><c>single_layers.{i}.mlp.{c_fc1,c_fc2,c_proj}.weight</c> → single FFN <c>ff.*</c></item>
/// <item><c>final_linear.weight</c> → <c>proj_out.weight</c></item>
/// <item><c>modF.1.weight</c> → <c>norm_out.linear.weight</c> with <see cref="SwapScaleShiftHalves"/> applied
/// (BFL native is <c>[shift, scale]</c>; diffusers' <c>AuraFlowPreFinalBlock</c> chunks as <c>[scale, shift]</c>).</item>
/// <item><c>positional_encoding</c> → <c>pos_embed.pos_embed</c></item>
/// <item><c>register_tokens</c> → <c>register_tokens</c> (passthrough)</item>
/// <item><c>cond_seq_linear.weight</c> → <c>context_embedder.weight</c></item>
/// <item><c>t_embedder.mlp.{0,2}.{weight,bias}</c> → <c>time_step_proj.linear_{1,2}.{weight,bias}</c></item>
/// </list>
///
/// AuraFlow's BFL single-file is bias-free for attention and FFN; only timestep MLP has biases.
/// Pile-T5-XL text encoder weights are shipped separately by <c>fal/AuraFlow-v0.3</c> in <c>text_encoder/</c>
/// (diffusers folder layout) — not included in the single-file safetensors. Same for the SDXL VAE.
/// </summary>
public sealed class AuraFlowCheckpointConverter
{
    /// <summary>Result of converting a single-file AuraFlow checkpoint. The single-file ships only the
    /// transformer; text encoder + VAE must be loaded separately from their own safetensors.</summary>
    public sealed class ConvertedWeights
    {
        /// <summary>AuraFlow MMDiT transformer weights in diffusers format. Loaded into <see cref="SharpInference.Diffusion.Models.Denoisers.AuraFlowTransformer.LoadWeights(IReadOnlyDictionary{string, Tensor})"/>.</summary>
        public required Dictionary<string, Tensor> Transformer { get; init; }
    }

    /// <summary>Converts a flat AuraFlow single-file weight dictionary to diffusers naming. Folds any
    /// ComfyUI fp8_scaled <c>.scale_weight</c> companions into <c>Tensor.Fp8ScaleFactor</c> first
    /// (some community repackagings ship fp8_scaled even for AuraFlow).</summary>
    public static ConvertedWeights Convert(Dictionary<string, Tensor> allWeights)
    {
        allWeights = CheckpointConvertUtils.ApplyFp8ScaledDequant(allWeights);

        Dictionary<string, Tensor> transformer = new(800);

        foreach (KeyValuePair<string, Tensor> kvp in allWeights)
        {
            string key = kvp.Key;
            Tensor tensor = kvp.Value;

            if (key.EndsWith(".scaled_fp8") || key == "scaled_fp8")
                continue;

            ConvertKey(key, tensor, transformer);
        }

        return new ConvertedWeights { Transformer = transformer };
    }

    /// <summary>Loads from disk and converts in one shot. Returns the converted weights plus the loader
    /// (the caller is responsible for disposing the loader once weights are no longer needed).</summary>
    public static (ConvertedWeights weights, SafeTensorsLoader loader) LoadAndConvert(string checkpointPath)
    {
        SafeTensorsLoader loader = new();
        Dictionary<string, Tensor> raw = loader.Load(checkpointPath);
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
        if (key == "final_linear.weight")
        {
            output["proj_out.weight"] = tensor;
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
