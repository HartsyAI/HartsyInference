using SharpInference.Core.Tensors;
using SharpInference.ModelHandler.CheckpointConverters.Utils;
using SharpInference.ModelHandler.SafeTensors;

namespace SharpInference.ModelHandler.CheckpointConverters;

/// <summary>Converts single-file SD3 checkpoints (Stability AI / ComfyUI format) to diffusers-format weight dictionaries. Handles fused QKV splitting and key remapping for the MMDiT transformer, triple text encoders, and 16-channel VAE.</summary>
public sealed class Sd3CheckpointConverter
{
    /// <summary>Result of converting a single-file SD3 checkpoint into per-component weight dictionaries.</summary>
    public sealed class ConvertedWeights
    {
        /// <summary>MMDiT transformer weights in diffusers format.</summary>
        public required Dictionary<string, Tensor> Transformer { get; init; }

        /// <summary>CLIP-L text encoder weights.</summary>
        public required Dictionary<string, Tensor> ClipL { get; init; }

        /// <summary>CLIP-G text encoder weights (with fused QKV split into q/k/v).</summary>
        public required Dictionary<string, Tensor> ClipG { get; init; }

        /// <summary>T5-XXL text encoder weights. Empty if T5 was not in the checkpoint.</summary>
        public required Dictionary<string, Tensor> T5 { get; init; }

        /// <summary>VAE weights in diffusers format.</summary>
        public required Dictionary<string, Tensor> Vae { get; init; }
    }

    /// <summary>Converts a single-file SD3 checkpoint into separate per-component weight dictionaries. Folds ComfyUI fp8_scaled per-tensor `.scale_weight` companions into <c>Tensor.Fp8ScaleFactor</c> before bucketing (SD3.5 FP8 distributions ship the T5-XXL encoder this way), and drops zero-byte `*.scaled_fp8` format markers.</summary>
    public static ConvertedWeights Convert(Dictionary<string, Tensor> allWeights)
    {
        // Fold per-tensor FP8 scale companions into Tensor.Fp8ScaleFactor (matches Flux fp8_scaled handling).
        allWeights = CheckpointConvertUtils.ApplyFp8ScaledDequant(allWeights);

        Dictionary<string, Tensor> transformer = new(2000);
        Dictionary<string, Tensor> clipL = new(200);
        Dictionary<string, Tensor> clipG = new(400);
        Dictionary<string, Tensor> t5 = new(800);
        Dictionary<string, Tensor> vae = new(250);

        foreach (KeyValuePair<string, Tensor> kvp in allWeights)
        {
            string key = kvp.Key;
            Tensor tensor = kvp.Value;

            // Skip ComfyUI fp8_scaled format markers (zero-byte sentinels indicating the file uses fp8_scaled)
            if (key.EndsWith(".scaled_fp8") || key == "scaled_fp8")
                continue;

            // MMDiT transformer (Stability format)
            if (key.StartsWith("model.diffusion_model."))
            {
                string ldmKey = key["model.diffusion_model.".Length..];
                ConvertTransformerKey(ldmKey, tensor, transformer);
            }
            // CLIP-L (Stability format — conditioner.embedders.0)
            else if (key.StartsWith("conditioner.embedders.0."))
            {
                ConvertClipLStability(key, tensor, clipL);
            }
            // CLIP-G (Stability format — conditioner.embedders.1)
            else if (key.StartsWith("conditioner.embedders.1."))
            {
                ConvertClipGStability(key, tensor, clipG);
            }
            // CLIP-L (ComfyUI format — text_encoders.clip_l)
            else if (key.StartsWith("text_encoders.clip_l."))
            {
                ConvertClipLComfy(key, tensor, clipL);
            }
            // CLIP-G (ComfyUI format — text_encoders.clip_g)
            else if (key.StartsWith("text_encoders.clip_g."))
            {
                ConvertClipGComfy(key, tensor, clipG);
            }
            // T5-XXL (ComfyUI format — text_encoders.t5xxl)
            else if (key.StartsWith("text_encoders.t5xxl."))
            {
                ConvertT5(key, tensor, t5);
            }
            // VAE
            else if (key.StartsWith("first_stage_model."))
            {
                string ldmKey = key["first_stage_model.".Length..];
                string? diffusersKey = CheckpointConvertUtils.ConvertVaeKey(ldmKey);
                if (diffusersKey is not null)
                    vae[diffusersKey] = tensor;
            }
        }

        return new ConvertedWeights
        {
            Transformer = transformer,
            ClipL = clipL,
            ClipG = clipG,
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

    // ── MMDiT Transformer Key Conversion ──────────────────────────────────────────

    private static void ConvertTransformerKey(string ldmKey, Tensor tensor, Dictionary<string, Tensor> output)
    {
        // Patch embedding
        if (ldmKey.StartsWith("x_embedder.proj."))
        {
            output["pos_embed.proj." + ldmKey["x_embedder.proj.".Length..]] = tensor;
            return;
        }
        if (ldmKey == "pos_embed")
        {
            output["pos_embed.pos_embed"] = tensor;
            return;
        }

        // Context embedder (key unchanged)
        if (ldmKey.StartsWith("context_embedder."))
        {
            output[ldmKey] = tensor;
            return;
        }

        // Timestep embedder
        if (ldmKey.StartsWith("t_embedder.mlp."))
        {
            string rest = ldmKey["t_embedder.mlp.".Length..];
            if (rest.StartsWith("0."))
                output["time_text_embed.timestep_embedder.linear_1." + rest[2..]] = tensor;
            else if (rest.StartsWith("2."))
                output["time_text_embed.timestep_embedder.linear_2." + rest[2..]] = tensor;
            return;
        }

        // Pooled text embedder (y_embedder)
        if (ldmKey.StartsWith("y_embedder.mlp."))
        {
            string rest = ldmKey["y_embedder.mlp.".Length..];
            if (rest.StartsWith("0."))
                output["time_text_embed.text_embedder.linear_1." + rest[2..]] = tensor;
            else if (rest.StartsWith("2."))
                output["time_text_embed.text_embedder.linear_2." + rest[2..]] = tensor;
            return;
        }

        // Final layer
        if (ldmKey.StartsWith("final_layer."))
        {
            ConvertFinalLayerKey(ldmKey["final_layer.".Length..], tensor, output);
            return;
        }

        // Joint transformer blocks
        if (ldmKey.StartsWith("joint_blocks."))
        {
            ConvertJointBlockKey(ldmKey["joint_blocks.".Length..], tensor, output);
            return;
        }
    }

    private static void ConvertFinalLayerKey(string rest, Tensor tensor, Dictionary<string, Tensor> output)
    {
        if (rest.StartsWith("adaLN_modulation.1."))
        {
            output["norm_out.linear." + rest["adaLN_modulation.1.".Length..]] = tensor;
            return;
        }
        if (rest.StartsWith("linear."))
        {
            output["proj_out." + rest["linear.".Length..]] = tensor;
            return;
        }
    }

    private static void ConvertJointBlockKey(string rest, Tensor tensor, Dictionary<string, Tensor> output)
    {
        // Parse block index: "{i}.x_block.*" or "{i}.context_block.*"
        int firstDot = rest.IndexOf('.');
        if (firstDot < 0) return;
        string blockIdxStr = rest[..firstDot];
        string afterBlockIdx = rest[(firstDot + 1)..];
        string prefix = $"transformer_blocks.{blockIdxStr}";

        if (afterBlockIdx.StartsWith("x_block."))
        {
            ConvertImageBlockKey(prefix, afterBlockIdx["x_block.".Length..], tensor, output);
            return;
        }
        if (afterBlockIdx.StartsWith("context_block."))
        {
            ConvertContextBlockKey(prefix, afterBlockIdx["context_block.".Length..], tensor, output);
            return;
        }
    }

    private static void ConvertImageBlockKey(string prefix, string rest, Tensor tensor, Dictionary<string, Tensor> output)
    {
        // AdaLN modulation (6 params for SD3, 9 for SD3.5 dual-attention layers — output dim differs but key path is the same)
        if (rest.StartsWith("adaLN_modulation.1."))
        {
            output[$"{prefix}.norm1.linear.{rest["adaLN_modulation.1.".Length..]}"] = tensor;
            return;
        }

        // Joint attention (`attn`)
        if (rest.StartsWith("attn."))
        {
            string attnKey = rest["attn.".Length..];

            if (attnKey == "qkv.weight")
            {
                int innerDim = (int)tensor.Shape[0] / 3;
                SplitQkvWeight(tensor, innerDim, prefix, "attn.to_q", "attn.to_k", "attn.to_v", output);
                return;
            }
            if (attnKey == "qkv.bias")
            {
                int innerDim = (int)tensor.Shape[0] / 3;
                SplitQkvBias(tensor, innerDim, prefix, "attn.to_q", "attn.to_k", "attn.to_v", output);
                return;
            }
            if (attnKey.StartsWith("proj."))
            {
                output[$"{prefix}.attn.to_out.0.{attnKey["proj.".Length..]}"] = tensor;
                return;
            }
            if (attnKey == "ln_q.weight")
            {
                output[$"{prefix}.attn.norm_q.weight"] = tensor;
                return;
            }
            if (attnKey == "ln_k.weight")
            {
                output[$"{prefix}.attn.norm_k.weight"] = tensor;
                return;
            }
            return;
        }

        // SD3.5 MMDiT-X dual self-attention (`attn2`) — image-only second attention
        if (rest.StartsWith("attn2."))
        {
            string attnKey = rest["attn2.".Length..];

            if (attnKey == "qkv.weight")
            {
                int innerDim = (int)tensor.Shape[0] / 3;
                SplitQkvWeight(tensor, innerDim, prefix, "attn2.to_q", "attn2.to_k", "attn2.to_v", output);
                return;
            }
            if (attnKey == "qkv.bias")
            {
                int innerDim = (int)tensor.Shape[0] / 3;
                SplitQkvBias(tensor, innerDim, prefix, "attn2.to_q", "attn2.to_k", "attn2.to_v", output);
                return;
            }
            if (attnKey.StartsWith("proj."))
            {
                output[$"{prefix}.attn2.to_out.0.{attnKey["proj.".Length..]}"] = tensor;
                return;
            }
            if (attnKey == "ln_q.weight")
            {
                output[$"{prefix}.attn2.norm_q.weight"] = tensor;
                return;
            }
            if (attnKey == "ln_k.weight")
            {
                output[$"{prefix}.attn2.norm_k.weight"] = tensor;
                return;
            }
            return;
        }

        // MLP
        if (rest.StartsWith("mlp.fc1."))
        {
            output[$"{prefix}.ff.net.0.proj.{rest["mlp.fc1.".Length..]}"] = tensor;
            return;
        }
        if (rest.StartsWith("mlp.fc2."))
        {
            output[$"{prefix}.ff.net.2.{rest["mlp.fc2.".Length..]}"] = tensor;
            return;
        }
    }

    private static void ConvertContextBlockKey(string prefix, string rest, Tensor tensor, Dictionary<string, Tensor> output)
    {
        // AdaLN modulation
        if (rest.StartsWith("adaLN_modulation.1."))
        {
            output[$"{prefix}.norm1_context.linear.{rest["adaLN_modulation.1.".Length..]}"] = tensor;
            return;
        }

        // Attention
        if (rest.StartsWith("attn."))
        {
            string attnKey = rest["attn.".Length..];

            // Fused QKV → split into separate add_q, add_k, add_v
            if (attnKey == "qkv.weight")
            {
                int innerDim = (int)tensor.Shape[0] / 3;
                SplitQkvWeight(tensor, innerDim, prefix, "attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj", output);
                return;
            }
            if (attnKey == "qkv.bias")
            {
                int innerDim = (int)tensor.Shape[0] / 3;
                SplitQkvBias(tensor, innerDim, prefix, "attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj", output);
                return;
            }

            // Output projection
            if (attnKey.StartsWith("proj."))
            {
                output[$"{prefix}.attn.to_add_out.{attnKey["proj.".Length..]}"] = tensor;
                return;
            }

            // QK-norm
            if (attnKey == "ln_q.weight")
            {
                output[$"{prefix}.attn.norm_added_q.weight"] = tensor;
                return;
            }
            if (attnKey == "ln_k.weight")
            {
                output[$"{prefix}.attn.norm_added_k.weight"] = tensor;
                return;
            }

            return;
        }

        // MLP
        if (rest.StartsWith("mlp.fc1."))
        {
            output[$"{prefix}.ff_context.net.0.proj.{rest["mlp.fc1.".Length..]}"] = tensor;
            return;
        }
        if (rest.StartsWith("mlp.fc2."))
        {
            output[$"{prefix}.ff_context.net.2.{rest["mlp.fc2.".Length..]}"] = tensor;
            return;
        }
    }


    // ── QKV Splitting ──────────────────────────────────────────

    /// <summary>Splits a fused QKV weight [3*innerDim, inDim] into three separate [innerDim, inDim] weights.</summary>
    private static unsafe void SplitQkvWeight(Tensor fused, int innerDim, string prefix,
        string qName, string kName, string vName, Dictionary<string, Tensor> output)
    {
        int inDim = (int)fused.Shape[1];
        long rowBytes = (long)inDim * fused.DType.SizeInBytes;
        long chunkBytes = (long)innerDim * rowBytes;
        TensorShape splitShape = new TensorShape(innerDim, inDim);

        Tensor qWeight = new Tensor(splitShape, fused.DType);
        Tensor kWeight = new Tensor(splitShape, fused.DType);
        Tensor vWeight = new Tensor(splitShape, fused.DType);

        byte* src = (byte*)fused.DataPointer;
        Buffer.MemoryCopy(src, (void*)qWeight.DataPointer, chunkBytes, chunkBytes);
        Buffer.MemoryCopy(src + chunkBytes, (void*)kWeight.DataPointer, chunkBytes, chunkBytes);
        Buffer.MemoryCopy(src + 2 * chunkBytes, (void*)vWeight.DataPointer, chunkBytes, chunkBytes);

        output[$"{prefix}.{qName}.weight"] = qWeight;
        output[$"{prefix}.{kName}.weight"] = kWeight;
        output[$"{prefix}.{vName}.weight"] = vWeight;
    }

    /// <summary>Splits a fused QKV bias [3*innerDim] into three separate [innerDim] biases.</summary>
    private static unsafe void SplitQkvBias(Tensor fused, int innerDim, string prefix,
        string qName, string kName, string vName, Dictionary<string, Tensor> output)
    {
        long elemBytes = fused.DType.SizeInBytes;
        long chunkBytes = (long)innerDim * elemBytes;
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


    // ── CLIP-L Key Conversion ──────────────────────────────────────────

    private static void ConvertClipLStability(string key, Tensor tensor, Dictionary<string, Tensor> clipL)
    {
        // conditioner.embedders.0.transformer.text_model.* → text_model.*
        string prefix = "conditioner.embedders.0.transformer.";
        if (!key.StartsWith(prefix)) return;
        string rest = key[prefix.Length..];
        if (rest.EndsWith("position_ids")) return;
        clipL[rest] = tensor;
    }

    private static void ConvertClipLComfy(string key, Tensor tensor, Dictionary<string, Tensor> clipL)
    {
        // text_encoders.clip_l.transformer.text_model.* → text_model.*
        string prefix = "text_encoders.clip_l.transformer.";
        if (!key.StartsWith(prefix)) return;
        string rest = key[prefix.Length..];
        if (rest.EndsWith("position_ids")) return;
        clipL[rest] = tensor;
    }


    // ── CLIP-G Key Conversion ──────────────────────────────────────────

    private static void ConvertClipGStability(string key, Tensor tensor, Dictionary<string, Tensor> clipG)
    {
        // conditioner.embedders.1.model.* → OpenCLIP → diffusers format
        // Same conversion as SDXL CLIP-G (OpenCLIP format)
        string modelPrefix = "conditioner.embedders.1.model.";
        if (!key.StartsWith(modelPrefix)) return;
        string rest = key[modelPrefix.Length..];
        ConvertOpenClipToHf(rest, tensor, clipG);
    }

    private static void ConvertClipGComfy(string key, Tensor tensor, Dictionary<string, Tensor> clipG)
    {
        // text_encoders.clip_g.transformer.text_model.* → text_model.*
        string prefix = "text_encoders.clip_g.transformer.";
        if (!key.StartsWith(prefix)) return;
        string rest = key[prefix.Length..];
        if (rest.EndsWith("position_ids")) return;
        clipG[rest] = tensor;
    }

    /// <summary>Converts OpenCLIP weight keys to HuggingFace diffusers format. Handles fused in_proj splitting.</summary>
    private static void ConvertOpenClipToHf(string rest, Tensor tensor, Dictionary<string, Tensor> clipG)
    {
        if (rest == "token_embedding.weight")
        {
            clipG["text_model.embeddings.token_embedding.weight"] = tensor;
            return;
        }
        if (rest == "positional_embedding")
        {
            clipG["text_model.embeddings.position_embedding.weight"] = tensor;
            return;
        }
        if (rest.StartsWith("ln_final."))
        {
            clipG[$"text_model.final_layer_norm.{rest["ln_final.".Length..]}"] = tensor;
            return;
        }
        if (rest == "text_projection")
        {
            clipG["text_projection.weight"] = tensor;
            return;
        }
        if (rest == "logit_scale") return;

        if (rest.StartsWith("transformer.resblocks."))
        {
            ConvertOpenClipResblock(rest["transformer.resblocks.".Length..], tensor, clipG);
        }
    }

    private static void ConvertOpenClipResblock(string rest, Tensor tensor, Dictionary<string, Tensor> clipG)
    {
        int firstDot = rest.IndexOf('.');
        if (firstDot < 0) return;
        string layerIdxStr = rest[..firstDot];
        string subKey = rest[(firstDot + 1)..];
        string layerPrefix = $"text_model.encoder.layers.{layerIdxStr}";

        if (subKey.StartsWith("ln_1."))
        {
            clipG[$"{layerPrefix}.layer_norm1.{subKey["ln_1.".Length..]}"] = tensor;
            return;
        }
        if (subKey.StartsWith("ln_2."))
        {
            clipG[$"{layerPrefix}.layer_norm2.{subKey["ln_2.".Length..]}"] = tensor;
            return;
        }
        if (subKey.StartsWith("mlp.c_fc."))
        {
            clipG[$"{layerPrefix}.mlp.fc1.{subKey["mlp.c_fc.".Length..]}"] = tensor;
            return;
        }
        if (subKey.StartsWith("mlp.c_proj."))
        {
            clipG[$"{layerPrefix}.mlp.fc2.{subKey["mlp.c_proj.".Length..]}"] = tensor;
            return;
        }
        if (subKey.StartsWith("attn.out_proj."))
        {
            clipG[$"{layerPrefix}.self_attn.out_proj.{subKey["attn.out_proj.".Length..]}"] = tensor;
            return;
        }
        if (subKey == "attn.in_proj_weight")
        {
            int hiddenSize = (int)tensor.Shape[0] / 3;
            CheckpointConvertUtils.SplitInProjWeight(tensor, hiddenSize, layerPrefix, clipG);
            return;
        }
        if (subKey == "attn.in_proj_bias")
        {
            int hiddenSize = (int)tensor.Shape[0] / 3;
            CheckpointConvertUtils.SplitInProjBias(tensor, hiddenSize, layerPrefix, clipG);
            return;
        }
    }


    // ── T5 Key Conversion ──────────────────────────────────────────

    private static void ConvertT5(string key, Tensor tensor, Dictionary<string, Tensor> t5)
    {
        // text_encoders.t5xxl.transformer.* → *
        string prefix = "text_encoders.t5xxl.transformer.";
        if (!key.StartsWith(prefix)) return;
        string rest = key[prefix.Length..];
        t5[rest] = tensor;
    }


    /// <summary>Auto-detects model depth from the transformer weights by counting joint_blocks.</summary>
    public static int DetectDepth(Dictionary<string, Tensor> transformerWeights)
    {
        int maxBlock = -1;
        foreach (string key in transformerWeights.Keys)
        {
            if (!key.StartsWith("transformer_blocks.")) continue;
            string afterPrefix = key["transformer_blocks.".Length..];
            int dot = afterPrefix.IndexOf('.');
            if (dot < 0) continue;
            if (int.TryParse(afterPrefix[..dot], out int blockIdx) && blockIdx > maxBlock)
                maxBlock = blockIdx;
        }
        return maxBlock + 1;
    }

    /// <summary>Detects which transformer block indices contain SD3.5 MMDiT-X dual-attention (`attn2`) weights. Returns an empty array for plain SD3.</summary>
    public static int[] DetectDualAttentionLayers(Dictionary<string, Tensor> transformerWeights)
    {
        SortedSet<int> dualBlocks = new();
        foreach (string key in transformerWeights.Keys)
        {
            if (!key.StartsWith("transformer_blocks.")) continue;
            string afterPrefix = key["transformer_blocks.".Length..];
            int dot = afterPrefix.IndexOf('.');
            if (dot < 0) continue;
            if (!afterPrefix.AsSpan(dot + 1).StartsWith("attn2.")) continue;
            if (int.TryParse(afterPrefix[..dot], out int blockIdx))
                dualBlocks.Add(blockIdx);
        }
        return dualBlocks.ToArray();
    }

    /// <summary>True if the converted transformer weights expose SD3.5-style QK-norm tensors.</summary>
    public static bool DetectQkNorm(Dictionary<string, Tensor> transformerWeights)
    {
        foreach (string key in transformerWeights.Keys)
        {
            if (key.EndsWith(".attn.norm_q.weight") || key.EndsWith(".attn.norm_k.weight"))
                return true;
        }
        return false;
    }

}
