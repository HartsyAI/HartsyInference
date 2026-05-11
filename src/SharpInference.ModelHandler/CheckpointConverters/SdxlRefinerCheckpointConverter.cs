using SharpInference.Core.Tensors;
using SharpInference.ModelHandler.CheckpointConverters.Utils;
using SharpInference.ModelHandler.SafeTensors;

namespace SharpInference.ModelHandler.CheckpointConverters;

/// <summary>Converts single-file SDXL refiner checkpoints (LDM/CompVis format) to diffusers-format weight dictionaries. The refiner UNet has a 4-level layout (vs base's 3), uniform transformer-depth 4, and only CLIP-G text conditioning. The VAE is identical to the base, so VAE conversion is delegated to <see cref="CheckpointConvertUtils.ConvertVaeKey"/>.</summary>
public sealed class SdxlRefinerCheckpointConverter
{
    /// <summary>Result of converting an SDXL refiner checkpoint.</summary>
    public sealed class ConvertedWeights
    {
        /// <summary>Refiner UNet weights in diffusers format.</summary>
        public required Dictionary<string, Tensor> UNet { get; init; }

        /// <summary>CLIP-G text encoder weights (the refiner's only text encoder).</summary>
        public required Dictionary<string, Tensor> ClipG { get; init; }

        /// <summary>VAE weights in diffusers format. Identical structure to the base SDXL VAE.</summary>
        public required Dictionary<string, Tensor> Vae { get; init; }
    }

    // Refiner: 4 levels [384, 768, 1536, 1536]
    // DownBlockHasAttention = [false, true, true, false]
    // Input blocks flat numbering (12 total):
    //   0: conv_in
    //   1, 2: level 0 resnets (no attention)
    //   3: level 0 downsample
    //   4, 5: level 1 resnets + attention
    //   6: level 1 downsample
    //   7, 8: level 2 resnets + attention
    //   9: level 2 downsample
    //   10, 11: level 3 resnets (no attention)
    private static readonly int[] InputBlockToLevel = [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3];
    private static readonly int[] InputBlockToResnetIdx = [0, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1];
    private static readonly bool[] InputBlockIsDownsample = [false, false, false, true, false, false, true, false, false, true, false, false];

    // Output blocks (12 total): 4 levels × 3 resnets
    // UpBlockHasAttention = [false, true, true, false]
    private static readonly int[] OutputBlockToUpLevel = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3];
    private static readonly int[] OutputBlockToResnetIdx = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2];
    private static readonly bool[] OutputBlockHasUpsample = [false, false, true, false, false, true, false, false, true, false, false, false];
    private static readonly bool[] OutputBlockHasAttention = [false, false, false, true, true, true, true, true, true, false, false, false];

    /// <summary>Converts a single-file SDXL refiner checkpoint into per-component weight dictionaries.</summary>
    public static ConvertedWeights Convert(Dictionary<string, Tensor> allWeights)
    {
        Dictionary<string, Tensor> unet = new(2200);
        Dictionary<string, Tensor> clipG = new(400);
        Dictionary<string, Tensor> vae = new(250);

        foreach (KeyValuePair<string, Tensor> kvp in allWeights)
        {
            string key = kvp.Key;
            Tensor tensor = kvp.Value;

            if (key.StartsWith("model.diffusion_model."))
            {
                string ldmKey = key["model.diffusion_model.".Length..];
                string? diffusersKey = ConvertUNetKey(ldmKey);
                if (diffusersKey is not null)
                    unet[diffusersKey] = tensor;
            }
            else if (key.StartsWith("conditioner.embedders.0."))
            {
                // Refiner's only text encoder is CLIP-G (at index 0 in the conditioner list, unlike base where 0=CLIP-L, 1=CLIP-G).
                ConvertClipGKey(key, tensor, clipG, "conditioner.embedders.0.model.");
            }
            else if (key.StartsWith("first_stage_model."))
            {
                string ldmKey = key["first_stage_model.".Length..];
                string? diffusersKey = CheckpointConvertUtils.ConvertVaeKey(ldmKey);
                if (diffusersKey is not null)
                    vae[diffusersKey] = tensor;
            }
        }

        return new ConvertedWeights { UNet = unet, ClipG = clipG, Vae = vae };
    }

    /// <summary>Loads a refiner single-file checkpoint and converts it in one step.</summary>
    public static (ConvertedWeights weights, SafeTensorsLoader loader) LoadAndConvert(string checkpointPath)
    {
        SafeTensorsLoader loader = new();
        loader.Load(checkpointPath);
        ConvertedWeights converted = Convert(loader.GetAllTensors());
        return (converted, loader);
    }

    // ── UNet Key Conversion ─────────────────────────────────────────────

    private static string? ConvertUNetKey(string ldmKey)
    {
        if (ldmKey.StartsWith("input_blocks.0.0."))
            return "conv_in." + ldmKey["input_blocks.0.0.".Length..];

        if (ldmKey.StartsWith("time_embed."))
            return CheckpointConvertUtils.ConvertTimeEmbedKey(ldmKey);

        if (ldmKey.StartsWith("label_emb."))
            return ConvertLabelEmbKey(ldmKey);

        if (ldmKey.StartsWith("out."))
            return CheckpointConvertUtils.ConvertOutKey(ldmKey);

        if (ldmKey.StartsWith("input_blocks."))
            return ConvertInputBlockKey(ldmKey);

        if (ldmKey.StartsWith("middle_block."))
            return CheckpointConvertUtils.ConvertMiddleBlockKey(ldmKey);

        if (ldmKey.StartsWith("output_blocks."))
            return ConvertOutputBlockKey(ldmKey);

        return null;
    }

    private static string ConvertLabelEmbKey(string ldmKey)
    {
        // Same shape as base SDXL: label_emb.0.{0,2} → add_embedding.linear_{1,2}
        string rest = ldmKey["label_emb.0.".Length..];
        if (rest.StartsWith("0."))
            return "add_embedding.linear_1." + rest[2..];
        if (rest.StartsWith("2."))
            return "add_embedding.linear_2." + rest[2..];
        return "add_embedding." + rest;
    }

    private static string? ConvertInputBlockKey(string ldmKey)
    {
        string afterPrefix = ldmKey["input_blocks.".Length..];
        int firstDot = afterPrefix.IndexOf('.');
        if (firstDot < 0) return null;

        int blockIdx = int.Parse(afterPrefix[..firstDot]);
        if (blockIdx < 0 || blockIdx >= InputBlockToLevel.Length) return null;
        string afterBlockIdx = afterPrefix[(firstDot + 1)..];

        if (blockIdx == 0) return "conv_in." + afterBlockIdx;

        int level = InputBlockToLevel[blockIdx];

        if (InputBlockIsDownsample[blockIdx])
        {
            if (afterBlockIdx.StartsWith("0.op."))
                return $"down_blocks.{level}.downsamplers.0.conv." + afterBlockIdx["0.op.".Length..];
            return null;
        }

        int subDot = afterBlockIdx.IndexOf('.');
        if (subDot < 0) return null;
        int subIdx = int.Parse(afterBlockIdx[..subDot]);
        string rest = afterBlockIdx[(subDot + 1)..];
        int resnetIdx = InputBlockToResnetIdx[blockIdx];

        if (subIdx == 0)
            return $"down_blocks.{level}.resnets.{resnetIdx}." + CheckpointConvertUtils.ConvertResNetSubKey(rest);
        if (subIdx == 1)
            return $"down_blocks.{level}.attentions.{resnetIdx}." + rest;

        return null;
    }

    private static string? ConvertOutputBlockKey(string ldmKey)
    {
        string afterPrefix = ldmKey["output_blocks.".Length..];
        int firstDot = afterPrefix.IndexOf('.');
        if (firstDot < 0) return null;

        int blockIdx = int.Parse(afterPrefix[..firstDot]);
        if (blockIdx < 0 || blockIdx >= OutputBlockToUpLevel.Length) return null;
        string afterBlockIdx = afterPrefix[(firstDot + 1)..];

        int upLevel = OutputBlockToUpLevel[blockIdx];
        int resnetIdx = OutputBlockToResnetIdx[blockIdx];

        int subDot = afterBlockIdx.IndexOf('.');
        if (subDot < 0) return null;
        int subIdx = int.Parse(afterBlockIdx[..subDot]);
        string rest = afterBlockIdx[(subDot + 1)..];

        if (subIdx == 0)
            return $"up_blocks.{upLevel}.resnets.{resnetIdx}." + CheckpointConvertUtils.ConvertResNetSubKey(rest);
        if (subIdx == 1)
        {
            if (OutputBlockHasAttention[blockIdx])
                return $"up_blocks.{upLevel}.attentions.{resnetIdx}." + rest;
            // No-attention level with subIdx=1 → upsample variant
            return $"up_blocks.{upLevel}.upsamplers.0.conv." + rest;
        }
        if (subIdx == 2)
        {
            if (rest.StartsWith("conv."))
                return $"up_blocks.{upLevel}.upsamplers.0.conv." + rest["conv.".Length..];
            return $"up_blocks.{upLevel}.upsamplers.0.conv." + rest;
        }

        return null;
    }

    // ── CLIP-G Key Conversion ───────────────────────────────────────────
    // Same conversion logic as the SDXL base CLIP-G, but the refiner has CLIP-G at conditioner index 0 (not 1).
    // Factored from SdxlCheckpointConverter so both paths share the same OpenCLIP→HF mapping.

    private static unsafe void ConvertClipGKey(string key, Tensor tensor, Dictionary<string, Tensor> clipG, string modelPrefix)
    {
        if (!key.StartsWith(modelPrefix)) return;
        string rest = key[modelPrefix.Length..];

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
            ConvertClipGResblock(rest["transformer.resblocks.".Length..], tensor, clipG);
    }

    private static unsafe void ConvertClipGResblock(string rest, Tensor tensor, Dictionary<string, Tensor> clipG)
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
}
