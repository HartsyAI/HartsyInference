using HartsyInference.Core.Tensors;
using HartsyInference.ModelHandler.CheckpointConverters.Utils;
using HartsyInference.ModelHandler.SafeTensors;

namespace HartsyInference.ModelHandler.CheckpointConverters;

/// <summary>Converts single-file SDXL checkpoints (LDM/CompVis format) to diffusers-format weight dictionaries that can be loaded by HartsyInference model components.</summary>
public sealed class SdxlCheckpointConverter
{
    /// <summary>Result of converting a single-file SDXL checkpoint into per-component weight dictionaries.</summary>
    public sealed class ConvertedWeights
    {
        /// <summary>UNet weights in diffusers format.</summary>
        public required Dictionary<string, Tensor> UNet { get; init; }

        /// <summary>CLIP-L text encoder weights.</summary>
        public required Dictionary<string, Tensor> ClipL { get; init; }

        /// <summary>CLIP-G text encoder weights (with in_proj split into q/k/v).</summary>
        public required Dictionary<string, Tensor> ClipG { get; init; }

        /// <summary>VAE weights in diffusers format.</summary>
        public required Dictionary<string, Tensor> Vae { get; init; }
    }

    // SDXL: 3 levels [320, 640, 1280], DownBlockHasAttention = [false, true, true]
    // Input blocks flat numbering:
    //   0: conv_in
    //   1,2: level 0 resnets (no attention)
    //   3: level 0 downsample
    //   4,5: level 1 resnets + attention
    //   6: level 1 downsample
    //   7,8: level 2 resnets + attention
    private static readonly int[] _inputBlockToLevel = [0, 0, 0, 0, 1, 1, 1, 2, 2];
    private static readonly int[] _inputBlockToResnetIdx = [0, 0, 1, -1, 0, 1, -1, 0, 1];
    private static readonly bool[] _inputBlockIsDownsample = [false, false, false, true, false, false, true, false, false];

    // Output blocks: 3 levels × 3 resnets = 9 total
    private static readonly int[] _outputBlockToUpLevel = [0, 0, 0, 1, 1, 1, 2, 2, 2];
    private static readonly int[] _outputBlockToResnetIdx = [0, 1, 2, 0, 1, 2, 0, 1, 2];
    private static readonly bool[] _outputBlockHasUpsample = [false, false, true, false, false, true, false, false, false];
    private static readonly bool[] _outputBlockHasAttention = [true, true, true, true, true, true, false, false, false];

    /// <summary>Converts a single-file SDXL checkpoint into separate per-component weight dictionaries.</summary>
    public static ConvertedWeights Convert(Dictionary<string, Tensor> allWeights)
    {
        Dictionary<string, Tensor> unet = new(1700);
        Dictionary<string, Tensor> clipL = new(200);
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
                ConvertClipLKey(key, tensor, clipL);
            }
            else if (key.StartsWith("conditioner.embedders.1."))
            {
                ConvertClipGKey(key, tensor, clipG);
            }
            else if (key.StartsWith("first_stage_model."))
            {
                string ldmKey = key["first_stage_model.".Length..];
                string? diffusersKey = CheckpointConvertUtils.ConvertVaeKey(ldmKey);
                if (diffusersKey is not null)
                    vae[diffusersKey] = tensor;
            }
        }

        return new ConvertedWeights { UNet = unet, ClipL = clipL, ClipG = clipG, Vae = vae };
    }

    /// <summary>Loads a single-file checkpoint and converts it in one step.</summary>
    public static (ConvertedWeights weights, SafeTensorsLoader loader) LoadAndConvert(string checkpointPath)
    {
        SafeTensorsLoader loader = new();
        loader.Load(checkpointPath);
        ConvertedWeights converted = Convert(loader.GetAllTensors());
        return (converted, loader);
    }

    // ── UNet Key Conversion ──────────────────────────────────────────

    /// <summary>Converts one LDM UNet key (after stripping the <c>model.diffusion_model.</c> prefix) to its diffusers name, or null when the key has no diffusers counterpart. Also used by <see cref="ControlNetCheckpointConverter"/> for the encoder half of SDXL ControlNets.</summary>
    public static string? ConvertUNetKey(string ldmKey)
    {
        if (ldmKey.StartsWith("input_blocks.0.0."))
            return "conv_in." + ldmKey["input_blocks.0.0.".Length..];

        if (ldmKey.StartsWith("time_embed."))
            return CheckpointConvertUtils.ConvertTimeEmbedKey(ldmKey);

        // label_emb.0.0 → add_embedding.linear_1, label_emb.0.2 → add_embedding.linear_2
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
        string afterBlockIdx = afterPrefix[(firstDot + 1)..];

        if (blockIdx == 0) return "conv_in." + afterBlockIdx;

        int level = _inputBlockToLevel[blockIdx];

        if (_inputBlockIsDownsample[blockIdx])
        {
            if (afterBlockIdx.StartsWith("0.op."))
                return $"down_blocks.{level}.downsamplers.0.conv." + afterBlockIdx["0.op.".Length..];
            return null;
        }

        int subDot = afterBlockIdx.IndexOf('.');
        if (subDot < 0) return null;
        int subIdx = int.Parse(afterBlockIdx[..subDot]);
        string rest = afterBlockIdx[(subDot + 1)..];
        int resnetIdx = _inputBlockToResnetIdx[blockIdx];

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
        string afterBlockIdx = afterPrefix[(firstDot + 1)..];

        int upLevel = _outputBlockToUpLevel[blockIdx];
        int resnetIdx = _outputBlockToResnetIdx[blockIdx];

        int subDot = afterBlockIdx.IndexOf('.');
        if (subDot < 0) return null;
        int subIdx = int.Parse(afterBlockIdx[..subDot]);
        string rest = afterBlockIdx[(subDot + 1)..];

        if (subIdx == 0)
            return $"up_blocks.{upLevel}.resnets.{resnetIdx}." + CheckpointConvertUtils.ConvertResNetSubKey(rest);
        if (subIdx == 1)
        {
            if (_outputBlockHasAttention[blockIdx])
                return $"up_blocks.{upLevel}.attentions.{resnetIdx}." + rest;
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


    // ── CLIP-L Key Conversion ──────────────────────────────────────────

    private static void ConvertClipLKey(string key, Tensor tensor, Dictionary<string, Tensor> clipL)
    {
        // conditioner.embedders.0.transformer.text_model.* → text_model.*
        string prefix = "conditioner.embedders.0.transformer.";
        if (!key.StartsWith(prefix)) return;

        string rest = key[prefix.Length..];
        if (rest.EndsWith("position_ids")) return;

        clipL[rest] = tensor;
    }


    // ── CLIP-G Key Conversion ──────────────────────────────────────────

    private static void ConvertClipGKey(string key, Tensor tensor, Dictionary<string, Tensor> clipG)
    {
        // OpenCLIP format → HuggingFace diffusers format
        string modelPrefix = "conditioner.embedders.1.model.";
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

    private static void ConvertClipGResblock(string rest, Tensor tensor, Dictionary<string, Tensor> clipG)
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
