using SharpInference.Core.Tensors;
using SharpInference.ModelHandler.CheckpointConverters.Utils;
using SharpInference.ModelHandler.SafeTensors;

namespace SharpInference.ModelHandler.CheckpointConverters;

/// <summary>Converts single-file SD1.5 checkpoints (LDM/CompVis format) to diffusers-format weight dictionaries that can be loaded by SharpInference model components.</summary>
public sealed class Sd15CheckpointConverter
{
    /// <summary>Result of converting a single-file SD1.5 checkpoint into per-component weight dictionaries.</summary>
    public sealed class ConvertedWeights
    {
        /// <summary>UNet weights in diffusers format.</summary>
        public required Dictionary<string, Tensor> UNet { get; init; }

        /// <summary>CLIP-L text encoder weights (single encoder, no CLIP-G).</summary>
        public required Dictionary<string, Tensor> ClipL { get; init; }

        /// <summary>VAE weights in diffusers format.</summary>
        public required Dictionary<string, Tensor> Vae { get; init; }
    }

    // SD1.5: 4 levels [320, 640, 1280, 1280], DownBlockHasAttention = [true, true, true, false]
    // Input blocks flat numbering (12 total):
    //   0: conv_in
    //   1,2: level 0 resnets + attention (320ch)
    //   3: level 0 downsample
    //   4,5: level 1 resnets + attention (640ch)
    //   6: level 1 downsample
    //   7,8: level 2 resnets + attention (1280ch)
    //   9: level 2 downsample
    //   10,11: level 3 resnets, NO attention (1280ch)
    private static readonly int[] _inputBlockToLevel =       [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3];
    private static readonly int[] _inputBlockToResnetIdx =   [0, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1];
    private static readonly bool[] _inputBlockIsDownsample = [false, false, false, true, false, false, true, false, false, true, false, false];
    private static readonly bool[] _inputBlockHasAttention = [false, true, true, false, true, true, false, true, true, false, false, false];

    // Output blocks: 4 levels × 3 resnets = 12 total
    // Flat: 0-2 = up_blocks.0 (level 3, no attn), 3-5 = up_blocks.1 (attn), 6-8 = up_blocks.2 (attn), 9-11 = up_blocks.3 (attn)
    private static readonly int[] _outputBlockToUpLevel =    [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3];
    private static readonly int[] _outputBlockToResnetIdx =  [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2];
    // UpBlockHasAttention = [false, true, true, true]
    private static readonly bool[] _outputBlockHasAttention = [false, false, false, true, true, true, true, true, true, true, true, true];
    // Upsample at last block of each level except the final level (up_blocks.3)
    private static readonly bool[] _outputBlockHasUpsample =  [false, false, true, false, false, true, false, false, true, false, false, false];

    /// <summary>Converts a single-file SD1.5 checkpoint into separate per-component weight dictionaries.</summary>
    public static ConvertedWeights Convert(Dictionary<string, Tensor> allWeights)
    {
        Dictionary<string, Tensor> unet = new(700);
        Dictionary<string, Tensor> clipL = new(200);
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
            else if (key.StartsWith("cond_stage_model.transformer."))
            {
                ConvertClipLKey(key, tensor, clipL);
            }
            else if (key.StartsWith("first_stage_model."))
            {
                string ldmKey = key["first_stage_model.".Length..];
                string? diffusersKey = CheckpointConvertUtils.ConvertVaeKey(ldmKey);
                if (diffusersKey is not null)
                    vae[diffusersKey] = tensor;
            }
        }

        return new ConvertedWeights { UNet = unet, ClipL = clipL, Vae = vae };
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

    private static string? ConvertUNetKey(string ldmKey)
    {
        if (ldmKey.StartsWith("input_blocks.0.0."))
            return "conv_in." + ldmKey["input_blocks.0.0.".Length..];

        if (ldmKey.StartsWith("time_embed."))
            return CheckpointConvertUtils.ConvertTimeEmbedKey(ldmKey);

        // SD1.5 has no label_emb/add_embedding (no ADM conditioning)

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

    private static string? ConvertInputBlockKey(string ldmKey)
    {
        string afterPrefix = ldmKey["input_blocks.".Length..];
        int firstDot = afterPrefix.IndexOf('.');
        if (firstDot < 0) return null;

        int blockIdx = int.Parse(afterPrefix[..firstDot]);
        string afterBlockIdx = afterPrefix[(firstDot + 1)..];

        if (blockIdx == 0) return "conv_in." + afterBlockIdx;
        if (blockIdx >= _inputBlockToLevel.Length) return null;

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
        if (subIdx == 1 && _inputBlockHasAttention[blockIdx])
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

        if (blockIdx >= _outputBlockToUpLevel.Length) return null;

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
            // No attention at this level → this is an upsample
            if (rest.StartsWith("conv."))
                return $"up_blocks.{upLevel}.upsamplers.0.conv." + rest["conv.".Length..];
            return $"up_blocks.{upLevel}.upsamplers.0.conv." + rest;
        }

        if (subIdx == 2)
        {
            // Upsample conv (after attention block in levels that have attention)
            if (rest.StartsWith("conv."))
                return $"up_blocks.{upLevel}.upsamplers.0.conv." + rest["conv.".Length..];
            return $"up_blocks.{upLevel}.upsamplers.0.conv." + rest;
        }

        return null;
    }


    // ── CLIP-L Key Conversion ──────────────────────────────────────────

    private static void ConvertClipLKey(string key, Tensor tensor, Dictionary<string, Tensor> clipL)
    {
        // SD1.5: cond_stage_model.transformer.text_model.* → text_model.*
        string prefix = "cond_stage_model.transformer.";
        if (!key.StartsWith(prefix)) return;

        string rest = key[prefix.Length..];

        // Skip position_ids (buffer, not a weight)
        if (rest.EndsWith("position_ids")) return;

        clipL[rest] = tensor;
    }

}
