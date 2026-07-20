using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelAssets.CheckpointConverters;

/// <summary>Converts LDM-layout ControlNet checkpoints (the original lllyasviel releases and the comfyanonymous fp16 repacks, keys prefixed <c>control_model.*</c> or bare <c>input_blocks.*</c>/<c>zero_convs.*</c>) to the diffusers <c>ControlNetModel</c> key layout. A ControlNet is the base UNet's encoder half (conv_in, time/label embed, input_blocks → down_blocks, middle_block → mid_block) plus three ControlNet-specific towers: <c>input_hint_block.*</c> → <c>controlnet_cond_embedding.*</c>, <c>zero_convs.N.0.*</c> → <c>controlnet_down_blocks.N.*</c>, and <c>middle_block_out.0.*</c> → <c>controlnet_mid_block.*</c>. Mapping mirrors diffusers' <c>convert_from_ckpt.convert_controlnet_checkpoint</c> and is verified key-for-key against lllyasviel's diffusers repacks.</summary>
public static class ControlNetCheckpointConverter
{
    private const string ControlModelPrefix = "control_model.";

    /// <summary>True when the key set is an LDM-layout ControlNet (either <c>control_model.*</c>-prefixed or bare LDM keys with the ControlNet-specific <c>input_hint_block.</c>/<c>zero_convs.</c>/<c>middle_block_out.</c> towers).</summary>
    public static bool IsLdmLayout(IEnumerable<string> keys)
    {
        foreach (string key in keys)
        {
            if (key.StartsWith(ControlModelPrefix, StringComparison.Ordinal) ||
                key.StartsWith("input_hint_block.", StringComparison.Ordinal) ||
                key.StartsWith("zero_convs.", StringComparison.Ordinal) ||
                key.StartsWith("middle_block_out.", StringComparison.Ordinal))
                return true;
        }
        return false;
    }

    /// <summary>Strips the optional <c>control_model.</c> prefix from an LDM ControlNet key.</summary>
    public static string StripPrefix(string key)
        => key.StartsWith(ControlModelPrefix, StringComparison.Ordinal) ? key[ControlModelPrefix.Length..] : key;

    /// <summary>Converts a full LDM-layout ControlNet weight dictionary to diffusers keys. Keys with no diffusers counterpart (none exist in real checkpoints) are dropped. Tensor instances are carried over unchanged — ownership stays with the originating loader.</summary>
    /// <param name="isSdxl">Selects the SDXL input-block numbering (9 encoder blocks, label_emb) over SD1.5's (12 encoder blocks).</param>
    public static Dictionary<string, Tensor> Convert(Dictionary<string, Tensor> allWeights, bool isSdxl)
    {
        Dictionary<string, Tensor> converted = new(allWeights.Count);
        foreach (KeyValuePair<string, Tensor> kvp in allWeights)
        {
            string? diffusersKey = ConvertKey(kvp.Key, isSdxl);
            if (diffusersKey is not null)
                converted[diffusersKey] = kvp.Value;
        }
        return converted;
    }

    /// <summary>Converts one LDM ControlNet key (with or without the <c>control_model.</c> prefix) to its diffusers name, or null when the key has no diffusers counterpart.</summary>
    public static string? ConvertKey(string key, bool isSdxl)
    {
        string ldmKey = StripPrefix(key);

        if (ldmKey.StartsWith("input_hint_block.", StringComparison.Ordinal))
            return ConvertHintBlockKey(ldmKey);

        if (ldmKey.StartsWith("zero_convs.", StringComparison.Ordinal))
            return ConvertZeroConvKey(ldmKey);

        if (ldmKey.StartsWith("middle_block_out.0.", StringComparison.Ordinal))
            return "controlnet_mid_block." + ldmKey["middle_block_out.0.".Length..];

        return isSdxl ? SdxlCheckpointConverter.ConvertUNetKey(ldmKey) : Sd15CheckpointConverter.ConvertUNetKey(ldmKey);
    }

    // input_hint_block is a Conv/SiLU alternation: convs sit at even indices 0..14.
    // Diffusers maps 0 → conv_in, 2/4/6/8/10/12 → blocks.0..5, 14 → conv_out.
    private static string? ConvertHintBlockKey(string ldmKey)
    {
        string afterPrefix = ldmKey["input_hint_block.".Length..];
        int firstDot = afterPrefix.IndexOf('.');
        if (firstDot < 0) return null;
        if (!int.TryParse(afterPrefix[..firstDot], out int idx)) return null;
        string rest = afterPrefix[(firstDot + 1)..];

        if (idx == 0) return "controlnet_cond_embedding.conv_in." + rest;
        if (idx == 14) return "controlnet_cond_embedding.conv_out." + rest;
        if (idx >= 2 && idx <= 12 && idx % 2 == 0)
            return $"controlnet_cond_embedding.blocks.{idx / 2 - 1}." + rest;
        return null;
    }

    // zero_convs.N is a single-module Sequential — the 1×1 conv always sits at sub-index 0.
    private static string? ConvertZeroConvKey(string ldmKey)
    {
        string afterPrefix = ldmKey["zero_convs.".Length..];
        int firstDot = afterPrefix.IndexOf('.');
        if (firstDot < 0) return null;
        if (!int.TryParse(afterPrefix[..firstDot], out int idx)) return null;
        string rest = afterPrefix[(firstDot + 1)..];

        if (!rest.StartsWith("0.", StringComparison.Ordinal)) return null;
        return $"controlnet_down_blocks.{idx}." + rest[2..];
    }
}
