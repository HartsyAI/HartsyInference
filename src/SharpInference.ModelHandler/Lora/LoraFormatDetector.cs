using SharpInference.ModelHandler.SafeTensors;

namespace SharpInference.ModelHandler.Lora;

/// <summary>Detects which LoRA naming format a safetensors file uses by inspecting key prefixes. See docs/Design/LORA_KEY_MAPPING.md for the precedence rules.</summary>
public static class LoraFormatDetector
{
    /// <summary>Returns the detected format, or LoraFormat.Unknown if no rule matches. Rules are checked in fixed precedence order.</summary>
    public static LoraFormat Detect(IReadOnlyDictionary<string, SafeTensorDescriptor> descriptors)
    {
        bool hasDiffusersFlux = false;
        bool hasAiToolkitFlux = false;
        bool hasKohyaFluxBlocks = false;
        bool hasKohyaUnetBlocks = false;
        bool hasTe2 = false;

        foreach (string key in descriptors.Keys)
        {
            if (key.StartsWith("transformer.transformer_blocks.", StringComparison.Ordinal)
                || key.StartsWith("transformer.single_transformer_blocks.", StringComparison.Ordinal))
            {
                hasDiffusersFlux = true;
            }
            else if (key.StartsWith("lora_transformer_", StringComparison.Ordinal))
            {
                hasAiToolkitFlux = true;
            }
            else if (key.StartsWith("lora_unet_double_blocks_", StringComparison.Ordinal)
                || key.StartsWith("lora_unet_single_blocks_", StringComparison.Ordinal))
            {
                hasKohyaFluxBlocks = true;
            }
            else if (key.StartsWith("lora_unet_down_blocks_", StringComparison.Ordinal)
                || key.StartsWith("lora_unet_up_blocks_", StringComparison.Ordinal)
                || key.StartsWith("lora_unet_mid_block_", StringComparison.Ordinal))
            {
                hasKohyaUnetBlocks = true;
            }

            if (key.StartsWith("lora_te2_", StringComparison.Ordinal))
            {
                hasTe2 = true;
            }
        }

        if (hasDiffusersFlux) return LoraFormat.DiffusersFlux;
        if (hasAiToolkitFlux) return LoraFormat.AiToolkitFlux;
        if (hasKohyaFluxBlocks) return LoraFormat.KohyaFlux;
        if (hasKohyaUnetBlocks && hasTe2) return LoraFormat.KohyaSdxl;
        if (hasKohyaUnetBlocks) return LoraFormat.KohyaSd15;

        return LoraFormat.Unknown;
    }
}
