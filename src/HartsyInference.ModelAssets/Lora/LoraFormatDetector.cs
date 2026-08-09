using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.ModelAssets.Lora;

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
        bool hasKohyaWan = false;
        bool hasDiffusersWan = false;
        bool hasTe2 = false;
        bool hasBareDit = false;

        foreach (string key in descriptors.Keys)
        {
            // No wrapper prefix at all — the root IS the checkpoint's own key (MiniMax-H3's published Turbo LoRA:
            // `blocks.0.attn.qkv_proj.lora_A.weight`, `token_refiner.blocks.0.mlp.fc1.lora_B.weight`). Recorded
            // rather than returned so it stays LAST in the precedence list below: `blocks.` is a weak marker and
            // must never win over a file that also carries a real prefix.
            if ((key.StartsWith("blocks.", StringComparison.Ordinal)
                    || key.StartsWith("token_refiner.", StringComparison.Ordinal)
                    || key.StartsWith("final_layer.", StringComparison.Ordinal))
                && (key.EndsWith(".lora_A.weight", StringComparison.Ordinal)
                    || key.EndsWith(".lora_B.weight", StringComparison.Ordinal)))
            {
                hasBareDit = true;
            }
            // HuggingFace PEFT diffusers format. The mapper is architecture-agnostic — it strips the
            // `transformer.` prefix and passes the body through as the canonical key — so the same arm
            // covers Flux (`transformer_blocks` / `single_transformer_blocks`) AND single-stream DiTs that
            // name blocks `transformer.blocks.{i}.*`: Anima / Cosmos-Predict2 (`blocks.{i}.self_attn.q_proj.weight`)
            // and diffusers-PEFT Wan (`blocks.{i}.attn1.to_q.weight`) — both exact passthrough matches.
            if (key.StartsWith("transformer.transformer_blocks.", StringComparison.Ordinal)
                || key.StartsWith("transformer.single_transformer_blocks.", StringComparison.Ordinal)
                || key.StartsWith("transformer.blocks.", StringComparison.Ordinal))
            {
                hasDiffusersFlux = true;
            }
            // ComfyUI-style Wan repacks: dotted original-Wan naming under a diffusion_model. prefix
            // (e.g. lightx2v distill LoRAs, Kijai's WanVideo conversions).
            else if (key.StartsWith("diffusion_model.blocks.", StringComparison.Ordinal))
            {
                hasDiffusersWan = true;
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
            // Kohya/musubi-tuner Wan: flat `blocks_{i}` (no down/up/mid/double/single) with the Wan
            // attention names as the distinguishing marker.
            else if (key.StartsWith("lora_unet_blocks_", StringComparison.Ordinal)
                && (key.Contains("self_attn", StringComparison.Ordinal) || key.Contains("cross_attn", StringComparison.Ordinal)))
            {
                hasKohyaWan = true;
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
        if (hasDiffusersWan) return LoraFormat.DiffusersWan;
        if (hasKohyaWan) return LoraFormat.KohyaWan;
        if (hasAiToolkitFlux) return LoraFormat.AiToolkitFlux;
        if (hasKohyaFluxBlocks) return LoraFormat.KohyaFlux;
        if (hasKohyaUnetBlocks && hasTe2) return LoraFormat.KohyaSdxl;
        if (hasKohyaUnetBlocks) return LoraFormat.KohyaSd15;
        if (hasBareDit) return LoraFormat.DiffusersBareDit;

        return LoraFormat.Unknown;
    }
}
