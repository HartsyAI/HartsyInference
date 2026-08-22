using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.ModelAssets.Lora;

/// <summary>Detects which LoRA naming format a safetensors file uses by inspecting key prefixes. The precedence rules are the return order at the bottom of <see cref="Detect"/>, strongest marker first: a file carrying a recognized wrapper prefix belongs to that wrapper's format, and only a file carrying none of them falls through to the roots-are-canonical bare-DiT reading.</summary>
public static class LoraFormatDetector
{
    /// <summary>Every wrapper prefix an earlier precedence arm claims. A key carrying one of these is that format's business, never the bare-root fallback's — without this exclusion a kohya file whose block roots aren't in any recognized shape would fall through to bare-root and derive `lora_unet_*` keys the converted dict never has.</summary>
    private static readonly string[] _knownWrapperPrefixes =
    [
        "transformer.", "text_encoder.", "diffusion_model.",
        "lora_transformer_", "lora_unet_", "lora_te_", "lora_te1_", "lora_te2_",
    ];

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
        bool hasComfyBfl = false;

        foreach (string key in descriptors.Keys)
        {
            // No wrapper prefix at all — the root IS the checkpoint's own key (MiniMax-H3's published Turbo LoRA:
            // `blocks.0.attn.qkv_proj.lora_A.weight`, `token_refiner.blocks.0.mlp.fc1.lora_B.weight`). Recorded
            // rather than returned so it stays LAST in the precedence list below: "carries a LoRA suffix and no
            // recognized wrapper" is the weakest marker there is and must never win over a real prefix.
            //
            // This used to be an allow-list of block roots (`blocks.`, `transformer_blocks.`, `token_refiner.`,
            // `final_layer.`, `single_transformer_blocks.`), which silently excluded every family that names its
            // blocks something else — `layers.{i}` (Ideogram 4 / ERNIE-Image / Lance), `text_transformer_blocks.` +
            // `visual_transformer_blocks.` (Kandinsky 5), `joint_transformer_blocks.` (AuraFlow),
            // `double_stream_layers.` / `noise_refiner.` / `context_refiner.` (Boogu-Image). A bare-root LoRA for
            // any of those was rejected as Unknown at load. The rule is now the general one the allow-list was
            // approximating, so a family added later needs no detector change at all.
            if (!HasKnownWrapperPrefix(key) && HasLoraSuffix(key))
            {
                hasBareDit = true;
            }
            // HuggingFace PEFT diffusers format — the `transformer.`-wrapped counterpart of the bare-root arm
            // above, and the format most community LoRAs actually ship in. The mapper is architecture-agnostic:
            // it strips `transformer.` and passes the body through as the canonical key, so the block root is
            // irrelevant to it and matching on one is a needless narrowing. This was a hard-coded list of three
            // roots (`transformer_blocks.` / `single_transformer_blocks.` / `blocks.`), which rejected — as an
            // undetectable format, at load — every LoRA for a family naming its blocks anything else:
            // `transformer.layers.{i}.*` (Ideogram 4 / ERNIE-Image / Lance), `transformer.text_transformer_blocks.{i}.*`
            // (Kandinsky 5), `transformer.joint_transformer_blocks.{i}.*` (AuraFlow), and so on.
            //
            // The `text_encoder.` arm is the mapper's other recognized wrapper (routed to LoraTarget.ClipL), and
            // belongs to the same format.
            if ((key.StartsWith("transformer.", StringComparison.Ordinal)
                    || key.StartsWith("text_encoder.", StringComparison.Ordinal))
                && HasLoraSuffix(key))
            {
                hasDiffusersFlux = true;
            }
            // ComfyUI-style Wan repacks: dotted original-Wan naming under a diffusion_model. prefix
            // (e.g. lightx2v distill LoRAs, Kijai's WanVideo conversions).
            else if (key.StartsWith("diffusion_model.double_blocks.", StringComparison.Ordinal)
                || key.StartsWith("diffusion_model.single_blocks.", StringComparison.Ordinal))
            {
                hasComfyBfl = true;
            }
            else if (key.StartsWith("diffusion_model.blocks.", StringComparison.Ordinal))
            {
                hasDiffusersWan = true;
            }
            // Bare original-Wan naming (no wrapper prefix): the osantinello Wan-Animate relight conversion.
            // self_attn/cross_attn segments distinguish it from the generic bare-DiT fallback, whose
            // roots-are-canonical rule would derive keys the converted (diffusers-named) dict never has.
            else if (key.StartsWith("blocks.", StringComparison.Ordinal)
                && (key.Contains(".self_attn.", StringComparison.Ordinal) || key.Contains(".cross_attn.", StringComparison.Ordinal)))
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
        if (hasComfyBfl) return LoraFormat.ComfyBflDit;
        if (hasDiffusersWan) return LoraFormat.DiffusersWan;
        if (hasKohyaWan) return LoraFormat.KohyaWan;
        if (hasAiToolkitFlux) return LoraFormat.AiToolkitFlux;
        if (hasKohyaFluxBlocks) return LoraFormat.KohyaFlux;
        if (hasKohyaUnetBlocks && hasTe2) return LoraFormat.KohyaSdxl;
        if (hasKohyaUnetBlocks) return LoraFormat.KohyaSd15;
        if (hasBareDit) return LoraFormat.DiffusersBareDit;

        return LoraFormat.Unknown;
    }

    /// <summary>Whether <paramref name="key"/> starts with any prefix an earlier-precedence format owns.</summary>
    private static bool HasKnownWrapperPrefix(string key)
    {
        foreach (string prefix in _knownWrapperPrefixes)
        {
            if (key.StartsWith(prefix, StringComparison.Ordinal))
            {
                return true;
            }
        }
        return false;
    }

    /// <summary>Whether <paramref name="key"/> ends with a PEFT (<c>.lora_A</c>/<c>.lora_B</c>) or kohya (<c>.lora_down</c>/<c>.lora_up</c>) role suffix — both spellings are accepted on every root <see cref="Mappers.DiffusersFluxMapper"/> parses.</summary>
    private static bool HasLoraSuffix(string key) =>
        key.EndsWith(".lora_A.weight", StringComparison.Ordinal)
        || key.EndsWith(".lora_B.weight", StringComparison.Ordinal)
        || key.EndsWith(".lora_down.weight", StringComparison.Ordinal)
        || key.EndsWith(".lora_up.weight", StringComparison.Ordinal);
}
