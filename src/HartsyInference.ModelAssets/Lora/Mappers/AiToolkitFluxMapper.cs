using HartsyInference.Core.Logging;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.ModelAssets.Lora.Mappers;

/// <summary>Parses AI Toolkit (ostris/ai-toolkit) Flux LoRA files. Hybrid format: lora_transformer_ / lora_te1_ prefixes (Kohya-style, underscored) with PEFT-style .lora_A.weight / .lora_B.weight suffixes. QKV is already split because AI Toolkit targets diffusers-loaded models with pre-split attention.</summary>
public static class AiToolkitFluxMapper
{
    public static IReadOnlyList<LoraLayer> ParseLayers(SafeTensorsLoader loader)
    {
        Dictionary<(LoraTarget, string), LoraGroupBuffer> groups = [];

        foreach (string key in loader.Descriptors.Keys)
        {
            if (!LoraRoleSuffix.TryStrip(key, out string root, out LoraRole role))
            {
                continue;
            }
            // AI Toolkit folds alpha at save time when peft_format=True (forced for Flux), so the scale is 1.0 and a
            // stray alpha entry must not change it — the buffer's default (alpha == rank) is the folded scale.
            if (role == LoraRole.Alpha)
            {
                continue;
            }
            if (role is LoraRole.Diff or LoraRole.BiasDiff)
            {
                Logs.Warning($"AI Toolkit Flux LoRA key '{key}' is a full-weight diff, which this format does not carry; skipping.");
                continue;
            }

            string body;
            LoraTarget target;
            if (root.StartsWith("lora_transformer_", StringComparison.Ordinal))
            {
                body = root["lora_transformer_".Length..];
                target = LoraTarget.Transformer;
            }
            else if (root.StartsWith("lora_te1_", StringComparison.Ordinal))
            {
                body = root["lora_te1_".Length..];
                target = LoraTarget.ClipL;
            }
            else if (root.StartsWith("lora_te_", StringComparison.Ordinal))
            {
                body = root["lora_te_".Length..];
                target = LoraTarget.ClipL;
            }
            else
            {
                Logs.Warning($"AI Toolkit Flux LoRA key '{key}' has unrecognized prefix; skipping.");
                continue;
            }

            string canonicalKey = LoraKeyTransformer.UnderscoreToDot(body) + ".weight";
            LoraGroupBuffer group = LoraGroupBuffer.GetOrCreate(groups, target, canonicalKey, key);
            group.Assign(role, loader.GetTensor(key));
        }

        return LoraGroupBuffer.BuildLayers(groups,
            sourceKey => $"AI Toolkit Flux LoRA group '{sourceKey}' is missing a matrix its decomposition needs; skipping.");
    }
}
