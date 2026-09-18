using HartsyInference.Core.Logging;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.ModelAssets.Lora.Mappers;

/// <summary>Parses HuggingFace PEFT-style diffusers Flux LoRA files. Keys are dotted throughout (transformer.transformer_blocks.0.attn.to_q.lora_A.weight) with any suffix <see cref="LoraRoleSuffix"/> recognizes; alpha may be embedded in the file or default to rank.</summary>
public static class DiffusersFluxMapper
{
    public static IReadOnlyList<LoraLayer> ParseLayers(SafeTensorsLoader loader) => ParseLayers(loader, bareRoots: false);

    /// <summary>Same parse, but <paramref name="bareRoots"/> accepts a root with NO wrapper prefix as a transformer target — the root is then already the canonical weight name. Shares this parser rather than getting its own because the two formats differ only in that one rule.</summary>
    public static IReadOnlyList<LoraLayer> ParseLayers(SafeTensorsLoader loader, bool bareRoots)
    {
        Dictionary<(LoraTarget, string), LoraGroupBuffer> groups = [];

        foreach (string key in loader.Descriptors.Keys)
        {
            if (!LoraRoleSuffix.TryStrip(key, out string root, out LoraRole role))
            {
                continue;
            }
            if (role is LoraRole.Diff or LoraRole.BiasDiff)
            {
                Logs.Warning($"Diffusers Flux LoRA key '{key}' is a full-weight diff, which this format does not carry; skipping.");
                continue;
            }

            string body;
            LoraTarget target;
            if (root.StartsWith("transformer.", StringComparison.Ordinal))
            {
                body = root["transformer.".Length..];
                target = LoraTarget.Transformer;
            }
            else if (root.StartsWith("text_encoder_2.", StringComparison.Ordinal))
            {
                // Checked before the shorter prefix: "text_encoder." is a prefix of "text_encoder_2." only in the
                // reverse order, but keeping the longer test first makes that independent of the spelling.
                body = root["text_encoder_2.".Length..];
                target = LoraTarget.TextEncoder2;
            }
            else if (root.StartsWith("text_encoder.", StringComparison.Ordinal))
            {
                body = root["text_encoder.".Length..];
                target = LoraTarget.ClipL;
            }
            else if (bareRoots)
            {
                body = root;
                target = LoraTarget.Transformer;
            }
            else
            {
                Logs.Warning($"Diffusers Flux LoRA key '{key}' has unrecognized prefix; skipping.");
                continue;
            }

            string canonicalKey = body + ".weight";
            LoraGroupBuffer group = LoraGroupBuffer.GetOrCreate(groups, target, canonicalKey, key);
            group.Assign(role, loader.GetTensor(key));
        }

        return LoraGroupBuffer.BuildLayers(groups,
            sourceKey => $"Diffusers Flux LoRA group '{sourceKey}' is missing a matrix its decomposition needs; skipping.");
    }
}
