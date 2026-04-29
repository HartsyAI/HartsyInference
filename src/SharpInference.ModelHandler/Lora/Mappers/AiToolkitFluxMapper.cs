using SharpInference.Core.Logging;
using SharpInference.Core.Tensors;
using SharpInference.ModelHandler.SafeTensors;

namespace SharpInference.ModelHandler.Lora.Mappers;

/// <summary>Parses AI Toolkit (ostris/ai-toolkit) Flux LoRA files. Hybrid format: lora_transformer_ / lora_te1_ prefixes (Kohya-style, underscored) with PEFT-style .lora_A.weight / .lora_B.weight suffixes. No .alpha entries — alpha is folded at save time when peft_format=True (forced for Flux). QKV is already split because AI Toolkit targets diffusers-loaded models with pre-split attention.</summary>
public static class AiToolkitFluxMapper
{
    private const string DownSuffix = ".lora_A.weight";
    private const string UpSuffix = ".lora_B.weight";

    /// <summary>Parses every LoRA layer in the file.</summary>
    public static IReadOnlyList<LoraLayer> ParseLayers(SafeTensorsLoader loader)
    {
        Dictionary<(LoraTarget, string), GroupBuffer> groups = [];

        foreach (string key in loader.Descriptors.Keys)
        {
            bool isDown = key.EndsWith(DownSuffix, StringComparison.Ordinal);
            bool isUp = !isDown && key.EndsWith(UpSuffix, StringComparison.Ordinal);
            if (!isDown && !isUp)
            {
                continue;
            }

            string root = isDown
                ? key[..^DownSuffix.Length]
                : key[..^UpSuffix.Length];

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
            (LoraTarget, string) gk = (target, canonicalKey);
            if (!groups.TryGetValue(gk, out GroupBuffer? group))
            {
                group = new GroupBuffer { Target = target, FirstSourceKey = key };
                groups[gk] = group;
            }
            if (isDown) group.Down = loader.GetTensor(key);
            else group.Up = loader.GetTensor(key);
        }

        List<LoraLayer> layers = new(groups.Count);
        foreach (((LoraTarget _, string canonicalKey), GroupBuffer group) in groups)
        {
            if (group.Down is null || group.Up is null)
            {
                Logs.Warning($"AI Toolkit Flux LoRA group '{group.FirstSourceKey}' missing down or up; skipping.");
                continue;
            }
            int rank = (int)group.Down.Shape[0];
            // AI Toolkit folds alpha at save time; default scale = 1.0 (alpha == rank).
            float alpha = rank;
            layers.Add(new LoraLayer
            {
                TargetKey = canonicalKey,
                Target = group.Target,
                LoraDown = group.Down,
                LoraUp = group.Up,
                Alpha = alpha,
                Rank = rank,
                Variant = LoraVariant.StandardLora,
            });
        }
        return layers;
    }

    private sealed class GroupBuffer
    {
        public required LoraTarget Target { get; init; }
        public required string FirstSourceKey { get; init; }
        public Tensor? Down { get; set; }
        public Tensor? Up { get; set; }
    }
}
