using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.ModelAssets.Lora.Mappers;

/// <summary>Parses HuggingFace PEFT-style diffusers Flux LoRA files. Keys are dotted throughout (transformer.transformer_blocks.0.attn.to_q.lora_A.weight) with .lora_A.weight / .lora_B.weight suffixes; alpha may be embedded in the file or default to rank.</summary>
public static class DiffusersFluxMapper
{
    private const string DownSuffix = ".lora_A.weight";
    private const string UpSuffix = ".lora_B.weight";
    private const string AlphaSuffix = ".alpha";
    // Kohya spellings of the same two roles — lightx2v's Lightning LoRAs ship bare diffusers roots with
    // .lora_down/.lora_up suffixes, so both suffix families are accepted on every root this parser takes.
    private const string KohyaDownSuffix = ".lora_down.weight";
    private const string KohyaUpSuffix = ".lora_up.weight";

    public static IReadOnlyList<LoraLayer> ParseLayers(SafeTensorsLoader loader) => ParseLayers(loader, bareRoots: false);

    /// <summary>Same parse, but <paramref name="bareRoots"/> accepts a root with NO wrapper prefix as a transformer target — the root is then already the canonical weight name. Shares this parser rather than getting its own because the two formats differ only in that one rule.</summary>
    public static IReadOnlyList<LoraLayer> ParseLayers(SafeTensorsLoader loader, bool bareRoots)
    {
        Dictionary<(LoraTarget, string), LoraGroupBuffer> groups = [];

        foreach (string key in loader.Descriptors.Keys)
        {
            string root;
            LoraRole role;
            if (key.EndsWith(DownSuffix, StringComparison.Ordinal))
            {
                role = LoraRole.Down;
                root = key[..^DownSuffix.Length];
            }
            else if (key.EndsWith(UpSuffix, StringComparison.Ordinal))
            {
                role = LoraRole.Up;
                root = key[..^UpSuffix.Length];
            }
            else if (key.EndsWith(KohyaDownSuffix, StringComparison.Ordinal))
            {
                role = LoraRole.Down;
                root = key[..^KohyaDownSuffix.Length];
            }
            else if (key.EndsWith(KohyaUpSuffix, StringComparison.Ordinal))
            {
                role = LoraRole.Up;
                root = key[..^KohyaUpSuffix.Length];
            }
            else if (key.EndsWith(AlphaSuffix, StringComparison.Ordinal))
            {
                role = LoraRole.Alpha;
                root = key[..^AlphaSuffix.Length];
            }
            else
            {
                continue;
            }

            string body;
            LoraTarget target;
            if (root.StartsWith("transformer.", StringComparison.Ordinal))
            {
                body = root["transformer.".Length..];
                target = LoraTarget.Transformer;
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
            switch (role)
            {
                case LoraRole.Down: group.Down = loader.GetTensor(key); break;
                case LoraRole.Up: group.Up = loader.GetTensor(key); break;
                case LoraRole.Alpha: group.Alpha = KohyaSdMapper.ReadScalar(loader.GetTensor(key)); break;
            }
        }

        return LoraGroupBuffer.BuildLayers(groups,
            sourceKey => $"Diffusers Flux LoRA group '{sourceKey}' missing down or up; skipping.");
    }

    private enum LoraRole { Down, Up, Alpha }
}
