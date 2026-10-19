using HartsyInference.Core.Logging;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.ModelAssets.Lora.Mappers;

/// <summary>Parses MOSS-Image LoRA files onto the transformer keys <c>MossImageTransformer</c> loads, in each spelling a
/// trainer emits: diffusers/PEFT under <c>transformer.</c> (OpenMOSS's diffusers pipeline), ComfyUI's
/// <c>diffusion_model.</c> (its built-in LoRA loader on the ComfyUI export), bare module names, and underscored
/// <c>lora_unet_</c> / <c>lora_transformer_</c> / <c>lora_diffusion_model_</c> (kohya and other trainers). Ostris /
/// ai-toolkit edit LoRAs adapt the same modules as any other. Every spelling names the same module once its wrapper is
/// gone, so no rows are reordered or split; names resolve through <see cref="MossImageKeyNames"/>.</summary>
public static class MossImageLoraMapper
{
    /// <summary>Underscored roots, longest first: kohya's <c>lora_unet_</c> and the <c>lora_transformer_</c> /
    /// <c>lora_diffusion_model_</c> spellings other trainers (ai-toolkit's older exports among them) write.</summary>
    private static readonly string[] UnderscoredPrefixes = ["lora_diffusion_model_", "lora_transformer_", "lora_unet_"];

    /// <summary>Dotted wrappers in front of a module path, longest first; a root carrying none is the module path itself.</summary>
    private static readonly string[] DottedWrappers = ["model.diffusion_model.", "diffusion_model.", "transformer."];

    /// <summary>Parses every MOSS-Image layer in the file.</summary>
    public static IReadOnlyList<LoraLayer> ParseLayers(SafeTensorsLoader loader)
    {
        ArgumentNullException.ThrowIfNull(loader);
        Dictionary<(LoraTarget, string), LoraGroupBuffer> groups = [];
        foreach (string key in loader.Descriptors.Keys)
        {
            if (LoraRoleSuffix.IsPeftDoraMagnitude(key))
            {
                throw new NotSupportedException($"MOSS-Image LoRA key '{key}' is a PEFT DoRA magnitude vector. PEFT DoRA "
                    + "adapters are not supported, and applying their A/B pairs without it would give wrong weights.");
            }
            if (!LoraRoleSuffix.TryStrip(key, out string root, out LoraRole role))
            {
                continue;
            }
            if (role is LoraRole.Diff or LoraRole.BiasDiff)
            {
                Logs.Warning($"MOSS-Image LoRA key '{key}' is a full-weight diff, which this format does not carry; skipping.");
                continue;
            }
            if (!TryResolve(root, out string module))
            {
                Logs.Warning($"MOSS-Image LoRA key '{key}' does not name a MOSS-Image transformer module; skipping.");
                continue;
            }
            LoraGroupBuffer.GetOrCreate(groups, LoraTarget.Transformer, module + ".weight", key).Assign(role, loader.GetTensor(key));
        }
        return LoraGroupBuffer.BuildLayers(groups,
            sourceKey => $"MOSS-Image LoRA group '{sourceKey}' is missing a matrix its decomposition needs; skipping.");
    }

    /// <summary>Resolves a LoRA root (role suffix already stripped) to the module it adapts. False when the root is not a
    /// MOSS-Image transformer module.</summary>
    public static bool TryResolve(string root, out string module)
    {
        ArgumentNullException.ThrowIfNull(root);
        string? resolved = null;
        foreach (string prefix in UnderscoredPrefixes)
        {
            if (root.StartsWith(prefix, StringComparison.Ordinal))
            {
                resolved = MossImageKeyNames.Modules.FromKohya(root[prefix.Length..]);
                break;
            }
        }
        if (resolved is null && !root.StartsWith("lora_", StringComparison.Ordinal))
            resolved = StripDottedWrapper(root) is string body && MossImageKeyNames.Modules.IsModule(body) ? body : null;
        module = resolved ?? string.Empty;
        return resolved is not null;
    }

    /// <summary>Whether the LoRA file key <paramref name="key"/> adapts a module only MOSS-Image has — the detector's marker.</summary>
    public static bool IsDistinctive(string key)
    {
        ArgumentNullException.ThrowIfNull(key);
        return LoraRoleSuffix.TryStrip(key, out string root, out LoraRole _)
            && TryResolve(root, out string module)
            && MossImageKeyNames.IsDistinctive(module);
    }

    private static string StripDottedWrapper(string root)
    {
        foreach (string wrapper in DottedWrappers)
        {
            if (root.StartsWith(wrapper, StringComparison.Ordinal))
                return root[wrapper.Length..];
        }
        return root;
    }
}
