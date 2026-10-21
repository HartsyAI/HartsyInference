using HartsyInference.Core.Logging;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.ModelAssets.Lora.Mappers;

/// <summary>Parses kohya/musubi-tuner <c>lora_unet_&lt;module path&gt;</c> files whose layout no specific kohya mapper claims. The file alone cannot say which weight <c>blocks_0_attn_wq</c> is, so each layer keeps the underscored name in <see cref="LoraLayer.KohyaModuleName"/> and <see cref="LoraStack"/> resolves it through <see cref="KohyaUnderscoreIndex"/> against the weights it is applied to.</summary>
public static class KohyaGenericMapper
{
    private const string Prefix = "lora_unet_";

    /// <summary>Parses every <c>lora_unet_</c> module in the file onto <see cref="LoraTarget.Transformer"/>, unresolved.</summary>
    public static IReadOnlyList<LoraLayer> ParseLayers(SafeTensorsLoader loader)
    {
        ArgumentNullException.ThrowIfNull(loader);
        Dictionary<(LoraTarget, string), LoraGroupBuffer> groups = [];
        int foreign = 0;
        foreach (string key in loader.Descriptors.Keys)
        {
            if (!LoraRoleSuffix.TryStrip(key, out string root, out LoraRole role))
            {
                continue;
            }
            if (role is LoraRole.Diff or LoraRole.BiasDiff)
            {
                Logs.Warning($"Kohya LoRA key '{key}' is a full-weight diff, which this format does not carry; skipping.");
                continue;
            }
            if (!root.StartsWith(Prefix, StringComparison.Ordinal))
            {
                foreign++;
                continue;
            }
            LoraGroupBuffer.GetOrCreate(groups, LoraTarget.Transformer, root, key).Assign(role, loader.GetTensor(key));
        }

        if (foreign > 0)
            Logs.Warning($"Kohya LoRA: skipped {foreign} keys outside lora_unet_ (text-encoder adapters are not applied for this format).");
        IReadOnlyList<LoraLayer> built = LoraGroupBuffer.BuildLayers(groups,
            sourceKey => $"Kohya LoRA group '{sourceKey}' is missing a matrix its decomposition needs; skipping.");
        List<LoraLayer> layers = new(built.Count);
        foreach (LoraLayer layer in built)
        {
            layers.Add(new LoraLayer
            {
                TargetKey = layer.TargetKey,
                Target = layer.Target,
                Delta = layer.Delta,
                KohyaModuleName = layer.TargetKey[Prefix.Length..],
            });
        }
        return layers;
    }
}
