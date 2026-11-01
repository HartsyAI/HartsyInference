using HartsyInference.Core.Logging;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.ModelAssets.Lora.Mappers;

/// <summary>Parses Krea 2 LoRA files (ai-toolkit, ComfyUI spelling) onto the transformer's diffusers weight names.
/// <para>The file names modules as the checkpoint does (<c>diffusion_model.blocks.N.attn.wq</c>, <c>mlp.gate</c>, and the
/// <c>txtfusion.layerwise_blocks.N</c> / <c>txtfusion.refiner_blocks.N</c> text-fusion streams). Those are the names
/// <see cref="Krea2CheckpointConverter.RemapTransformerKey"/> takes to the keys the transformer holds, so each layer is
/// mapped through the same function the checkpoint itself uses. The Wan rename table would send <c>attn.wq</c> to nothing,
/// which is why a Krea file matched 0 weights and was refused.</para></summary>
public static class Krea2LoraMapper
{
    /// <summary>Root wrappers in front of the checkpoint's own module path; a root carrying none is the path itself.</summary>
    private static readonly string[] Wrappers = ["model.diffusion_model.", "diffusion_model."];

    /// <summary>The block roots a Krea 2 file uses: the main stream, then the two text-fusion streams.</summary>
    private static readonly string[] BlockRoots = ["blocks.", "txtfusion.layerwise_blocks.", "txtfusion.refiner_blocks."];

    /// <summary>Module paths inside a block that only Krea 2 names this way: the attention projections (<c>attn.wq</c>,
    /// <c>attn.gate</c>, …) and the gated FFN (<c>mlp.gate</c>, <c>mlp.up</c>, <c>mlp.down</c>).</summary>
    private static readonly HashSet<string> Modules = new(StringComparer.Ordinal)
    {
        "attn.wq", "attn.wk", "attn.wv", "attn.wo", "attn.gate",
        "mlp.gate", "mlp.up", "mlp.down",
    };

    /// <summary>The modules that identify the family on their own. The FFN names are shared with other families, so a file
    /// is claimed only by an attention projection, and its FFN layers then ride along.</summary>
    private static readonly HashSet<string> DistinctiveModules = new(StringComparer.Ordinal)
    {
        "attn.wq", "attn.wk", "attn.wv", "attn.wo", "attn.gate",
    };

    /// <summary>Whether <paramref name="key"/> names a Krea 2 block module by one of its attention projections.</summary>
    public static bool IsDistinctive(string key)
    {
        ArgumentNullException.ThrowIfNull(key);
        return LoraRoleSuffix.TryStrip(key, out string root, out LoraRole _)
            && TryCanonicalBody(root, out string body)
            && DistinctiveModules.Contains(ModuleOf(body));
    }

    /// <summary>Parses every Krea 2 block LoRA layer in the file onto <see cref="LoraTarget.Transformer"/>. A full-weight diff is
    /// refused by name: it is not a low-rank adapter, so applying the rest of the file would give wrong weights.</summary>
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
            if (!TryCanonicalBody(root, out string body))
            {
                foreign++;
                continue;
            }
            if (role is LoraRole.Diff or LoraRole.BiasDiff)
            {
                throw new NotSupportedException($"Krea 2 LoRA key '{key}' is a full-weight diff. Only low-rank adapters are supported; "
                    + "applying the rest of this file would give wrong weights.");
            }
            string transformerKey = Krea2CheckpointConverter.RemapTransformerKey(body + ".weight");
            LoraGroupBuffer.GetOrCreate(groups, LoraTarget.Transformer, transformerKey, key).Assign(role, loader.GetTensor(key));
        }

        if (foreign > 0)
        {
            Logs.Warning($"Krea 2 LoRA: skipped {foreign} keys outside the DiT block modules (text-encoder adapters are not applied for this format).");
        }
        IReadOnlyList<LoraLayer> layers = LoraGroupBuffer.BuildLayers(groups,
            sourceKey => $"Krea 2 LoRA group '{sourceKey}' is missing a matrix its decomposition needs; skipping.");
        // A file the format detector claimed but that yields no complete layer would otherwise load as an empty LoRA and generate unchanged.
        if (layers.Count == 0)
        {
            throw new NotSupportedException("Krea 2 LoRA has no complete block-module layers, so it would change nothing. Check that the file "
                + "is a low-rank Krea 2 adapter with both matrices of each pair.");
        }
        return layers;
    }

    /// <summary>Strips one accepted wrapper and checks the rest is a block module Krea 2 owns, returning that module path
    /// (<c>blocks.N.attn.wq</c>) as <paramref name="body"/>.</summary>
    private static bool TryCanonicalBody(string root, out string body)
    {
        body = root;
        foreach (string wrapper in Wrappers)
        {
            if (root.StartsWith(wrapper, StringComparison.Ordinal))
            {
                body = root[wrapper.Length..];
                break;
            }
        }
        foreach (string blockRoot in BlockRoots)
        {
            if (!body.StartsWith(blockRoot, StringComparison.Ordinal))
            {
                continue;
            }
            string rest = body[blockRoot.Length..];
            int separator = rest.IndexOf('.', StringComparison.Ordinal);
            return separator > 0 && rest[..separator].All(char.IsAsciiDigit) && Modules.Contains(rest[(separator + 1)..]);
        }
        return false;
    }

    /// <summary>The module path inside a block (<c>attn.wq</c> for <c>blocks.0.attn.wq</c>), for a body already checked by
    /// <see cref="TryCanonicalBody"/>.</summary>
    private static string ModuleOf(string body)
    {
        foreach (string blockRoot in BlockRoots)
        {
            if (body.StartsWith(blockRoot, StringComparison.Ordinal))
            {
                string rest = body[blockRoot.Length..];
                return rest[(rest.IndexOf('.', StringComparison.Ordinal) + 1)..];
            }
        }
        return string.Empty;
    }
}
