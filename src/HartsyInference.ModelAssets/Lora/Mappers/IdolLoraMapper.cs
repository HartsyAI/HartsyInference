using HartsyInference.Core.Logging;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.ModelAssets.Lora.Mappers;

/// <summary>Parses Idol LoRA files (the Comfy-format <c>idol_turbo_slow_comfy</c> / <c>idol_turbo_fast_comfy</c> phase experts) onto Idol's own denoiser weight names
/// (<c>blocks.N.attn.qkv.weight</c>, <c>noise_refiner.N.mlp.mlp.0.weight</c>, …), which are also the names the Comfy-Org/Idol checkpoint and idol-sdk use.
/// <para>The module paths already match, so nothing is renamed. The matrices are <c>.lora_down</c> / <c>.lora_up</c>, bare or under the ComfyUI
/// <c>diffusion_model.</c> wrapper, with no <c>.alpha</c> entry: the file is pre-scaled for strength 1.0 (idol-turbo's own LoRA is rank 32, alpha 128, scale 4,
/// and the Comfy files carry that factor already), so a missing alpha has to mean scale 1, which is what <see cref="LoraGroupBuffer"/> does.</para></summary>
public static class IdolLoraMapper
{
    private static readonly string[] Wrappers = ["model.diffusion_model.", "diffusion_model."];

    /// <summary>The block stacks of the denoiser that carry a LoRA.</summary>
    private static readonly string[] Roots = ["blocks.", "noise_refiner.", "context_refiner."];

    /// <summary>The modules of one block a LoRA adapts: the fused attention projections and the two MLP layers (the MLP names are what no other family uses).</summary>
    private static readonly HashSet<string> Modules = new(StringComparer.Ordinal) { "attn.qkv", "attn.out", "mlp.mlp.0", "mlp.mlp.2" };

    /// <summary>Whether <paramref name="key"/> is an Idol block module's low-rank matrix. The MLP spelling <c>mlp.mlp.N</c> is what makes a key distinctive.</summary>
    public static bool IsDistinctive(string key)
    {
        ArgumentNullException.ThrowIfNull(key);
        return LoraRoleSuffix.TryStrip(key, out string root, out LoraRole _) && TryCanonicalBody(root, out string body)
            && (body.EndsWith(".mlp.mlp.0", StringComparison.Ordinal) || body.EndsWith(".mlp.mlp.2", StringComparison.Ordinal));
    }

    /// <summary>Parses every Idol block-module layer in the file onto <see cref="LoraTarget.Transformer"/>. A full-weight diff is refused by name.</summary>
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
                throw new NotSupportedException($"Idol LoRA key '{key}' is a full-weight diff. Only low-rank adapters are supported; "
                    + "applying the rest of this file would give wrong weights.");
            }
            LoraGroupBuffer.GetOrCreate(groups, LoraTarget.Transformer, body + ".weight", key).Assign(role, loader.GetTensor(key));
        }

        if (foreign > 0)
        {
            Logs.Warning($"Idol LoRA: skipped {foreign} keys outside the denoiser's block modules.");
        }
        IReadOnlyList<LoraLayer> layers = LoraGroupBuffer.BuildLayers(groups,
            sourceKey => $"Idol LoRA group '{sourceKey}' is missing a matrix its decomposition needs; skipping.");
        if (layers.Count == 0)
        {
            throw new NotSupportedException("Idol LoRA has no complete block-module layers, so it would change nothing. Check that the file "
                + "is a low-rank Idol adapter with both matrices of each pair.");
        }
        return layers;
    }

    /// <summary>Strips one accepted wrapper and checks the rest is <c>&lt;root&gt;N.&lt;module&gt;</c>; <paramref name="body"/> is that path, which is also the weight name without <c>.weight</c>.</summary>
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
        foreach (string stack in Roots)
        {
            if (!body.StartsWith(stack, StringComparison.Ordinal))
            {
                continue;
            }
            string rest = body[stack.Length..];
            int separator = rest.IndexOf('.', StringComparison.Ordinal);
            return separator > 0 && rest[..separator].All(char.IsAsciiDigit) && Modules.Contains(rest[(separator + 1)..]);
        }
        return false;
    }
}
