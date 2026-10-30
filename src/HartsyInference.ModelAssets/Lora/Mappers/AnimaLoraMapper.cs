using HartsyInference.Core.Logging;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.ModelAssets.Lora.Mappers;

/// <summary>Parses Anima LoRA files onto the checkpoint's own DiT weight names (<c>blocks.N.self_attn.q_proj.weight</c>,
/// <c>blocks.N.mlp.layer1.weight</c>, …), which <see cref="AnimaCheckpointConverter"/> produces and <c>AnimaBlock</c> reads.
/// <para>Anima's modules are already named that way, so the body passes through with no rename. The Wan mapper's rename
/// table would turn <c>self_attn.q_proj</c> into <c>attn1.to_q_proj</c> and <c>adaln_modulation_*</c> into
/// <c>scale_shift_table</c>, so the file applied only its MLP keys and nothing said so. Accepted spellings: the ComfyUI
/// wrappers <c>diffusion_model.</c> and <c>model.diffusion_model.</c>, the checkpoint's own <c>net.</c> root, and a bare
/// <c>blocks.</c> root. The underscored kohya spelling is routed to <see cref="KohyaGenericMapper"/> instead (see
/// <see cref="IsKohyaSpelling"/>), which resolves names against the target weights.</para></summary>
public static class AnimaLoraMapper
{
    /// <summary>Root wrappers in front of the <c>blocks.N.</c> body; a root carrying none is the body itself.</summary>
    private static readonly string[] Wrappers = ["model.diffusion_model.", "diffusion_model.", "net."];

    /// <summary>The module paths a block of the Anima DiT owns, relative to <c>blocks.N.</c>: the attention projections, the
    /// two-layer FFN, and the three adaLN modulation MLPs (<c>AnimaBlock.LoadWeights</c>).</summary>
    private static readonly HashSet<string> Modules = new(StringComparer.Ordinal)
    {
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.output_proj",
        "cross_attn.q_proj", "cross_attn.k_proj", "cross_attn.v_proj", "cross_attn.output_proj",
        "mlp.layer1", "mlp.layer2",
        "adaln_modulation_self_attn.1", "adaln_modulation_self_attn.2",
        "adaln_modulation_cross_attn.1", "adaln_modulation_cross_attn.2",
        "adaln_modulation_mlp.1", "adaln_modulation_mlp.2",
    };

    /// <summary>The same modules in kohya's spelling, where every <c>.</c> is written <c>_</c>.</summary>
    private static readonly HashSet<string> UnderscoredModules = [.. Modules.Select(module => module.Replace('.', '_'))];

    private const string KohyaPrefix = "lora_unet_blocks_";

    /// <summary>Whether <paramref name="key"/> names an Anima block module in the dotted spellings this mapper reads.</summary>
    public static bool IsDistinctive(string key)
    {
        ArgumentNullException.ThrowIfNull(key);
        return LoraRoleSuffix.TryStrip(key, out string root, out LoraRole _) && TryCanonicalBody(root, out _);
    }

    /// <summary>Whether <paramref name="key"/> is the kohya spelling of an Anima block module (<c>lora_unet_blocks_0_self_attn_q_proj.lora_down.weight</c>).
    /// Such a file is parsed by <see cref="KohyaGenericMapper"/>, which resolves the underscored names against the transformer's weights.</summary>
    public static bool IsKohyaSpelling(string key)
    {
        ArgumentNullException.ThrowIfNull(key);
        if (!LoraRoleSuffix.TryStrip(key, out string root, out LoraRole _) || !root.StartsWith(KohyaPrefix, StringComparison.Ordinal))
        {
            return false;
        }
        string rest = root[KohyaPrefix.Length..];
        int separator = rest.IndexOf('_', StringComparison.Ordinal);
        return separator > 0 && rest[..separator].All(char.IsAsciiDigit) && UnderscoredModules.Contains(rest[(separator + 1)..]);
    }

    /// <summary>Parses every Anima block LoRA layer in the file onto <see cref="LoraTarget.Transformer"/>. A full-weight
    /// diff is refused by name: it is not a low-rank adapter, so applying the rest of the file would give wrong weights.</summary>
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
                throw new NotSupportedException($"Anima LoRA key '{key}' is a full-weight diff. Only low-rank adapters are supported; "
                    + "applying the rest of this file would give wrong weights.");
            }
            LoraGroupBuffer.GetOrCreate(groups, LoraTarget.Transformer, body + ".weight", key).Assign(role, loader.GetTensor(key));
        }

        if (foreign > 0)
        {
            Logs.Warning($"Anima LoRA: skipped {foreign} keys outside the DiT block modules (text-encoder adapters are not applied for this format).");
        }
        IReadOnlyList<LoraLayer> layers = LoraGroupBuffer.BuildLayers(groups,
            sourceKey => $"Anima LoRA group '{sourceKey}' is missing a matrix its decomposition needs; skipping.");
        // A file the format detector claimed but that yields no complete layer would otherwise load as an empty LoRA and generate unchanged.
        if (layers.Count == 0)
        {
            throw new NotSupportedException("Anima LoRA has no complete block-module layers, so it would change nothing. Check that the file "
                + "is a low-rank Anima adapter with both matrices of each pair.");
        }
        return layers;
    }

    /// <summary>Strips one accepted wrapper and checks the rest is <c>blocks.N.&lt;module&gt;</c> for a module an Anima block owns.
    /// <paramref name="body"/> is that <c>blocks.N.&lt;module&gt;</c> path, which is also the checkpoint's weight name without <c>.weight</c>.</summary>
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
        const string blocks = "blocks.";
        if (!body.StartsWith(blocks, StringComparison.Ordinal))
        {
            return false;
        }
        string rest = body[blocks.Length..];
        int separator = rest.IndexOf('.', StringComparison.Ordinal);
        return separator > 0 && rest[..separator].All(char.IsAsciiDigit) && Modules.Contains(rest[(separator + 1)..]);
    }
}
