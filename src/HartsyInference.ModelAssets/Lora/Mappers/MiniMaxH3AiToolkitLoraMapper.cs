using HartsyInference.Core.Logging;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.ModelAssets.Lora.Mappers;

/// <summary>Parses MiniMax-H3 LoRA files trained with ai-toolkit (<c>diffusion_model.blocks.N.attn.qkv_proj.lora_A.weight</c>, and the
/// <c>token_refiner.blocks.N.…</c> stream). The file names modules as the H3 DiT does, so after the wrapper is stripped each body is
/// already the transformer's canonical weight name and passes through with no rename.
/// <para>Detection claims a file only by its attention names (<see cref="IsDistinctive"/>); the MLP names are common to other
/// families. Once claimed, every block module passes through, so a module the DiT does not hold is counted as a missing key in the
/// merge report and is decided by <c>lora.partialApplication</c>, not dropped here.</para></summary>
public static class MiniMaxH3AiToolkitLoraMapper
{
    /// <summary>The wrapper ai-toolkit puts in front of the checkpoint's own module path.</summary>
    private const string Wrapper = "diffusion_model.";

    /// <summary>The block roots an H3 DiT uses: the main stream, then the two token-refiner blocks.</summary>
    private static readonly string[] BlockRoots = ["blocks.", "token_refiner.blocks."];

    /// <summary>The attention modules that identify the family. Only H3 among the families here fuses qkv under these names.</summary>
    private static readonly HashSet<string> DistinctiveModules = new(StringComparer.Ordinal)
    {
        "attn.qkv_proj", "attn.out_proj",
    };

    /// <summary>Whether <paramref name="key"/> names an H3 attention module under the ai-toolkit <c>diffusion_model.</c> wrapper. A bare
    /// root (the published Turbo LoRA's <c>blocks.0.attn.qkv_proj</c>) is deliberately not claimed here: it stays on the bare-DiT path.</summary>
    public static bool IsDistinctive(string key)
    {
        ArgumentNullException.ThrowIfNull(key);
        return key.StartsWith(Wrapper, StringComparison.Ordinal)
            && LoraRoleSuffix.TryStrip(key, out string root, out LoraRole _)
            && TryCanonicalBody(root, out string body)
            && DistinctiveModules.Contains(ModuleOf(body));
    }

    /// <summary>Parses every H3 block LoRA layer in the file onto <see cref="LoraTarget.Transformer"/>. A full-weight diff is refused by
    /// name, since applying the rest of the file would then give wrong weights.</summary>
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
                throw new NotSupportedException($"MiniMax-H3 LoRA key '{key}' is a full-weight diff. Only low-rank adapters are supported; "
                    + "applying the rest of this file would give wrong weights.");
            }
            LoraGroupBuffer.GetOrCreate(groups, LoraTarget.Transformer, body + ".weight", key).Assign(role, loader.GetTensor(key));
        }

        if (foreign > 0)
        {
            Logs.Warning($"MiniMax-H3 LoRA: skipped {foreign} keys outside the DiT block stacks.");
        }
        IReadOnlyList<LoraLayer> layers = LoraGroupBuffer.BuildLayers(groups,
            sourceKey => $"MiniMax-H3 LoRA group '{sourceKey}' is missing a matrix its decomposition needs; skipping.");
        // A file the detector claimed but that yields no complete layer would otherwise load as an empty LoRA and generate unchanged.
        if (layers.Count == 0)
        {
            throw new NotSupportedException("MiniMax-H3 LoRA has no complete block-module layers, so it would change nothing. Check that the file "
                + "is a low-rank MiniMax-H3 adapter with both matrices of each pair.");
        }
        return layers;
    }

    /// <summary>Strips the optional ai-toolkit wrapper and checks the rest is a numbered block of a block stack, returning that
    /// module path (<c>blocks.N.attn.qkv_proj</c>) as <paramref name="body"/>. Any module name is accepted here.</summary>
    private static bool TryCanonicalBody(string root, out string body)
    {
        body = root.StartsWith(Wrapper, StringComparison.Ordinal) ? root[Wrapper.Length..] : root;
        foreach (string blockRoot in BlockRoots)
        {
            if (!body.StartsWith(blockRoot, StringComparison.Ordinal))
            {
                continue;
            }
            string rest = body[blockRoot.Length..];
            int separator = rest.IndexOf('.', StringComparison.Ordinal);
            return separator > 0 && rest[..separator].All(char.IsAsciiDigit) && separator + 1 < rest.Length;
        }
        return false;
    }

    /// <summary>The module path inside a block (<c>attn.qkv_proj</c> for <c>blocks.0.attn.qkv_proj</c>), for a body already checked by
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
