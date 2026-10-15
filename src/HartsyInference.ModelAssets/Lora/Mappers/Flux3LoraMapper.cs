using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.CheckpointConverters.Utils;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.ModelAssets.Lora.Mappers;

/// <summary>Parses FLUX.3 LoRA files onto the BFL weight keys <c>Flux3Transformer</c> loads, in each spelling a trainer
/// emits: diffusers/PEFT under <c>transformer.</c> (SimpleTuner, including its <c>extract_peft_lora.py</c> extracts,
/// e.g. <c>terminusresearch/flux-3-klein-image-distilled-lora-extract</c>), BFL names under <c>diffusion_model.</c> or
/// bare (ComfyUI's own key map), and kohya <c>lora_unet_</c>. Names resolve through <see cref="Flux3KeyNames"/>.
/// <para>Two layouts need more than a rename. An adapter on a fused <c>qkv</c> is spread over the split
/// <c>q_proj</c>/<c>k_proj</c>/<c>v_proj</c> the engine loads (up rows in thirds, down and alpha shared), as ComfyUI
/// addresses the halves of its own fused weight. Diffusers' <c>norm_out.linear</c> rows are <c>[scale | shift]</c> where
/// the BFL weight's are <c>[shift | scale]</c>, so its up matrix is reordered into a copy that
/// <paramref name="ownedTensors"/> keeps alive for the file's lifetime.</para></summary>
public static class Flux3LoraMapper
{
    private const string KohyaPrefix = "lora_unet_";

    /// <summary>Dotted wrappers in front of a module path; a root carrying neither is the module path itself.</summary>
    private static readonly string[] DottedWrappers = ["transformer.", "diffusion_model."];

    /// <summary>Parses every FLUX.3 layer in the file. Tensors the mapper allocates are added to <paramref name="ownedTensors"/>, which the caller disposes with the file.</summary>
    public static IReadOnlyList<LoraLayer> ParseLayers(SafeTensorsLoader loader, ICollection<Tensor> ownedTensors)
    {
        ArgumentNullException.ThrowIfNull(loader);
        ArgumentNullException.ThrowIfNull(ownedTensors);
        Dictionary<(LoraTarget, string), LoraGroupBuffer> groups = [];
        foreach (string key in loader.Descriptors.Keys)
        {
            if (!LoraRoleSuffix.TryStrip(key, out string root, out LoraRole role))
            {
                continue;
            }
            if (role is LoraRole.Diff or LoraRole.BiasDiff)
            {
                Logs.Warning($"FLUX.3 LoRA key '{key}' is a full-weight diff, which this format does not carry; skipping.");
                continue;
            }
            if (!TryResolve(root, out string module, out bool diffusersSpelling))
            {
                Logs.Warning($"FLUX.3 LoRA key '{key}' does not name a FLUX.3 transformer module; skipping.");
                continue;
            }

            Tensor tensor = loader.GetTensor(key);
            const string fusedLeaf = "." + Flux3KeyNames.FusedQkvLeaf;
            if (module.EndsWith(fusedLeaf, StringComparison.Ordinal))
            {
                AddFusedQkv(groups, module[..^fusedLeaf.Length], key, role, tensor);
                continue;
            }
            if (diffusersSpelling && module == Flux3KeyNames.FinalAdaLn && role is not (LoraRole.Down or LoraRole.Alpha))
            {
                if (role != LoraRole.Up)
                {
                    Logs.Warning($"FLUX.3 LoRA key '{key}' carries a {role} matrix on diffusers' [scale | shift] "
                        + "norm_out.linear, whose rows this mapper cannot reorder; skipping.");
                    continue;
                }
                tensor = CheckpointConvertUtils.SwapScaleShiftHalves(tensor);
                ownedTensors.Add(tensor);
            }
            LoraGroupBuffer.GetOrCreate(groups, LoraTarget.Transformer, module + ".weight", key).Assign(role, tensor);
        }

        return LoraGroupBuffer.BuildLayers(groups,
            sourceKey => $"FLUX.3 LoRA group '{sourceKey}' is missing a matrix its decomposition needs; skipping.");
    }

    /// <summary>Resolves a LoRA root (role suffix already stripped) to the BFL module it adapts, reporting whether it was
    /// spelled the diffusers way (pure, testable). False when the root is not a FLUX.3 transformer module.</summary>
    public static bool TryResolve(string root, out string module, out bool diffusersSpelling)
    {
        ArgumentNullException.ThrowIfNull(root);
        diffusersSpelling = false;
        string? resolved;
        if (root.StartsWith(KohyaPrefix, StringComparison.Ordinal))
        {
            resolved = Flux3KeyNames.FromKohya(root[KohyaPrefix.Length..]);
        }
        else
        {
            string body = StripDottedWrapper(root);
            resolved = Flux3KeyNames.FromDiffusers(body);
            diffusersSpelling = resolved is not null;
            if (resolved is null && Flux3KeyNames.IsModule(body))
            {
                resolved = body;
            }
        }
        module = resolved ?? string.Empty;
        return resolved is not null;
    }

    /// <summary>Whether the LoRA file key <paramref name="key"/> adapts a module only FLUX.3 has — the detector's marker.
    /// Keys every Flux shares (diffusers <c>single_transformer_blocks.{i}.attn.to_q</c>, say) are not evidence.</summary>
    public static bool IsDistinctive(string key)
    {
        ArgumentNullException.ThrowIfNull(key);
        if (!LoraRoleSuffix.TryStrip(key, out string root, out LoraRole _))
        {
            return false;
        }
        if (root.StartsWith(KohyaPrefix, StringComparison.Ordinal))
        {
            string? module = Flux3KeyNames.FromKohya(root[KohyaPrefix.Length..]);
            return module is not null && Flux3KeyNames.IsDistinctiveBfl(module);
        }
        string body = StripDottedWrapper(root);
        return Flux3KeyNames.IsDistinctiveDiffusers(body) || Flux3KeyNames.IsDistinctiveBfl(body);
    }

    /// <summary>Spreads a fused-<c>qkv</c> adapter over the split projections, refusing roles that cannot be cut by rows.</summary>
    private static void AddFusedQkv(Dictionary<(LoraTarget, string), LoraGroupBuffer> groups, string blockPrefix,
        string sourceKey, LoraRole role, Tensor tensor)
    {
        if (role == LoraRole.Up && (tensor.Shape.Rank != 2 || tensor.Shape[0] % 3 != 0))
        {
            Logs.Warning($"FLUX.3 LoRA fused qkv up matrix '{sourceKey}' has shape {tensor.Shape}, not [3·D, rank]; skipping.");
            return;
        }
        long rows = role == LoraRole.Up ? tensor.Shape[0] / 3 : 0;
        (string CanonicalKey, long Rows)[] slices = new (string CanonicalKey, long Rows)[Flux3KeyNames.SplitQkvLeaves.Count];
        for (int i = 0; i < slices.Length; i++)
        {
            slices[i] = ($"{blockPrefix}.{Flux3KeyNames.SplitQkvLeaves[i]}.weight", rows);
        }
        if (!LoraGroupBuffer.AssignFusedRows(groups, LoraTarget.Transformer, slices, sourceKey, role, tensor))
        {
            Logs.Warning($"FLUX.3 LoRA key '{sourceKey}' carries a {role} matrix on a fused qkv, which this mapper "
                + "cannot split; skipping.");
        }
    }

    private static string StripDottedWrapper(string root)
    {
        foreach (string wrapper in DottedWrappers)
        {
            if (root.StartsWith(wrapper, StringComparison.Ordinal))
            {
                return root[wrapper.Length..];
            }
        }
        return root;
    }
}
