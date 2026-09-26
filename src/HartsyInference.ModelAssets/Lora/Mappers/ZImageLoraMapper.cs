using HartsyInference.Core.Logging;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.ModelAssets.Lora.Mappers;

/// <summary>Parses Z-Image (Tongyi Lumina2/NextDiT) LoRA files into canonical <see cref="LoraTarget.Transformer"/>
/// keys — Comfy-Org's <c>z_image_turbo_distill_patch_lora_bf16.safetensors</c> and anything sharing its vocabulary:
/// <c>diffusion_model.{layers|context_refiner|noise_refiner}.{i}.{attention.to_q|attention.to_k|attention.to_v|attention.to_out.0|feed_forward.w1|feed_forward.w2|feed_forward.w3}</c>
/// with PEFT suffixes. The checkpoint's own naming is already the block body's (ZImageCheckpointConverter is
/// passthrough), so the mapper strips the wrapper and renames only where the LoRA's diffusers spelling differs from
/// the Tongyi module: <c>attention.to_out.0 → attention.out</c>.
/// <para>Q/K/V deliberately keep their SPLIT names. The checkpoint stores them fused as
/// <c>attention.qkv.weight</c> ([3·hidden, hidden], contiguous Q|K|V rows — <c>ZImageBlock.SplitQkv</c>), and
/// <see cref="FusedProjectionLayouts"/> resolves the split key to the right row slice at merge time, the same way
/// fp8 Flux and MiniMax-H3 attention deltas land.</para></summary>
public static class ZImageLoraMapper
{
    private const string ComfyPrefix = "diffusion_model.";

    /// <summary>Parses every LoRA layer in the file.</summary>
    public static IReadOnlyList<LoraLayer> ParseLayers(SafeTensorsLoader loader)
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
                Logs.Warning($"Z-Image LoRA key '{key}' is a full-weight diff, which this format does not carry; skipping.");
                continue;
            }
            if (!root.StartsWith(ComfyPrefix, StringComparison.Ordinal))
            {
                Logs.Warning($"Z-Image LoRA key '{key}' has an unrecognized prefix; skipping.");
                continue;
            }

            string canonicalKey = MapBodyToCanonical(root[ComfyPrefix.Length..]);
            LoraGroupBuffer group = LoraGroupBuffer.GetOrCreate(groups, LoraTarget.Transformer, canonicalKey, key);
            group.Assign(role, loader.GetTensor(key));
        }

        return LoraGroupBuffer.BuildLayers(groups,
            sourceKey => $"Z-Image LoRA group '{sourceKey}' is missing a matrix its decomposition needs; skipping.");
    }

    /// <summary>Maps a dotted module body (wrapper prefix and role suffix already stripped) to the canonical
    /// <c>ZImageTransformer</c> weight key (pure, testable). Everything but the attention output projection is
    /// already the checkpoint's own spelling and passes through.</summary>
    public static string MapBodyToCanonical(string body)
    {
        const string SplitOut = ".attention.to_out.0";
        string mapped = body.EndsWith(SplitOut, StringComparison.Ordinal)
            ? string.Concat(body.AsSpan(0, body.Length - SplitOut.Length), ".attention.out")
            : body;
        return mapped + ".weight";
    }
}
