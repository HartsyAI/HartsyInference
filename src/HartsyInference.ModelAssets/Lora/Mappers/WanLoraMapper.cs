using HartsyInference.Core.Logging;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.ModelAssets.Lora.Mappers;

/// <summary>Parses Wan-family LoRA files (Wan2.1 / Wan2.2 DiTs incl. TI2V-5B and Matrix-Game finetunes) into canonical <c>WanVideoTransformer</c> keys. Two on-disk formats route here:
/// <list type="bullet">
/// <item><see cref="LoraFormat.KohyaWan"/> — kohya/musubi-tuner: <c>lora_unet_blocks_{i}_self_attn_q.lora_down.weight</c> (underscored original Wan naming; <see cref="LoraKeyTransformer.UnderscoreToDot"/> restores the dots).</item>
/// <item><see cref="LoraFormat.DiffusersWan"/> — ComfyUI-style repacks (lightx2v distills, Kijai conversions): <c>diffusion_model.blocks.{i}.self_attn.q.lora_A.weight</c> (dotted original naming), with either PEFT or kohya suffixes.</item>
/// </list>
/// Both end in <b>original Wan module naming</b>, mapped to the diffusers-style canonical keys via the same verbatim
/// rename table the checkpoint converter uses (<see cref="WanVideoCheckpointConverter.MapKey"/> — <c>self_attn.q →
/// attn1.to_q</c>, <c>ffn.0 → ffn.net.0.proj</c>, <c>cross_attn.k_img → attn2.add_k_proj</c>, …). Bodies already in
/// diffusers naming pass through untouched, so mixed-era repacks load too. (Diffusers-PEFT Wan LoRAs with a
/// <c>transformer.</c> prefix never reach this mapper — they ride the existing passthrough arm.) Comfy full-weight
/// diff entries (<c>.diff</c> targets the module's .weight, <c>.diff_b</c> its .bias) are not low-rank and come out
/// as <see cref="LoraFullWeightDiff"/> entries instead of layers.</summary>
public static class WanLoraMapper
{
    private const string KohyaPrefix = "lora_unet_";
    private const string ComfyPrefix = "diffusion_model.";

    /// <summary>Parses every LoRA layer in the file; full-weight <c>.diff</c>/<c>.diff_b</c> entries come out via <paramref name="fullWeightDiffs"/>.</summary>
    public static IReadOnlyList<LoraLayer> ParseLayers(SafeTensorsLoader loader, LoraFormat format,
        out IReadOnlyList<LoraFullWeightDiff> fullWeightDiffs)
    {
        Dictionary<(LoraTarget, string), LoraGroupBuffer> groups = [];
        List<LoraFullWeightDiff> diffs = [];
        foreach (string key in loader.Descriptors.Keys)
        {
            if (!LoraRoleSuffix.TryStrip(key, out string root, out LoraRole role))
                continue;

            string? body = format switch
            {
                LoraFormat.KohyaWan when root.StartsWith(KohyaPrefix, StringComparison.Ordinal) =>
                    LoraKeyTransformer.UnderscoreToDot(root[KohyaPrefix.Length..]),
                LoraFormat.DiffusersWan when root.StartsWith(ComfyPrefix, StringComparison.Ordinal) =>
                    root[ComfyPrefix.Length..],
                // Bare original naming (no wrapper prefix) — e.g. the Wan-Animate relight conversion.
                LoraFormat.DiffusersWan when root.StartsWith("blocks.", StringComparison.Ordinal) => root,
                _ => null,
            };
            if (body is null)
            {
                Logs.Warning($"Wan LoRA key '{key}' has an unrecognized prefix for format {format}; skipping.");
                continue;
            }

            if (role is LoraRole.Diff or LoraRole.BiasDiff)
            {
                diffs.Add(new LoraFullWeightDiff
                {
                    TargetKey = MapBodyToCanonical(body, role == LoraRole.BiasDiff ? ".bias" : ".weight"),
                    Target = LoraTarget.Transformer,
                    Diff = loader.GetTensor(key),
                    IsBias = role == LoraRole.BiasDiff,
                });
                continue;
            }

            string canonicalKey = MapBodyToCanonical(body);
            LoraGroupBuffer group = LoraGroupBuffer.GetOrCreate(groups, LoraTarget.Transformer, canonicalKey, key);
            group.Assign(role, loader.GetTensor(key));
        }

        fullWeightDiffs = diffs;
        return LoraGroupBuffer.BuildLayers(groups,
            sourceKey => $"Wan LoRA group '{sourceKey}' is missing a matrix its decomposition needs; skipping.");
    }

    /// <summary>Maps a dotted module body to the canonical <c>WanVideoTransformer</c> weight key (pure, testable).
    /// Bodies in original Wan naming get the checkpoint converter's verbatim rename table; bodies already in
    /// diffusers naming pass through. Detection is per-body via the unambiguous original-naming markers — a body
    /// ending in <c>norm3</c> can only be original naming (diffusers norm3 is non-affine), so it safely rides the
    /// converter's ordered <c>norm2⇄norm3</c> swap; <c>patch_embedding</c> matches no rename rule and passes through
    /// either way.</summary>
    public static string MapBodyToCanonical(string body) => MapBodyToCanonical(body, ".weight");

    /// <summary>As <see cref="MapBodyToCanonical(string)"/> with an explicit trailing suffix — <c>.bias</c> for
    /// <c>.diff_b</c> targets.</summary>
    public static string MapBodyToCanonical(string body, string suffix)
    {
        bool original = body.Contains("self_attn.", StringComparison.Ordinal)
            || body.Contains("cross_attn.", StringComparison.Ordinal)
            || body.EndsWith(".ffn.0", StringComparison.Ordinal) || body.EndsWith(".ffn.2", StringComparison.Ordinal)
            || body.EndsWith(".norm3", StringComparison.Ordinal)
            || body.StartsWith("img_emb.", StringComparison.Ordinal)
            || body.StartsWith("time_embedding.", StringComparison.Ordinal)
            || body.StartsWith("text_embedding.", StringComparison.Ordinal)
            || body.StartsWith("time_projection.", StringComparison.Ordinal)
            || body.StartsWith("head.", StringComparison.Ordinal);
        string mapped = WanVideoCheckpointConverter.MapKey(body, original) ?? body;
        return mapped + suffix;
    }
}
