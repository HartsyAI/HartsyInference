using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
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
        Dictionary<string, GroupBuffer> groups = [];
        List<LoraFullWeightDiff> diffs = [];
        foreach (string key in loader.Descriptors.Keys)
        {
            if (!TryClassifyRole(key, out LoraRole role, out string root))
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
            if (!groups.TryGetValue(canonicalKey, out GroupBuffer? group))
            {
                group = new GroupBuffer { FirstSourceKey = key };
                groups[canonicalKey] = group;
            }
            switch (role)
            {
                case LoraRole.Down: group.Down = loader.GetTensor(key); break;
                case LoraRole.Up: group.Up = loader.GetTensor(key); break;
                case LoraRole.Alpha: group.Alpha = KohyaSdMapper.ReadScalar(loader.GetTensor(key)); break;
            }
        }

        List<LoraLayer> layers = new(groups.Count);
        foreach ((string canonicalKey, GroupBuffer group) in groups)
        {
            if (group.Down is null || group.Up is null)
            {
                Logs.Warning($"Wan LoRA group '{group.FirstSourceKey}' missing down or up; skipping.");
                continue;
            }
            int rank = (int)group.Down.Shape[0];
            layers.Add(new LoraLayer
            {
                TargetKey = canonicalKey,
                Target = LoraTarget.Transformer,
                LoraDown = group.Down,
                LoraUp = group.Up,
                Alpha = group.Alpha ?? rank,
                Rank = rank,
                Variant = LoraVariant.StandardLora,
            });
        }
        fullWeightDiffs = diffs;
        return layers;
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

    private static bool TryClassifyRole(string key, out LoraRole role, out string root)
    {
        foreach ((string suffix, LoraRole r) in _suffixRoles)
        {
            if (key.EndsWith(suffix, StringComparison.Ordinal))
            {
                role = r;
                root = key[..^suffix.Length];
                return true;
            }
        }
        role = default;
        root = string.Empty;
        return false;
    }

    private static readonly (string Suffix, LoraRole Role)[] _suffixRoles =
    [
        (".lora_down.weight", LoraRole.Down),
        (".lora_up.weight", LoraRole.Up),
        (".lora_A.weight", LoraRole.Down),
        (".lora_B.weight", LoraRole.Up),
        (".alpha", LoraRole.Alpha),
        (".diff", LoraRole.Diff),
        (".diff_b", LoraRole.BiasDiff),
    ];

    private enum LoraRole { Down, Up, Alpha, Diff, BiasDiff }

    private sealed class GroupBuffer
    {
        public required string FirstSourceKey { get; init; }
        public Tensor? Down { get; set; }
        public Tensor? Up { get; set; }
        public float? Alpha { get; set; }
    }
}
