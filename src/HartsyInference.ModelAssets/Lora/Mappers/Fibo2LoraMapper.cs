using HartsyInference.Core.Logging;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.ModelAssets.Lora.Mappers;

/// <summary>Parses Fibo 2.0 LoRA files onto the weight names the Comfy-Org checkpoint (and so <c>Fibo2Transformer</c>) uses.
/// <para>Two spellings exist. The ComfyUI-native one is what Comfy-Org's published turbo LoRA
/// (<c>fibo_2.0_turbo_lora_avg_rank_270_bf16</c>) uses: <c>diffusion_model.{module}.lora_down/lora_up/alpha</c> with
/// every adapter on the FUSED module (<c>attention.qkv</c> [3·dim], <c>cross_attention.to_kv</c> [2·dim],
/// <c>feed_forward.w13</c> [2·H]), plus full-weight <c>.diff</c> deltas on the RMSNorm weights and <c>.diff_b</c> on the
/// biases. Those names already equal the checkpoint's, so nothing is split or renamed. The diffusers/PEFT one
/// (<c>transformer.</c> wrapper, the unfused <c>to_q/to_k/to_v</c>, <c>w1/w3</c> and <c>to_out.0</c> of
/// <c>transformer_bria_fibo2.py</c>) is renamed to the checkpoint's modules; its split projections keep their split names and
/// <see cref="FusedProjectionLayouts"/> resolves them to the right row slice of the fused weight at merge time.</para></summary>
public static class Fibo2LoraMapper
{
    /// <summary>Dotted wrappers in front of a module path, longest first; a root carrying none is the module path itself.</summary>
    private static readonly string[] Wrappers = ["model.diffusion_model.", "diffusion_model.", "transformer."];

    /// <summary>Parses every Fibo 2.0 LoRA layer in the file; full-weight <c>.diff</c>/<c>.diff_b</c> entries come out via <paramref name="fullWeightDiffs"/>.</summary>
    public static IReadOnlyList<LoraLayer> ParseLayers(SafeTensorsLoader loader, out IReadOnlyList<LoraFullWeightDiff> fullWeightDiffs)
    {
        ArgumentNullException.ThrowIfNull(loader);
        Dictionary<(LoraTarget, string), LoraGroupBuffer> groups = [];
        List<LoraFullWeightDiff> diffs = [];
        foreach (string key in loader.Descriptors.Keys)
        {
            if (LoraRoleSuffix.IsPeftDoraMagnitude(key))
            {
                throw new NotSupportedException($"Fibo 2.0 LoRA key '{key}' is a PEFT DoRA magnitude vector. PEFT DoRA adapters "
                    + "are not supported, and applying their A/B pairs without it would give wrong weights.");
            }
            if (!LoraRoleSuffix.TryStrip(key, out string root, out LoraRole role))
            {
                continue;
            }
            string body = StripWrapper(root);
            if (role is LoraRole.Diff or LoraRole.BiasDiff)
            {
                bool bias = role == LoraRole.BiasDiff;
                diffs.Add(new LoraFullWeightDiff
                {
                    TargetKey = MapBodyToCanonical(body, bias ? ".bias" : ".weight"),
                    Target = LoraTarget.Transformer,
                    Diff = loader.GetTensor(key),
                    IsBias = bias,
                });
                continue;
            }
            if (body.StartsWith("lora_", StringComparison.Ordinal))
            {
                Logs.Warning($"Fibo 2.0 LoRA key '{key}' is an underscored (kohya) name; it is not mapped by this format.");
                continue;
            }
            LoraGroupBuffer.GetOrCreate(groups, LoraTarget.Transformer, MapBodyToCanonical(body, ".weight"), key)
                .Assign(role, loader.GetTensor(key));
        }
        fullWeightDiffs = diffs;
        return LoraGroupBuffer.BuildLayers(groups,
            sourceKey => $"Fibo 2.0 LoRA group '{sourceKey}' is missing a matrix its decomposition needs; skipping.");
    }

    /// <summary>Maps a dotted module body (wrapper and role suffix already stripped) to the checkpoint weight key (pure, testable).
    /// The only renames are diffusers' <c>to_out.0</c> projections: <c>attention.to_out.0</c> is the checkpoint's
    /// <c>attention.out</c>, and the cross/self/Perceiver attentions' <c>to_out.0</c> is <c>to_out</c>. Diffusers' <c>norm_q/norm_k</c>
    /// of the main attention are the checkpoint's <c>q_norm/k_norm</c>.</summary>
    public static string MapBodyToCanonical(string body, string suffix)
    {
        ArgumentNullException.ThrowIfNull(body);
        const string MainOut = ".attention.to_out.0";
        const string OtherOut = ".to_out.0";
        string mapped = body;
        if (mapped.EndsWith(MainOut, StringComparison.Ordinal))
            mapped = string.Concat(mapped.AsSpan(0, mapped.Length - MainOut.Length), ".attention.out");
        else if (mapped.EndsWith(OtherOut, StringComparison.Ordinal))
            mapped = string.Concat(mapped.AsSpan(0, mapped.Length - OtherOut.Length), ".to_out");
        if (mapped.EndsWith(".attention.norm_q", StringComparison.Ordinal))
            mapped = string.Concat(mapped.AsSpan(0, mapped.Length - ".norm_q".Length), ".q_norm");
        else if (mapped.EndsWith(".attention.norm_k", StringComparison.Ordinal))
            mapped = string.Concat(mapped.AsSpan(0, mapped.Length - ".norm_k".Length), ".k_norm");
        return mapped + suffix;
    }

    /// <summary>Whether the LoRA file key <paramref name="key"/> adapts a module only Fibo 2.0 has — the detector's marker. These are the
    /// Perceiver resampler, the gated cross-attention of the injection blocks, and the fused SwiGLU <c>w13</c>.</summary>
    public static bool IsDistinctive(string key)
    {
        ArgumentNullException.ThrowIfNull(key);
        if (!LoraRoleSuffix.TryStrip(key, out string root, out LoraRole _))
            return false;
        string body = StripWrapper(root);
        return body.StartsWith("perceiver.layers.", StringComparison.Ordinal)
            || (body.StartsWith("layers.", StringComparison.Ordinal) && body.Contains(".cross_attention.", StringComparison.Ordinal))
            || (body.StartsWith("noise_refiner.", StringComparison.Ordinal) && body.EndsWith(".feed_forward.w13", StringComparison.Ordinal))
            || (body.StartsWith("layers.", StringComparison.Ordinal) && body.EndsWith(".feed_forward.w13", StringComparison.Ordinal));
    }

    private static string StripWrapper(string root)
    {
        foreach (string wrapper in Wrappers)
        {
            if (root.StartsWith(wrapper, StringComparison.Ordinal))
                return root[wrapper.Length..];
        }
        return root;
    }
}
