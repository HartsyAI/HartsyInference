namespace HartsyInference.ModelAssets.Lora;

/// <summary>One row of <see cref="FusedProjectionLayouts"/>: a SPLIT weight key suffix, the FUSED sibling that carries it, and which slice of that sibling's rows it occupies.</summary>
/// <param name="SplitSuffix">Canonical split-form suffix a LoRA layer targets, including the trailing <c>.weight</c>.</param>
/// <param name="FusedSuffix">The fused weight's suffix, replacing <paramref name="SplitSuffix"/> on the same root.</param>
/// <param name="SliceIndex">Zero-based slice this projection occupies within the fused weight's rows.</param>
/// <param name="SliceCount">How many equal row slices the fused weight is divided into.</param>
public readonly record struct FusedProjectionLayout(string SplitSuffix, string FusedSuffix, int SliceIndex, int SliceCount);

/// <summary>Which fused weights a split projection key can hide inside, as data rather than a chain of string comparisons.</summary>
/// <remarks><para>fp8 builds of the Flux-lineage checkpoints keep attention fused (<c>attn.qkv</c>,
/// <c>attn.add_qkv</c>) while a LoRA's canonical keys are the split names. Without this mapping every attention delta
/// missed on those builds, the stack still merged the non-attention weights and reported success, and the image came
/// out subtly under-LoRA'd with nothing to point at. Two families keep attention fused in their ORDINARY layout too:
/// Ideogram 4 (<c>layers.{i}.attention.qkv.weight</c>) and F-Lite (<c>blocks.{i}.qkv.weight</c>). MiniMax-H3 is the
/// third — <c>blocks.{i}.attn.qkv_proj.weight</c> — and is the one the GGUF rollout needs.</para>
/// <para>A miss is safe by construction: the fallback only fires when the split key is absent from the dictionary AND
/// the fused candidate is present, so it can never capture a key the direct lookup would have served.</para></remarks>
public static class FusedProjectionLayouts
{
    /// <summary>Every known split→fused mapping, matched by suffix in this order.</summary>
    public static IReadOnlyList<FusedProjectionLayout> All { get; } =
    [
        new(".attn.to_q.weight", ".attn.qkv.weight", 0, 3),
        new(".attn.to_k.weight", ".attn.qkv.weight", 1, 3),
        new(".attn.to_v.weight", ".attn.qkv.weight", 2, 3),
        new(".attn.add_q_proj.weight", ".attn.add_qkv.weight", 0, 3),
        new(".attn.add_k_proj.weight", ".attn.add_qkv.weight", 1, 3),
        new(".attn.add_v_proj.weight", ".attn.add_qkv.weight", 2, 3),
        // MiniMax-H3 keeps attention fused in every build; its own LoRAs name qkv_proj directly, a converted one does not.
        new(".attn.to_q.weight", ".attn.qkv_proj.weight", 0, 3),
        new(".attn.to_k.weight", ".attn.qkv_proj.weight", 1, 3),
        new(".attn.to_v.weight", ".attn.qkv_proj.weight", 2, 3),
        // Ideogram 4: layers.{i}.attention.qkv.weight
        new(".attention.to_q.weight", ".attention.qkv.weight", 0, 3),
        new(".attention.to_k.weight", ".attention.qkv.weight", 1, 3),
        new(".attention.to_v.weight", ".attention.qkv.weight", 2, 3),
        // F-Lite: blocks.{i}.qkv.weight — no attention segment at all.
        new(".to_q.weight", ".qkv.weight", 0, 3),
        new(".to_k.weight", ".qkv.weight", 1, 3),
        new(".to_v.weight", ".qkv.weight", 2, 3),
    ];

    /// <summary>Finds the fused sibling of <paramref name="canonicalKey"/> that <paramref name="containsKey"/> reports present, or false when none matches.</summary>
    public static bool TryResolve(string canonicalKey, Func<string, bool> containsKey,
        out string fusedKey, out int sliceIndex, out int sliceCount)
    {
        foreach (FusedProjectionLayout layout in All)
        {
            if (!canonicalKey.EndsWith(layout.SplitSuffix, StringComparison.Ordinal))
            {
                continue;
            }
            string candidate = canonicalKey[..^layout.SplitSuffix.Length] + layout.FusedSuffix;
            if (containsKey(candidate))
            {
                fusedKey = candidate;
                sliceIndex = layout.SliceIndex;
                sliceCount = layout.SliceCount;
                return true;
            }
        }
        fusedKey = string.Empty;
        sliceIndex = 0;
        sliceCount = 0;
        return false;
    }
}
