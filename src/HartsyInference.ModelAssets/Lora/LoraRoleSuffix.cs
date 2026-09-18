namespace HartsyInference.ModelAssets.Lora;

/// <summary>The one place a LoRA file key's trailing role marker is recognized. Every mapper and <see cref="LoraFormatDetector"/> classify through <see cref="TryStrip"/>, so the suffix vocabulary — PEFT, kohya, LyCORIS LoHa/LoKr, DoRA and Comfy full-weight diffs — is shared rather than re-spelled per format. That is what makes a LyCORIS file detectable for every family without any per-mapper code: the roots a mapper already knows how to canonicalize are orthogonal to which decomposition the file stores.</summary>
public static class LoraRoleSuffix
{
    // Longest-first within each family. Ordinal EndsWith already separates every pair here (".lokr_w1" does not
    // match a key ending ".lokr_w1_a"), but the order is kept explicit so adding a suffix that IS a tail of another
    // cannot silently bind to the shorter one.
    private static readonly (string Suffix, LoraRole Role)[] _suffixes =
    [
        (".lora_A.default.weight", LoraRole.Down),
        (".lora_B.default.weight", LoraRole.Up),
        (".lora_A.weight", LoraRole.Down),
        (".lora_B.weight", LoraRole.Up),
        (".lora_down.weight", LoraRole.Down),
        (".lora_up.weight", LoraRole.Up),
        (".hada_w1_a", LoraRole.HadaW1A),
        (".hada_w1_b", LoraRole.HadaW1B),
        (".hada_w2_a", LoraRole.HadaW2A),
        (".hada_w2_b", LoraRole.HadaW2B),
        (".hada_t1", LoraRole.HadaT1),
        (".hada_t2", LoraRole.HadaT2),
        (".lokr_w1_a", LoraRole.LokrW1A),
        (".lokr_w1_b", LoraRole.LokrW1B),
        (".lokr_w2_a", LoraRole.LokrW2A),
        (".lokr_w2_b", LoraRole.LokrW2B),
        (".lokr_w1", LoraRole.LokrW1),
        (".lokr_w2", LoraRole.LokrW2),
        (".lokr_t2", LoraRole.LokrT2),
        (".dora_scale", LoraRole.DoraScale),
        (".alpha", LoraRole.Alpha),
        (".diff_b", LoraRole.BiasDiff),
        (".diff", LoraRole.Diff),
    ];

    /// <summary>Splits <paramref name="key"/> into the module <paramref name="root"/> and the <paramref name="role"/> its suffix names; false when the key carries no recognized role, in which case the outputs are unset.</summary>
    public static bool TryStrip(string key, out string root, out LoraRole role)
    {
        foreach ((string suffix, LoraRole candidate) in _suffixes)
        {
            if (key.EndsWith(suffix, StringComparison.Ordinal))
            {
                role = candidate;
                root = key[..^suffix.Length];
                return true;
            }
        }
        role = default;
        root = string.Empty;
        return false;
    }

    /// <summary>Whether <paramref name="role"/> is a decomposition matrix rather than a scalar companion (<see cref="LoraRole.Alpha"/>, <see cref="LoraRole.DoraScale"/>) or a full-weight delta (<see cref="LoraRole.Diff"/>, <see cref="LoraRole.BiasDiff"/>). Format detection keys on this: a stray alpha or magnitude vector is not by itself evidence that a file is a LoRA for the root it sits on.</summary>
    public static bool IsDecompositionMatrix(LoraRole role) =>
        role is not (LoraRole.Alpha or LoraRole.DoraScale or LoraRole.Diff or LoraRole.BiasDiff);
}
