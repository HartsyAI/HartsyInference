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

    /// <summary>Whether <paramref name="key"/> is a PEFT DoRA magnitude vector (<c>.lora_magnitude_vector</c>). PEFT's DoRA
    /// renormalizes differently from the LyCORIS <c>dora_scale</c> this vocabulary carries, and its magnitude is not
    /// additive, so a consumer refuses the adapter rather than applying its A/B pair alone.</summary>
    public static bool IsPeftDoraMagnitude(string key) =>
        key.Contains(".lora_magnitude_vector", StringComparison.Ordinal);

    /// <summary>Whether <paramref name="key"/> is a LoCon conv mid matrix (<c>.lora_mid.weight</c>). The mid sits between the
    /// down and up convolutions of a LoCon conv layer, so applying the pair without it computes a different operator.</summary>
    public static bool IsLoconMid(string key) =>
        key.EndsWith(".lora_mid.weight", StringComparison.Ordinal);

    /// <summary>Why <paramref name="key"/> names a LoRA form this engine cannot apply, or null when it names none. A file carrying
    /// one is refused whole: its other tensors would apply without it, and the result would be wrong with no signal.</summary>
    public static string? UnsupportedFormReason(string key)
    {
        if (IsPeftDoraMagnitude(key))
        {
            return "a PEFT DoRA magnitude vector, whose rescaling this engine does not apply";
        }
        if (IsLoconMid(key))
        {
            return "a LoCon conv mid matrix (lora_mid), which this engine does not apply";
        }
        return null;
    }

    /// <summary>Whether <paramref name="key"/> looks like an adapter tensor under a suffix <see cref="TryStrip"/> does not recognize.
    /// A consumer reports these once per file rather than refusing, because a file may carry non-adapter tensors beside its adapter.</summary>
    public static bool IsUnrecognizedAdapterKey(string key) =>
        !TryStrip(key, out _, out _)
        && (key.Contains(".lora_", StringComparison.Ordinal)
            || key.Contains(".hada_", StringComparison.Ordinal)
            || key.Contains(".lokr_", StringComparison.Ordinal));

    /// <summary>Whether <paramref name="role"/> is a decomposition matrix rather than a scalar companion (<see cref="LoraRole.Alpha"/>, <see cref="LoraRole.DoraScale"/>) or a full-weight delta (<see cref="LoraRole.Diff"/>, <see cref="LoraRole.BiasDiff"/>). Format detection keys on this: a stray alpha or magnitude vector is not by itself evidence that a file is a LoRA for the root it sits on.</summary>
    public static bool IsDecompositionMatrix(LoraRole role) =>
        role is not (LoraRole.Alpha or LoraRole.DoraScale or LoraRole.Diff or LoraRole.BiasDiff);
}
