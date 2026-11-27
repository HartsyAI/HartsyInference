namespace HartsyInference.Core.Backends.Gating;

/// <summary>The SiTU-GLU betas configs default to (MOSS-Image's fixed values and ERNIE-5.5's shipped ones).</summary>
public static class SituDefaults
{
    /// <summary>The gate soft-cap β. MOSS-Image hard-codes it; ERNIE-5.5 ships it, so the default keeps existing SiTU callers unchanged.</summary>
    public const float Beta = 4f;

    /// <summary>The value soft-cap β<sub>lin</sub>. MOSS-Image hard-codes it; ERNIE-5.5 ships it.</summary>
    public const float LinearBeta = 25f;
}
