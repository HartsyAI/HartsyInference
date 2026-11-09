namespace HartsyInference.Core.Backends.Gating;

/// <summary>The SiTU-GLU constants that device kernels compile in and that configs default to.</summary>
public static class SituDefaults
{
    /// <summary>The gate soft-cap β. MOSS-Image hard-codes it; ERNIE-5.5 ships it, so the default keeps existing SiTU callers unchanged.</summary>
    public const float Beta = 4f;

    /// <summary>The value soft-cap β<sub>lin</sub>. MOSS-Image hard-codes it; ERNIE-5.5 ships it.</summary>
    public const float LinearBeta = 25f;
}
