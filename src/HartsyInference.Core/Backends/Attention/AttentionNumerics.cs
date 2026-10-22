namespace HartsyInference.Core.Backends;

/// <summary>Formats narrower than F32 an attention kernel may store operands in and multiply with.</summary>
[Flags]
public enum AttentionFormats
{
    /// <summary>F32 only.</summary>
    None = 0,

    /// <summary>TF32 tensor-core multiply of F32 operands.</summary>
    Tf32 = 1,

    /// <summary>BF16 storage and multiply.</summary>
    Bf16 = 2,

    /// <summary>F16 storage and multiply.</summary>
    F16 = 4,

    /// <summary>INT8 quantization with per-row or finer scales (and, for K, smoothing).</summary>
    Int8 = 8,

    /// <summary>FP8 E4M3 storage and multiply.</summary>
    Fp8 = 16,
}

/// <summary>The approximations a call permits and the bounds that make them safe (docs/ATTENTION_CONTRACT.md §5).</summary>
/// <remarks>Independent permissions, not a ladder: F32 storage and arithmetic are always allowed and everything else
/// must be granted. A bound of 0 means unknown, and an unknown bound grants nothing.</remarks>
public readonly record struct AttentionNumerics
{
    /// <summary>Formats Q and K may be stored in for the QKᵀ multiply.</summary>
    public AttentionFormats QueryKey { get; init; }

    /// <summary>Formats P and V may be stored in for the PV multiply.</summary>
    public AttentionFormats ProbValue { get; init; }

    /// <summary>Accumulation or softmax below F32 is permitted.</summary>
    public bool LowPrecisionAccumulate { get; init; }

    /// <summary>Clipping an out-of-range value to a narrower format's maximum is permitted (an approximation, not a fix).</summary>
    public bool Saturation { get; init; }

    /// <summary>Bitwise run-to-run reproducibility on one device and build is required.</summary>
    public bool Deterministic { get; init; }

    /// <summary>Bound on <c>|α·q·k|</c> derived from the model's geometry; 0 = unknown.</summary>
    public float ScoreBound { get; init; }

    /// <summary>Bound on any Q or K element; 0 = unknown.</summary>
    public float QueryKeyBound { get; init; }

    /// <summary>Bound on any V element; 0 = unknown.</summary>
    public float ValueBound { get; init; }

    /// <summary>A legacy entry point's <c>allowF16</c>: an assertion, not a derived bound. Only route-pinned calls honour it.</summary>
    public bool LegacyAssertedBounded { get; init; }

    /// <summary>Multiply formats the call refuses outright: a row that would use one is excluded, whatever the call grants or
    /// asserts.</summary>
    public AttentionFormats Refused { get; init; }

    /// <summary>F32 everywhere: no TF32, no narrowing, no saturation, no quantization.</summary>
    public static AttentionNumerics Exact => default;

    /// <summary>Q, K, P and V may be stored and multiplied in BF16; scores, softmax and accumulation stay F32. BF16 keeps F32's
    /// range, so this is a precision permission, not a range one.</summary>
    public static AttentionNumerics Bf16Storage => new() { QueryKey = AttentionFormats.Bf16, ProbValue = AttentionFormats.Bf16 };

    /// <summary>The permissions a legacy entry point's <c>allowF16</c> carried.</summary>
    public static AttentionNumerics FromLegacy(bool allowF16) => new() { LegacyAssertedBounded = allowF16 };

    /// <summary>True when both multiplies may use the given formats.</summary>
    public bool Permits(AttentionFormats queryKey, AttentionFormats probValue) =>
        (QueryKey & queryKey) == queryKey && (ProbValue & probValue) == probValue;
}
