using HartsyInference.Core.Backends.Attention;

namespace HartsyInference.GpuBenchmarks.Attention;

/// <summary>Rejects mislabeled fixture operations before backend creation, warmup or timing.</summary>
internal static class AttentionEvidenceContract
{
    /// <summary>Checks the catalog arm against immutable fixture semantics without modifying either.</summary>
    public static void Validate(AttentionArm arm, in AttentionCall call, string boundary, string? expectedBranch)
    {
        AttentionRoute required = arm.Entry switch
        {
            "sdpa" => AttentionRoute.Dense,
            "flash" => AttentionRoute.Flash,
            _ => throw new NotSupportedException("No shared-fixture adapter for this catalog entry."),
        };
        if (call.Route != required)
            throw new InvalidDataException($"Arm {arm.Name} requires route {required}; fixture declares {call.Route}.");
        if (call.Query.DType != arm.Dtype || call.Output.DType != arm.Dtype
            || call.Key.DType != arm.KvDtype || call.Value.DType != arm.KvDtype)
            throw new InvalidDataException($"Arm {arm.Name} requires Q/O={arm.Dtype} and K/V={arm.KvDtype}; fixture dtypes disagree.");
        AttentionNumerics permissions = arm.Grant ?? AttentionNumerics.FromLegacy(arm.AllowF16);
        if (call.Numerics.LegacyAssertedBounded != permissions.LegacyAssertedBounded)
            throw new InvalidDataException($"Arm {arm.Name} and fixture disagree on the legacy numerical assertion.");
        if (arm.Grant is not null && (call.Numerics.QueryKey != permissions.QueryKey
            || call.Numerics.ProbValue != permissions.ProbValue || call.Numerics.Refused != permissions.Refused
            || call.Numerics.Saturation != permissions.Saturation
            || call.Numerics.LowPrecisionAccumulate != permissions.LowPrecisionAccumulate
            || call.Numerics.Deterministic != permissions.Deterministic))
            throw new InvalidDataException($"Arm {arm.Name} and fixture disagree on explicit numerical permissions.");
        if (boundary is not ("public" or "resident" or "observe"))
            throw new ArgumentException("Unknown timing boundary.", nameof(boundary));
        if (boundary != "observe" && string.IsNullOrWhiteSpace(expectedBranch))
            throw new ArgumentException("Timing requires a preregistered --expect-branch; use --boundary observe for route discovery.",
                nameof(expectedBranch));
    }
}
