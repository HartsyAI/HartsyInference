namespace HartsyInference.Core.Backends.Attention;

/// <summary>A row predicate or launch over one call.</summary>
public delegate bool AttentionRowFunc(in AttentionCall call);

/// <summary>States the preparation a row expects for a call.</summary>
public delegate AttentionPreparationCost AttentionCostFunc(in AttentionCall call);

/// <summary>A kernel row built from delegates: runtime admission, the launch, and optionally the preparation it expects.</summary>
/// <param name="branch">The branch id this row reports (docs/ATTENTION_CONTRACT.md routes); null when the launch records
/// its own variant (a row covering several dtype paths, or one that re-dispatches).</param>
/// <param name="preparation">The row's statement of the preparation it runs for a call; null leaves the row unstated.</param>
public sealed class DelegateAttentionRow(AttentionKernelCapability capability, string? branch, AttentionRowFunc admits,
    AttentionRowFunc run, AttentionCostFunc? preparation = null) : AttentionKernelRow
{
    /// <inheritdoc/>
    public override AttentionPreparationCost? Preparation(in AttentionCall call) => preparation?.Invoke(in call);

    /// <summary>The branch id, or null when the launch records it.</summary>
    public string? Branch { get; } = branch;

    /// <inheritdoc/>
    public override AttentionKernelCapability Capability { get; } = capability;

    /// <inheritdoc/>
    public override bool Admits(in AttentionCall call) => admits(in call);

    /// <inheritdoc/>
    public override bool Run(in AttentionCall call) => run(in call);
}

/// <summary>Route names for a launch that re-enters another entry point ("M.3/S.15"), interned so the per-call record
/// allocates once per distinct pair.</summary>
public static class AttentionRouteText
{
    private static readonly System.Collections.Concurrent.ConcurrentDictionary<(string, string?), string> Cache = new();

    /// <summary><paramref name="outer"/> followed by the route the nested call recorded.</summary>
    public static string Nested(string outer, string? inner) => Cache.GetOrAdd((outer, inner), static k => k.Item1 + k.Item2);
}
