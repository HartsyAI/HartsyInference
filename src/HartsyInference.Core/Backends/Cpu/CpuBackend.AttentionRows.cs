using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Backends.Cpu.Kernels;

namespace HartsyInference.Core.Backends.Cpu;

// The CPU's attention: the SIMD kernel for plain head-major calls, the flash kernel for the LLM entry, and the host
// contract reference (strided operands, every mask component) for anything else. Neither kernel is reached through a
// legacy IBackend entry, so the interface defaults that route those entries here cannot recurse.
public sealed partial class CpuBackend
{
    /// <inheritdoc/>
    public void Attention(in AttentionCall call)
    {
        switch (LegacyRoute(in call))
        {
            case AttentionRoute.Dense:
                ScaledDotProductAttention(call.Output.Storage, call.Query.Storage, call.Key.Storage, call.Value.Storage, call.Mask.Bias,
                    call.Scale, call.Numerics.LegacyAssertedBounded);
                break;
            case AttentionRoute.Flash or AttentionRoute.FlashDevice:
                if (call.Positions.OnDevice)
                    throw new AttentionUnsupportedException("The CPU backend keeps positions on the host; it hands out no device position buffer.");
                FlashAttentionKernels.Run(call.Output.Storage, call.Query.Storage, call.Key.Storage, call.Value.Storage, call.ValidKvLength,
                    call.Mask.Causal, call.Positions.QueryOffset, call.Scale, call.Logits.Softcap, call.Logits.Sinks, call.Mask.LegacySlidingWindow,
                    call.Logits.AlibiSlopes);
                break;
            default:
                AttentionHostReference.Run(in call);
                break;
        }
    }

    /// <inheritdoc/>
    public AttentionPlan PlanAttention(in AttentionCall call)
    {
        string reason = LegacyRoute(in call) switch
        {
            AttentionRoute.Dense => "cpu.simd: the vectorized head-major kernel",
            AttentionRoute.Flash or AttentionRoute.FlashDevice => "cpu.flash: the online-softmax flash kernel",
            _ => "cpu.host.reference: the contract reference",
        };
        return new(null, null, reason, null);
    }

    /// <summary>The legacy entry that carries the whole call, or <see cref="AttentionRoute.Auto"/> for the host reference. The
    /// SIMD kernel reads K/V with the query's head count, so grouped dense calls take the reference.</summary>
    private static AttentionRoute LegacyRoute(in AttentionCall call) => call.LegacyGap() is not null ? AttentionRoute.Auto
        : call.Route switch
        {
            AttentionRoute.Dense when call.KvHeads == call.Heads => AttentionRoute.Dense,
            AttentionRoute.Flash => AttentionRoute.Flash,
            AttentionRoute.FlashDevice => AttentionRoute.FlashDevice,
            _ => AttentionRoute.Auto,
        };
}
