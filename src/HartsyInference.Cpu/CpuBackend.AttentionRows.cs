using HartsyInference.Core.Backends;

namespace HartsyInference.Cpu;

// The CPU's attention: the SIMD kernel for plain head-major calls, the FlashAttention reference for the LLM entry, and
// the host contract reference (strided operands, every mask component) for anything else.
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
            case AttentionRoute.Flash:
                ((IBackend)this).FlashAttention(call.Output.Storage, call.Query.Storage, call.Key.Storage, call.Value.Storage,
                    call.ValidKvLength, call.KvGroup, call.Mask.Causal, call.Positions.QueryOffset, call.Scale, call.Logits.Softcap,
                    call.Logits.Sinks, call.Mask.LegacySlidingWindow, call.Logits.AlibiSlopes);
                break;
            case AttentionRoute.FlashDevice:
                ((IBackend)this).FlashAttentionDev(call.Output.Storage, call.Query.Storage, call.Key.Storage, call.Value.Storage,
                    call.ValidKvLength, call.KvGroup, call.Mask.Causal, call.Positions.QueryOffset, call.Scale, call.Positions.DeviceHandle,
                    call.Logits.Softcap, call.Mask.LegacySlidingWindow);
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
            AttentionRoute.Flash or AttentionRoute.FlashDevice => "cpu.flash.reference: the FlashAttention reference",
            _ => "cpu.host.reference: the contract reference",
        };
        return new(null, null, reason, null);
    }

    /// <summary>The legacy entry that carries the whole call, or <see cref="AttentionRoute.Auto"/> for the host reference. The
    /// SIMD kernel reads K/V with the query's head count, so grouped calls take the reference.</summary>
    private static AttentionRoute LegacyRoute(in AttentionCall call) => call.LegacyGap() is not null ? AttentionRoute.Auto
        : call.Route switch
        {
            AttentionRoute.Dense when call.KvHeads == call.Heads => AttentionRoute.Dense,
            AttentionRoute.Flash => AttentionRoute.Flash,
            AttentionRoute.FlashDevice => AttentionRoute.FlashDevice,
            _ => AttentionRoute.Auto,
        };
}
