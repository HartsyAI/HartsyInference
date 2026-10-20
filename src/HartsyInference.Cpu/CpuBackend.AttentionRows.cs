using HartsyInference.Core.Backends;

namespace HartsyInference.Cpu;

// The CPU's attention: the SIMD kernel for plain head-major calls, the FlashAttention reference for the LLM entry, and
// the host contract reference (strided operands, every mask component) for anything else.
public sealed partial class CpuBackend
{
    /// <inheritdoc/>
    public void Attention(in AttentionCall call)
    {
        bool plain = call.Query.IsPlain && call.Key.IsPlain && call.Value.IsPlain && call.Output.IsPlain;
        if (plain && call.Route == AttentionRoute.Dense)
            ScaledDotProductAttention(call.Output.Storage, call.Query.Storage, call.Key.Storage, call.Value.Storage, call.Mask.Bias, call.Scale,
                call.Numerics.LegacyAssertedBounded);
        else if (plain && call.Route == AttentionRoute.Flash)
            ((IBackend)this).FlashAttention(call.Output.Storage, call.Query.Storage, call.Key.Storage, call.Value.Storage, call.Positions.KvLength,
                call.KvGroup, call.Mask.Causal, call.Positions.QueryOffset, call.Scale, call.Logits.Softcap, call.Logits.Sinks,
                call.Mask.LegacySlidingWindow, call.Logits.AlibiSlopes);
        else
            AttentionHostReference.Run(in call);
    }

    /// <inheritdoc/>
    public AttentionPlan PlanAttention(in AttentionCall call)
    {
        bool plain = call.Query.IsPlain && call.Key.IsPlain && call.Value.IsPlain && call.Output.IsPlain;
        string reason = plain && call.Route == AttentionRoute.Dense ? "cpu.simd: the vectorized head-major kernel"
            : plain && call.Route == AttentionRoute.Flash ? "cpu.flash.reference: the FlashAttention reference"
            : "cpu.host.reference: the contract reference over strided operands";
        return new(null, null, reason, null);
    }
}
