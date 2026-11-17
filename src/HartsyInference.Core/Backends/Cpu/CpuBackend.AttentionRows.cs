using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Backends.Cpu.Kernels;

namespace HartsyInference.Core.Backends.Cpu;

// The CPU's attention: the SIMD kernel for plain head-major calls, the flash kernel for the LLM entry, and the host
// contract reference (strided operands, every mask component) for anything else. Legacy entries adapt to Attention.
public sealed partial class CpuBackend
{
    /// <inheritdoc/>
    public void Attention(in AttentionCall call)
    {
        AttentionCall prepared = AttentionExecution.Prepare(in call);
        switch (LegacyRoute(in prepared))
        {
            case AttentionRoute.Dense:
                RunDenseAttention(in prepared);
                break;
            case AttentionRoute.Flash or AttentionRoute.FlashDevice:
                if (prepared.Positions.OnDevice)
                    throw new AttentionUnsupportedException("The CPU backend keeps positions on the host; it hands out no device position buffer.");
                FlashAttentionKernels.Run(prepared.Output.Storage, prepared.Query.Storage, prepared.Key.Storage, prepared.Value.Storage, prepared.ValidKvLength,
                    prepared.Mask.Causal, prepared.Positions.QueryOffset, prepared.Scale, prepared.Logits.Softcap, prepared.Logits.Sinks,
                    prepared.Mask.LegacySlidingWindow, prepared.Logits.AlibiSlopes);
                AttentionEvidence.Executed("cpu", "cpu.flash", null, in prepared);
                break;
            default:
                AttentionHostReference.Run(in prepared);
                AttentionEvidence.Executed("cpu", "cpu.host.reference", null, in prepared);
                break;
        }
    }

    /// <inheritdoc/>
    public AttentionPlan PlanAttention(in AttentionCall call)
    {
        AttentionCall prepared = AttentionExecution.Prepare(in call);
        string reason = LegacyRoute(in prepared) switch
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
