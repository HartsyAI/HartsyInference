using HartsyInference.Core.MemoryManagement;

namespace HartsyInference.Engine.Planning.Memory;

/// <summary>The device-independent half of a fit decision: given an estimate and what one device offers, the verdict.</summary>
/// <remarks>Pure so it can be tested at every tier and capacity boundary without a GPU, and so any host with its own
/// notion of a device can reuse the rule rather than restating it.</remarks>
internal static class MemoryFitJudge
{
    /// <summary>Judges <paramref name="estimate"/> against one device.</summary>
    /// <param name="policy">The effective policy: the backend's with the request's overrides applied.</param>
    /// <param name="canStream">Whether the model wires block streaming AND the backend has a streaming cache; the
    /// policy's own streaming lever is applied here.</param>
    /// <param name="primaryBytes">Usable bytes on the engine's own device.</param>
    /// <param name="denoiserBytes">Usable bytes for the denoiser: the primary plus any pooled shard devices.</param>
    /// <param name="onPrimary">Which components run on the engine's own device.</param>
    public static MemoryFit Judge(MemoryEstimate estimate, VramPolicy policy, bool canStream, long primaryBytes,
        long denoiserBytes, Func<MemoryComponent, bool> onPrimary)
    {
        ArgumentNullException.ThrowIfNull(estimate);
        ArgumentNullException.ThrowIfNull(policy);
        ArgumentNullException.ThrowIfNull(onPrimary);
        bool unload = policy.PhaseUnload != LeverState.Off;
        bool streamingAllowed = canStream && policy.WeightStreaming != LeverState.Off;

        (bool resident, long residentNeed) = Fits(estimate, onPrimary, unload, primaryBytes, denoiserBytes, streamed: false);
        (bool streamed, long streamNeed) = streamingAllowed
            ? Fits(estimate, onPrimary, unload, primaryBytes, denoiserBytes, streamed: true)
            : (false, residentNeed);

        MemoryFitVerdict verdict = resident ? MemoryFitVerdict.Resident
            : streamed ? MemoryFitVerdict.Streamed
            : MemoryFitVerdict.Infeasible;
        string usable = ByteFormat.GbF1(denoiserBytes);
        string reason = verdict switch
        {
            MemoryFitVerdict.Resident => $"Fits resident (~{ByteFormat.GbF1(residentNeed)} of {usable} usable).",
            MemoryFitVerdict.Streamed => $"Fits only by streaming weights (~{ByteFormat.GbF1(residentNeed)} resident, "
                + $"~{ByteFormat.GbF1(streamNeed)} streamed, {usable} usable).",
            _ => $"Needs ~{ByteFormat.GbF1(streamingAllowed ? streamNeed : residentNeed)}"
                + (streamingAllowed ? " even with weight streaming" : "") + $"; {usable} usable on this device.",
        };
        return new MemoryFit
        {
            Verdict = verdict,
            Estimate = estimate,
            CapacityBytes = denoiserBytes,
            EffectiveTier = policy.Tier,
            Reason = reason,
        };
    }

    /// <summary>Whether every phase fits the device it runs on, and the largest single requirement for the reason text.</summary>
    /// <remarks>Phase by phase when phases unload (each must fit its own device on its own); the whole resident set on
    /// the pooled capacity when they do not. Only the denoiser is pooled across shard devices.</remarks>
    private static (bool Fits, long Need) Fits(MemoryEstimate estimate, Func<MemoryComponent, bool> onPrimary,
        bool unload, long primaryBytes, long denoiserBytes, bool streamed)
    {
        if (!unload)
        {
            long total = streamed ? estimate.FloorBytes(false, onPrimary) : estimate.PeakBytes(false, onPrimary);
            return (total <= denoiserBytes, total);
        }
        bool fits = true;
        long need = 0;
        foreach (MemoryPhase phase in estimate.Phases)
        {
            if (!onPrimary(phase.Component))
            {
                continue;
            }
            long bytes = streamed ? phase.FloorBytes : phase.ResidentBytes;
            long capacity = phase.Component == MemoryComponent.Denoiser ? denoiserBytes : primaryBytes;
            fits &= bytes <= capacity;
            need = Math.Max(need, bytes);
        }
        return (fits, need);
    }
}
