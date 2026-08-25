using HartsyInference.Core.Backends;

namespace HartsyInference.Core.MemoryManagement;

/// <summary>The rule for touching device memory while a captured step graph exists: a replay re-executes baked pointers, so anything that frees or moves what those pointers name has to settle with the graph first.</summary>
/// <remarks>The two directions here are NOT interchangeable, and picking the wrong one is how this becomes a silent
/// bug rather than a loud one.
/// <list type="bullet">
/// <item><see cref="InvalidateBeforeRelease"/> — for a release the caller has already decided on (evicting a resident
/// model, freeing a phase's weights). The graph must go, never the free: skipping the free would silently ignore the
/// eviction and OOM later, having reported success.</item>
/// <item><see cref="CanMoveMemoryFreely"/> — for an OPTIONAL lever that is only ever an optimisation (paging a
/// step cache to host). Here skipping is correct, because not doing it costs nothing but correctness is at stake if
/// a live graph's pointers move underneath it.</item>
/// </list>
/// <para>A third shape — a streamed forward that frees and re-uploads per block — is neither of these: it suspends
/// capture for the loop rather than invalidating once, and the suspension is per-transformer state
/// (<c>StepGraphSuspended</c>), so it stays with the denoisers that own it.</para></remarks>
public static class VramGraphGuard
{
    /// <summary>Drops any captured step graph on <paramref name="backend"/> so a subsequent free cannot strand a replay against reclaimed memory. Call immediately BEFORE the release, never instead of it.</summary>
    /// <remarks>Mirrors what each denoiser's own <c>InvalidateStepGraph</c> does — reset, then disown — but works
    /// from Core, which cannot see the concrete transformer types. Clearing the owner unconditionally is safe: the
    /// graph is already gone after the reset, so a stale owner reference would only suppress a later re-capture.</remarks>
    public static void InvalidateBeforeRelease(IBackend? backend)
    {
        if (backend is null || !backend.StepGraphSupported)
        {
            return;
        }
        backend.StepGraphReset();
        backend.StepGraphOwner = null;
    }

    /// <summary>True when no captured graph is holding pointers into device memory, so an optional lever may move it. Never gate a required free on this.</summary>
    public static bool CanMoveMemoryFreely(IBackend? backend)
        => backend is null || (!backend.StepGraphReady && backend.StepGraphOwner is null);
}
