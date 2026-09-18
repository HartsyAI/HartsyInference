namespace HartsyInference.Gpu;

/// <summary>Hands out the process-unique identities that mark which cache owns a tensor's GPU binding.
///
/// <para>Deliberately not generic, and deliberately not a static member of <see cref="GpuResidencyCache{TBuffer}"/>.
/// A static field inside a generic class exists once per CLOSED type, so a counter declared there would give every
/// buffer type its own sequence and every backend's first cache the same key. The keys must be unique across all of
/// them, because the thing they distinguish — which cache owns a binding on a tensor resident on several devices —
/// spans backends by definition.</para></summary>
internal static class GpuBindingKeys
{
    private static long _next;

    /// <summary>The next identity. Starts at 1; zero stays reserved for callers that predate keyed bindings.</summary>
    internal static nint Next() => (nint)Interlocked.Increment(ref _next);
}
