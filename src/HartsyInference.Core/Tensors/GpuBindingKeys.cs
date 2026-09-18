namespace HartsyInference.Core.Tensors;

/// <summary>Hands out the process-unique identities that mark which cache owns a tensor's GPU binding.
///
/// <para>Lives in Core, beside the bindings it names, because the key identifies a bucket in
/// <see cref="Tensor"/>'s binding table and every backend needs one whether or not it uses the shared residency
/// cache. Keeping the sequence here is what makes uniqueness structural: a backend cannot grow a private counter
/// without also growing a reason not to call this, and the last two times this went wrong it was exactly that —
/// two counters, each correct alone, both starting at 1.</para>
///
/// <para>Deliberately not a static member of a generic cache. A static field inside a generic class exists once per
/// CLOSED type, so a counter declared there gives every buffer type its own sequence and every backend's first cache
/// the same key. The identities must be unique across all of them, because the thing they distinguish — which cache
/// owns a binding on a tensor resident on several devices — spans backends by definition.</para></summary>
internal static class GpuBindingKeys
{
    private static long _next;

    /// <summary>The next identity. Starts at 1; zero stays reserved for callers that predate keyed bindings.</summary>
    internal static nint Next() => (nint)Interlocked.Increment(ref _next);
}
