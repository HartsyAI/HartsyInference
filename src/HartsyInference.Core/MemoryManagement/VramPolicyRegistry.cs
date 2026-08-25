using System.Runtime.CompilerServices;
using HartsyInference.Core.Backends;

namespace HartsyInference.Core.MemoryManagement;

/// <summary>Per-backend VRAM policy. Two engines in one process — one per GPU, on differently-sized cards — each keep their own.</summary>
/// <remarks>Weak keys: a disposed backend's entry vanishes with it, so the failure paths need no unregistration.
/// This replaces the process-wide environment variable as the authority, whose last-writer-wins semantics broke
/// multi-backend setups.</remarks>
public static class VramPolicyRegistry
{
    private static readonly ConditionalWeakTable<IBackend, VramPolicy> _policies = new();

    /// <summary>Pins <paramref name="policy"/> for <paramref name="backend"/>, replacing any previous entry.</summary>
    public static void Set(IBackend backend, VramPolicy policy)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(policy);
        _policies.AddOrUpdate(backend, policy);
    }

    /// <summary>Drops <paramref name="backend"/>'s policy so it falls back to the environment resolution.</summary>
    public static void Clear(IBackend backend)
    {
        ArgumentNullException.ThrowIfNull(backend);
        _policies.Remove(backend);
    }

    /// <summary>The policy governing <paramref name="backend"/>: the running generation's scope first, then its pinned entry, then the environment-derived default. A null backend resolves the scope and environment only.</summary>
    /// <remarks>The scope wins because it is strictly more specific: it exists only while a request that overrode
    /// something is actually generating, and that request asked for this behavior for itself. The registry stays
    /// the answer for everything outside a generation, and for every generation that overrode nothing.</remarks>
    public static VramPolicy Resolve(IBackend? backend)
    {
        if (VramPolicyScope.Current is VramPolicy scoped)
        {
            return scoped;
        }
        if (backend is not null && _policies.TryGetValue(backend, out VramPolicy? pinned))
        {
            return pinned;
        }
        return VramPolicyResolver.FromLegacyMode(LowVramPolicy.ResolveEnvironment());
    }

    /// <summary>The policy for one generation: <paramref name="backend"/>'s, refined by any per-request overrides.</summary>
    public static VramPolicy Resolve(IBackend? backend, VramOverrides? overrides)
        => VramPolicyResolver.Apply(Resolve(backend), overrides);

    /// <summary>True when <paramref name="backend"/> has an explicit policy rather than inheriting the environment's.</summary>
    public static bool HasPolicy(IBackend? backend) => backend is not null && _policies.TryGetValue(backend, out _);
}
