namespace HartsyInference.Core.MemoryManagement;

/// <summary>The policy in force for the generation currently running on this async flow, which is what makes a PER-REQUEST override reach the levers that are decided mid-generation.</summary>
/// <remarks>Without this, a request-level override only reaches decisions made while the pipeline is being
/// CONSTRUCTED (it rides <c>RecipeContext</c>), and pipelines are cached — so the levers read during the
/// generation itself (<see cref="VramPlanner"/>'s streaming decision, <see cref="VramLevers.KeepResident(Backends.IBackend)"/>)
/// would keep answering from the per-backend registry and the override would silently do nothing.
/// <para>Deliberately <see cref="AsyncLocal{T}"/> rather than a static: two engines on two devices generate
/// concurrently, and a process-wide value would let one generation's override decide the other's placement — the
/// same last-writer-wins defect the per-backend registry exists to avoid.</para>
/// <para>Scoped rather than assigned: an override belongs to one request, and a leaked value would quietly govern
/// every later generation on that flow with no way for the operator to see why.</para></remarks>
public static class VramPolicyScope
{
    private static readonly AsyncLocal<VramPolicy?> _current = new();

    /// <summary>The current generation's policy, or null outside one.</summary>
    public static VramPolicy? Current => _current.Value;

    /// <summary>Makes <paramref name="policy"/> current until the returned scope is disposed. A null policy pushes nothing, so a request without overrides costs no behavior change.</summary>
    public static IDisposable Push(VramPolicy? policy) => new Scope(policy);

    private sealed class Scope : IDisposable
    {
        private readonly VramPolicy? _previous;
        private bool _disposed;

        internal Scope(VramPolicy? policy)
        {
            _previous = _current.Value;
            _current.Value = policy ?? _previous;
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            _current.Value = _previous;
        }
    }
}
