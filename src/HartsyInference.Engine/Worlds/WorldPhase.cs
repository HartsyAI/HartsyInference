using HartsyInference.Core.Logging;

namespace HartsyInference.Engine.Worlds;

/// <summary>Owns every resource acquired during one phase of world work: a model load, or one batch run. Resources are
/// released newest first, each exactly once, by <see cref="Release"/>; callers run it from a finally block so success,
/// cancellation and failure all release the same set. A release that throws is logged and the rest still run.</summary>
public sealed class WorldPhase : IDisposable
{
    private readonly string _name;
    private readonly WorldPhaseLedger? _ledger;
    private readonly object _gate = new();
    private readonly List<Owned> _owned = [];
    private int _acquired;
    private int _released;
    private bool _releasing;

    /// <summary>Creates an empty phase. <paramref name="ledger"/> receives its acquisitions and releases when given.</summary>
    public WorldPhase(string name, WorldPhaseLedger? ledger = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        _name = name;
        _ledger = ledger;
    }

    /// <summary>The phase's label, used in log lines.</summary>
    public string Name => _name;

    /// <summary>Resources the phase has taken ownership of.</summary>
    public int AcquiredCount
    {
        get
        {
            lock (_gate)
            {
                return _acquired;
            }
        }
    }

    /// <summary>Resources the phase has released, including any whose release threw.</summary>
    public int ReleasedCount
    {
        get
        {
            lock (_gate)
            {
                return _released;
            }
        }
    }

    /// <summary>Takes ownership of <paramref name="resource"/>; it is disposed when the phase releases.</summary>
    /// <exception cref="ObjectDisposedException">The phase was already released. The resource is disposed before this is thrown.</exception>
    public T Own<T>(T resource) where T : IDisposable
    {
        ArgumentNullException.ThrowIfNull(resource);
        Register(resource.GetType().Name, resource.Dispose);
        return resource;
    }

    /// <summary>Takes ownership of a teardown that is not a dispose: a converted mapping, or a resource with no IDisposable.</summary>
    public void OwnRelease(string what, Action release)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(what);
        ArgumentNullException.ThrowIfNull(release);
        Register(what, release);
    }

    /// <summary>Releases every owned resource, newest first. Idempotent: a second call does nothing.</summary>
    public void Release()
    {
        Owned[] toRelease;
        lock (_gate)
        {
            if (_releasing)
            {
                return;
            }
            _releasing = true;
            toRelease = _owned.ToArray();
            _owned.Clear();
        }
        for (int i = toRelease.Length - 1; i >= 0; i--)
        {
            Owned owned = toRelease[i];
            try
            {
                owned.Release();
            }
            catch (Exception ex)
            {
                Logs.Error($"[World] Releasing {owned.What} in phase '{_name}' failed; continuing with the rest.", ex);
            }
            finally
            {
                lock (_gate)
                {
                    _released++;
                }
                _ledger?.RecordRelease();
            }
        }
    }

    private void Register(string what, Action release)
    {
        bool accepted;
        lock (_gate)
        {
            accepted = !_releasing;
            if (accepted)
            {
                _owned.Add(new Owned(what, release));
                _acquired++;
                _ledger?.RecordAcquire();
            }
        }
        if (accepted)
        {
            return;
        }
        // Nothing will release a resource that arrives after the phase ended, so release it here.
        release();
        throw new ObjectDisposedException(_name, $"World phase '{_name}' is already released; {what} was released on arrival.");
    }

    /// <inheritdoc/>
    public void Dispose() => Release();

    private readonly record struct Owned(string What, Action Release);
}
