using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Worlds;

namespace HartsyInference.WorldViewer;

/// <summary>Turns the keys a person holds into the session's actions. Only keys the scheme declares are tracked, so an unbound
/// key changes nothing, and a key repeated while held changes nothing. An OnChange scheme sends its action as the held set
/// changes; a PerStep scheme sends nothing here, and <see cref="WorldFrameStream"/> reads <see cref="CurrentAction"/> before each
/// step.</summary>
public sealed class WorldSteering
{
    private readonly object _gate = new();
    private readonly IWorldSession _session;
    private readonly List<string> _held = [];
    private bool _closed;

    /// <summary>Steers <paramref name="session"/> through <paramref name="scheme"/>; a null scheme tracks nothing and sends nothing.</summary>
    public WorldSteering(IWorldSession session, IWorldControlScheme? scheme)
    {
        ArgumentNullException.ThrowIfNull(session);
        _session = session;
        Scheme = scheme;
    }

    /// <summary>The scheme that reads the keys, or null for a model with no live control.</summary>
    public IWorldControlScheme? Scheme { get; }

    /// <summary>Records a press and returns true when the held set changed.</summary>
    public bool Press(string key) => Change(key, press: true);

    /// <summary>Records a release and returns true when the held set changed.</summary>
    public bool Release(string key) => Change(key, press: false);

    /// <summary>Releases every held key, for example when the window loses focus; an OnChange scheme sends the idle action.</summary>
    public void ReleaseAll()
    {
        lock (_gate)
        {
            if (_closed || _held.Count == 0)
            {
                return;
            }
            _held.Clear();
            SendOnChangeLocked();
        }
    }

    /// <summary>The action for the keys held now.</summary>
    public string CurrentAction()
    {
        lock (_gate)
        {
            return Scheme is null ? "none" : Scheme.ActionFor(_held);
        }
    }

    /// <summary>The keys held now, in the order they were pressed.</summary>
    public IReadOnlyList<string> HeldKeys()
    {
        lock (_gate)
        {
            return _held.ToArray();
        }
    }

    /// <summary>Stops sending: keys pressed after this change nothing. Call it before the session is disposed.</summary>
    public void Close()
    {
        lock (_gate)
        {
            _closed = true;
        }
    }

    private bool Change(string key, bool press)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(key);
        lock (_gate)
        {
            if (_closed || Scheme is null || !IsDeclared(Scheme, key))
            {
                return false;
            }
            int index = IndexOfHeld(key);
            if (press)
            {
                if (index >= 0)
                {
                    return false;
                }
                _held.Add(key);
            }
            else
            {
                if (index < 0)
                {
                    return false;
                }
                _held.RemoveAt(index);
            }
            SendOnChangeLocked();
            return true;
        }
    }

    private void SendOnChangeLocked()
    {
        IWorldControlScheme? scheme = Scheme;
        if (scheme is { Cadence: WorldControlCadence.OnChange })
        {
            _session.SendAction(scheme.ActionFor(_held));
        }
    }

    private static bool IsDeclared(IWorldControlScheme scheme, string key)
    {
        foreach (WorldControl control in scheme.Controls)
        {
            if (string.Equals(control.Key, key, StringComparison.OrdinalIgnoreCase))
            {
                return true;
            }
        }
        return false;
    }

    private int IndexOfHeld(string key)
    {
        for (int i = 0; i < _held.Count; i++)
        {
            if (string.Equals(_held[i], key, StringComparison.OrdinalIgnoreCase))
            {
                return i;
            }
        }
        return -1;
    }
}
