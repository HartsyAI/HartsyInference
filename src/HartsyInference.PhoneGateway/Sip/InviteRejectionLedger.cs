using SIPSorcery.SIP;

namespace HartsyInference.PhoneGateway.Sip;

/// <summary>The INVITEs the gateway has refused, keyed by Call-ID and top Via branch and kept for the lifetime of an
/// INVITE transaction, so one refusal is counted once however often the caller retransmits, and a copy that reaches
/// the gateway again is answered the same way instead of being offered as a new call.</summary>
/// <remarks>Thread-safe. Entries expire in the order they were recorded, and beyond <see cref="Capacity"/> the oldest
/// is dropped, so a flood of distinct INVITEs costs bounded memory.</remarks>
internal sealed class InviteRejectionLedger
{
    /// <summary>64·T1 with T1 = 500 ms: how long a caller keeps retransmitting one INVITE (RFC 3261 Timer B).</summary>
    public const long DefaultLifetimeMs = 32_000;

    public const int DefaultCapacity = 1024;

    private readonly object _lock = new();
    private readonly Dictionary<(string CallId, string Branch), InviteRejection> _entries = new();
    private readonly Queue<((string CallId, string Branch) Key, long RecordedAtMs)> _order = new();
    private readonly long _lifetimeMs;
    private readonly int _capacity;
    private readonly Func<long> _clockMs;

    /// <param name="clockMs">Millisecond clock; <see cref="Environment.TickCount64"/> when null.</param>
    public InviteRejectionLedger(long lifetimeMs = DefaultLifetimeMs, int capacity = DefaultCapacity, Func<long>? clockMs = null)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(lifetimeMs, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(capacity, 1);
        _lifetimeMs = lifetimeMs;
        _capacity = capacity;
        _clockMs = clockMs ?? (static () => Environment.TickCount64);
    }

    public int Capacity => _capacity;

    /// <summary>Refusals still remembered.</summary>
    public int Count
    {
        get
        {
            lock (_lock)
            {
                Prune(_clockMs());
                return _entries.Count;
            }
        }
    }

    /// <summary>Records the refusal of <paramref name="invite"/>.</summary>
    /// <returns>True for the first sighting, the one to count; false for a copy, with <paramref name="first"/> set to
    /// the refusal recorded for it.</returns>
    public bool TryRecord(SIPRequest invite, SIPResponseStatusCodesEnum status, string reason, out InviteRejection first)
    {
        ArgumentNullException.ThrowIfNull(invite);
        (string CallId, string Branch) key = KeyOf(invite);
        long now = _clockMs();
        lock (_lock)
        {
            Prune(now);
            if (_entries.TryGetValue(key, out first))
            {
                return false;
            }
            while (_entries.Count >= _capacity && _order.Count > 0)
            {
                Forget(_order.Dequeue());
            }
            first = new InviteRejection(status, reason, now);
            _entries[key] = first;
            _order.Enqueue((key, now));
            return true;
        }
    }

    /// <summary>True when <paramref name="invite"/>, or a copy of it, was refused within the lifetime.</summary>
    public bool TryGet(SIPRequest invite, out InviteRejection rejection)
    {
        ArgumentNullException.ThrowIfNull(invite);
        (string CallId, string Branch) key = KeyOf(invite);
        lock (_lock)
        {
            Prune(_clockMs());
            return _entries.TryGetValue(key, out rejection);
        }
    }

    private static (string CallId, string Branch) KeyOf(SIPRequest invite) =>
        (invite.Header.CallId ?? "", invite.Header.Vias?.TopViaHeader?.Branch ?? "");

    private void Prune(long now)
    {
        while (_order.Count > 0 && now - _order.Peek().RecordedAtMs >= _lifetimeMs)
        {
            Forget(_order.Dequeue());
        }
    }

    /// <summary>Drops the entry a queue slot refers to, unless the key has since been recorded again.</summary>
    private void Forget(((string CallId, string Branch) Key, long RecordedAtMs) slot)
    {
        if (_entries.TryGetValue(slot.Key, out InviteRejection entry) && entry.RecordedAtMs == slot.RecordedAtMs)
        {
            _entries.Remove(slot.Key);
        }
    }
}
