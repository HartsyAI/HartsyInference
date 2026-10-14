using System.Collections.Concurrent;

namespace HartsyInference.Gpu;

/// <summary>The reentrant lock that makes one residency cache single-threaded, with deadlock detection across caches.
///
/// <para>An op on one backend can need another backend's cache: a host read of a tensor resident there syncs it
/// under that cache's lock. Two threads doing that in opposite directions at once each hold one lock and wait on the
/// other, and no global order can prevent it, because each already holds its own lock when the need arises. So a
/// blocked waiter records what it waits for, follows the chain of owners, and throws
/// <see cref="ResidencyLockCycleException"/> when the chain leads back to itself. The request on that thread fails;
/// the other one proceeds.</para></summary>
public sealed class ResidencyLock
{
    /// <summary>How long a blocked waiter sleeps between cycle checks.</summary>
    private static readonly TimeSpan CheckInterval = TimeSpan.FromMilliseconds(20);

    /// <summary>Owner chain hops followed before giving up; far beyond the backends one process drives.</summary>
    private const int MaxHops = 64;

    /// <summary>The lock each blocked thread is waiting for, by managed thread id.</summary>
    private static readonly ConcurrentDictionary<int, ResidencyLock> _waiting = new();

    private readonly object _sync = new();
    private int _owner;
    private int _depth;

    /// <summary>Whether the calling thread holds this lock.</summary>
    public bool IsHeldByCurrentThread => Monitor.IsEntered(_sync);

    /// <summary>Takes the lock, waiting for its holder. Throws <see cref="ResidencyLockCycleException"/> instead of
    /// waiting forever when the holder is itself waiting, directly or through others, on a lock this thread holds.</summary>
    public void Enter()
    {
        if (TryEnter())
        {
            return;
        }
        int self = Environment.CurrentManagedThreadId;
        _waiting[self] = this;
        try
        {
            bool cycleSeen = false;
            while (!Monitor.TryEnter(_sync, CheckInterval))
            {
                // Twice in a row, so a chain read while another thread was between acquiring and unregistering
                // cannot fail a request that was about to proceed.
                bool cycle = LeadsBackTo(self);
                if (cycle && cycleSeen)
                {
                    throw new ResidencyLockCycleException();
                }
                cycleSeen = cycle;
            }
            MarkEntered(self);
        }
        finally
        {
            _waiting.TryRemove(self, out _);
        }
    }

    /// <summary>Takes the lock only if it is free or already held by this thread.</summary>
    public bool TryEnter()
    {
        if (!Monitor.TryEnter(_sync))
        {
            return false;
        }
        MarkEntered(Environment.CurrentManagedThreadId);
        return true;
    }

    /// <summary>Releases one hold. Returns true when this was the last one, so the lock is now free.</summary>
    public bool Exit()
    {
        bool last = --_depth == 0;
        if (last)
        {
            Volatile.Write(ref _owner, 0);
        }
        Monitor.Exit(_sync);
        return last;
    }

    private void MarkEntered(int thread)
    {
        if (_depth++ == 0)
        {
            Volatile.Write(ref _owner, thread);
        }
    }

    /// <summary>Whether this lock's owner waits, directly or through other owners, on a lock <paramref name="self"/>
    /// holds.</summary>
    private bool LeadsBackTo(int self)
    {
        ResidencyLock current = this;
        for (int hop = 0; hop < MaxHops; hop++)
        {
            int owner = Volatile.Read(ref current._owner);
            if (owner == 0)
            {
                return false;
            }
            if (owner == self)
            {
                return true;
            }
            if (!_waiting.TryGetValue(owner, out ResidencyLock? next))
            {
                return false;
            }
            current = next;
        }
        return false;
    }
}
