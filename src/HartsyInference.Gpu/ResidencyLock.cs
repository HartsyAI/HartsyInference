using System.Collections.Concurrent;

namespace HartsyInference.Gpu;

/// <summary>The reentrant lock that makes one residency cache single-threaded, with deadlock detection across caches.
///
/// <para>An op on one backend can need another backend's cache: a host read of a tensor resident there syncs it
/// under that cache's lock. Two threads doing that in opposite directions at once each hold one lock and wait on the
/// other, and no global order can prevent it, because each already holds its own lock when the need arises. So a
/// blocked waiter records what it waits for and follows the chain of owners. When the chain leads back to itself,
/// exactly one thread in the cycle yields: the one with the largest managed thread id throws
/// <see cref="ResidencyLockCycleException"/>, its request fails and unwinds, and the others keep waiting and then get
/// their locks.</para>
///
/// <para>Only waits made through <see cref="Enter"/> are followed. A cycle that passes through any other lock (the
/// engine's <c>DeviceGate</c>, a service lock, a monitor) is not detected. None exists today: the residency lock is
/// taken only inside a backend op, a tensor callback or a backend dispose, none of which reaches engine code, and the
/// engine disposes a backend while holding the device gate, never the other way round. So the order is always the
/// device gate first, then the residency lock.</para></summary>
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
                // cannot fail a request that was about to proceed. Every thread in the cycle sees the same members,
                // so the largest id yields and the rest wait for it to unwind.
                bool cycle = LeadsBackTo(self, out int largest);
                if (cycle && cycleSeen && largest == self)
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
    /// <exception cref="SynchronizationLockException">The calling thread does not hold the lock; nothing is
    /// changed.</exception>
    public bool Exit()
    {
        if (!Monitor.IsEntered(_sync))
        {
            throw new SynchronizationLockException("The residency lock was released by a thread that does not hold it.");
        }
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
    /// holds; <paramref name="largest"/> is the largest thread id in that chain, <paramref name="self"/> included.</summary>
    private bool LeadsBackTo(int self, out int largest)
    {
        largest = self;
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
            largest = Math.Max(largest, owner);
            if (!_waiting.TryGetValue(owner, out ResidencyLock? next))
            {
                return false;
            }
            current = next;
        }
        return false;
    }
}
