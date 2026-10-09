using System.Runtime.ExceptionServices;
using HartsyInference.Core.Numerics;

namespace HartsyInference.Cpu.Moe;

/// <summary>
/// A persistent pool of CPU workers that runs a batch of independent expert jobs, with the calling thread taking part.
/// </summary>
/// <remarks>
/// <para><b>API.</b> <see cref="Run"/> takes a job count and an <see cref="Action{T}"/> of <see cref="int"/> that is called
/// once per index in <c>[0, jobCount)</c>. The delegate is the caller's to keep: create it once and store it in a field,
/// and a steady-state <see cref="Run"/> allocates nothing. A capturing lambda created at the call site allocates a closure
/// and delegate each time, which is the one thing this type does not do for you. The simplest shape that meets the
/// zero-allocation requirement was a delegate rather than a generic job struct, because a struct cannot be stored in a
/// field of this non-generic pool without boxing.</para>
///
/// <para><b>Workers.</b> The pool starts <see cref="Parallelism"/> minus one background threads. The calling thread is the
/// remaining participant, so a pool with <see cref="Parallelism"/> of 1 has no threads at all and still completes every
/// job. <see cref="Parallelism"/> is the requested count clamped to <see cref="CpuParallel.MaxThreads"/>, which is the
/// <c>numerics.cpuThreads</c> knob when set and the logical core count otherwise. That is the same ceiling every other CPU
/// kernel uses; this type deliberately does not detect cores itself. The ceiling is read once, at construction.</para>
///
/// <para><b>Protocol.</b> Each <see cref="Run"/> publishes a new epoch. A worker spins briefly on the epoch word, then parks on
/// a monitor. Items are handed out by an atomic counter, so each index is claimed by exactly one participant. Every worker
/// acknowledges every epoch, even when it claims nothing, and <see cref="Run"/> returns only after all of them have, so no
/// worker can still be inside a job when the next one begins.</para>
///
/// <para><b>Failures.</b> The first exception thrown by a job is recorded, the remaining unclaimed items are skipped, and that
/// exception is rethrown on the caller once the job has fully drained. The pool stays usable for the next <see cref="Run"/>.</para>
///
/// <para><b>Threading rules.</b> <see cref="Run"/> is serialised: concurrent callers queue behind one another. It is not
/// reentrant, so a job must not call <see cref="Run"/> on the same pool, and it must not call <see cref="Dispose"/>.
/// <see cref="Dispose"/> waits for an in-flight <see cref="Run"/> to finish, stops and joins the workers, and makes every
/// later <see cref="Run"/> throw <see cref="ObjectDisposedException"/>. A second <see cref="Dispose"/> does nothing.</para>
/// </remarks>
public sealed class CpuExpertPool : IDisposable
{
    // Spin iterations before a waiting thread parks. Each iteration is a short SpinWait, so this is a few tens of
    // microseconds, enough to catch back-to-back Run calls without the wake-up cost, and short enough that an idle pool
    // falls asleep quickly.
    private const int SpinIterations = 2000;
    private const int SpinCycles = 8;

    // Parks and wakes both workers (waiting for an epoch) and the caller (waiting for the workers to acknowledge).
    private readonly object _gate = new();

    // Serialises Run and Dispose, and guards the reentrancy flag. Never held while a job executes on another thread.
    private readonly object _callerLock = new();
    private readonly Thread[] _workers;

    // Job state. Written by the caller before it publishes an epoch and read by workers only after they observe that
    // epoch, so the epoch word is the release/acquire point for all of these.
    private Action<int>? _body;
    private int _count;
    private int _next;
    private int _pending;
    private Exception? _error;

    // Published by the caller with Volatile.Write; observed by workers with Volatile.Read.
    private long _epoch;
    private volatile bool _disposed;
    private bool _inRun;

    /// <summary>Creates the pool and starts its workers.</summary>
    /// <param name="maxParticipants">Threads that take part in a job, including the caller. Null means the
    /// <see cref="CpuParallel.MaxThreads"/> ceiling. Larger values are clamped to that ceiling.</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="maxParticipants"/> is below 1.</exception>
    public CpuExpertPool(int? maxParticipants = null)
    {
        int requested = maxParticipants ?? CpuParallel.MaxThreads;
        ArgumentOutOfRangeException.ThrowIfLessThan(requested, 1, nameof(maxParticipants));
        Parallelism = Math.Min(requested, CpuParallel.MaxThreads);
        _workers = new Thread[Parallelism - 1];
        for (int i = 0; i < _workers.Length; i++)
        {
            _workers[i] = new Thread(WorkerLoop)
            {
                IsBackground = true,
                Name = "CpuExpertPool worker " + i,
            };
            _workers[i].Start();
        }
    }

    /// <summary>Threads that take part in each job, including the caller. Fixed at construction.</summary>
    public int Parallelism { get; }

    /// <summary>Calls <paramref name="body"/> once for each index in <c>[0, jobCount)</c>, spread across the pool.</summary>
    /// <param name="jobCount">Number of independent items. Zero returns immediately.</param>
    /// <param name="body">Called once per index, possibly on any participant. Each call must write only its own output.
    /// Keep this delegate in a field so repeated calls do not allocate.</param>
    /// <exception cref="ObjectDisposedException">The pool has been disposed.</exception>
    /// <exception cref="InvalidOperationException">Called from inside a job on this same pool.</exception>
    /// <exception cref="Exception">The first exception thrown by <paramref name="body"/>, rethrown after the job has drained.</exception>
    public void Run(int jobCount, Action<int> body)
    {
        ArgumentNullException.ThrowIfNull(body);
        ArgumentOutOfRangeException.ThrowIfNegative(jobCount);
        lock (_callerLock)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (_inRun)
            {
                throw new InvalidOperationException("CpuExpertPool.Run is not reentrant; a job must not start another job on its pool.");
            }
            if (jobCount == 0)
            {
                return;
            }
            _inRun = true;
            try
            {
                Execute(jobCount, body);
            }
            finally
            {
                _inRun = false;
            }
        }
    }

    /// <summary>Stops and joins the workers. Waits for any <see cref="Run"/> in progress to finish first.</summary>
    /// <remarks>Calling it from inside a job is not supported.</remarks>
    public void Dispose()
    {
        lock (_callerLock)
        {
            if (_disposed)
            {
                return;
            }
            _disposed = true;
        }
        lock (_gate)
        {
            Monitor.PulseAll(_gate);
        }
        foreach (Thread worker in _workers)
        {
            worker.Join();
        }
    }

    private void Execute(int jobCount, Action<int> body)
    {
        _body = body;
        _count = jobCount;
        _next = 0;
        _error = null;
        Volatile.Write(ref _pending, _workers.Length);
        Volatile.Write(ref _epoch, _epoch + 1);
        if (_workers.Length > 0)
        {
            lock (_gate)
            {
                Monitor.PulseAll(_gate);
            }
        }

        RunClaims(body, jobCount);
        AwaitWorkers();

        _body = null;
        Exception? error = _error;
        _error = null;
        if (error is not null)
        {
            ExceptionDispatchInfo.Throw(error);
        }
    }

    // Claims indices until none are left or a job has failed. A failure is recorded once; the first one wins.
    private void RunClaims(Action<int> body, int count)
    {
        while (Volatile.Read(ref _error) is null)
        {
            int index = Interlocked.Increment(ref _next) - 1;
            if (index >= count)
            {
                return;
            }
            try
            {
                body(index);
            }
            catch (Exception ex)
            {
                Interlocked.CompareExchange(ref _error, ex, null);
                return;
            }
        }
    }

    private void AwaitWorkers()
    {
        for (int spin = 0; spin < SpinIterations; spin++)
        {
            if (Volatile.Read(ref _pending) == 0)
            {
                return;
            }
            Thread.SpinWait(SpinCycles);
        }
        lock (_gate)
        {
            while (Volatile.Read(ref _pending) != 0)
            {
                Monitor.Wait(_gate);
            }
        }
    }

    private void WorkerLoop()
    {
        // Starts at 0, the epoch the pool was created at, not at whatever _epoch holds now. The thread may not be
        // scheduled until after the first Run has published epoch 1; reading _epoch here would make it skip that job.
        long seen = 0;
        while (AwaitNewEpoch(ref seen))
        {
            Action<int>? body = _body;
            int count = _count;
            if (body is not null)
            {
                RunClaims(body, count);
            }
            if (Interlocked.Decrement(ref _pending) == 0)
            {
                lock (_gate)
                {
                    Monitor.PulseAll(_gate);
                }
            }
        }
    }

    // Returns false once the pool is disposed. Otherwise it returns true with seen set to the new epoch.
    private bool AwaitNewEpoch(ref long seen)
    {
        for (int spin = 0; spin < SpinIterations; spin++)
        {
            if (Volatile.Read(ref _epoch) != seen || _disposed)
            {
                break;
            }
            Thread.SpinWait(SpinCycles);
        }
        lock (_gate)
        {
            while (!_disposed && Volatile.Read(ref _epoch) == seen)
            {
                Monitor.Wait(_gate);
            }
        }
        if (_disposed)
        {
            return false;
        }
        seen = Volatile.Read(ref _epoch);
        return true;
    }
}
