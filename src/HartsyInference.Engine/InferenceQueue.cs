namespace HartsyInference.Engine;

/// <summary>Thrown when the inference queue is full; the endpoint maps it to HTTP 429.</summary>
public sealed class QueueFullException(string message) : Exception(message);

/// <summary>Bounds concurrent inference. Up to <c>MaxConcurrency</c> requests run at once; up to
/// <c>MaxQueueDepth</c> more may wait; further requests are rejected with <see cref="QueueFullException"/>.
/// Inference (especially diffusion) is GPU/CPU-bound and not safely re-entrant per backend, so the
/// default concurrency is 1.</summary>
public sealed class InferenceQueue : IDisposable
{
    private readonly SemaphoreSlim _slots;
    private readonly int _capacity;
    private int _pending;

    /// <summary>Creates a queue with the given concurrency and queue depth.</summary>
    public InferenceQueue(int maxConcurrency, int maxQueueDepth)
    {
        int concurrency = Math.Max(1, maxConcurrency);
        _slots = new SemaphoreSlim(concurrency, concurrency);
        _capacity = concurrency + Math.Max(0, maxQueueDepth);
    }

    /// <summary>Runs <paramref name="work"/> under a concurrency slot, queuing if necessary. Throws
    /// <see cref="QueueFullException"/> immediately if the queue is at capacity.</summary>
    public async Task<T> EnqueueAsync<T>(Func<Task<T>> work, CancellationToken ct)
    {
        if (Interlocked.Increment(ref _pending) > _capacity)
        {
            Interlocked.Decrement(ref _pending);
            throw new QueueFullException("Inference queue is full. Retry later.");
        }

        try
        {
            await _slots.WaitAsync(ct).ConfigureAwait(false);
            try
            {
                return await work().ConfigureAwait(false);
            }
            finally
            {
                _slots.Release();
            }
        }
        finally
        {
            Interlocked.Decrement(ref _pending);
        }
    }

    /// <summary>Current number of requests admitted (running + queued-and-waiting) — for observability/logging.</summary>
    public int PendingCount => Volatile.Read(ref _pending);

    public void Dispose() => _slots.Dispose();
}
