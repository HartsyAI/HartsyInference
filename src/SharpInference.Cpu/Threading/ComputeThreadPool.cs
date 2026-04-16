namespace SharpInference.Cpu.Threading;

/// <summary>Lightweight compute thread pool for parallelizing CPU kernel work across cores. Uses <see cref="Parallel.For"/> with a fixed degree of parallelism to avoid managed thread pool contention.</summary>
public sealed class ComputeThreadPool
{
    private static readonly Lazy<ComputeThreadPool> _instance = new(() => new ComputeThreadPool());

    /// <summary>Shared singleton instance.</summary>
    public static ComputeThreadPool Shared => _instance.Value;

    /// <summary>Number of worker threads (defaults to processor count).</summary>
    public int ThreadCount { get; }

    private readonly ParallelOptions _parallelOptions;

    /// <summary>Creates a compute thread pool with the specified thread count.</summary>
    public ComputeThreadPool(int? threadCount = null)
    {
        ThreadCount = threadCount ?? Environment.ProcessorCount;
        _parallelOptions = new ParallelOptions
        {
            MaxDegreeOfParallelism = ThreadCount,
        };
    }

    /// <summary>Executes a parallel for loop over the range [0, count) with the pool's thread count.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void ParallelFor(int fromInclusive, int toExclusive, Action<int> body)
    {
        if (toExclusive - fromInclusive <= 1)
        {
            for (int i = fromInclusive; i < toExclusive; i++)
                body(i);
            return;
        }

        Parallel.For(fromInclusive, toExclusive, _parallelOptions, body);
    }

    /// <summary>Executes a parallel for loop with a local state accumulator for reduction patterns.</summary>
    public void ParallelFor<TLocal>(
        int fromInclusive,
        int toExclusive,
        Func<TLocal> localInit,
        Func<int, ParallelLoopState, TLocal, TLocal> body,
        Action<TLocal> localFinally)
    {
        Parallel.For(fromInclusive, toExclusive, _parallelOptions, localInit, body, localFinally);
    }
}
