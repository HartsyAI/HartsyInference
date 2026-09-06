using System.Runtime.ExceptionServices;
using HartsyInference.Core.Configuration;

namespace HartsyInference.Core.Numerics;

/// <summary>Fans a kernel's outermost loop across cores, with one degree-of-parallelism ceiling for the whole
/// process.</summary>
/// <remarks>CPU kernels here were written single-threaded, which on a many-core host leaves every core but one
/// idle for the whole of a synthesis or a decode. This is the shared dispatch they fan out through, rather than
/// each kernel growing its own <see cref="Parallel.For"/> with its own thread count and its own threshold.
///
/// <para>One ceiling matters because the CPU device gate is a no-op: an LLM decode and a TTS synthesis can be in
/// flight at the same moment on the same box, and two independent <see cref="Parallel.For"/> calls would each
/// claim every core. <see cref="EngineKnobs.CpuThreads"/> is what a host turns down to leave room.</para>
///
/// <para>The work threshold is not optional. Dispatch was measured at ~22 µs for 16 chunks on a 16-core box (see
/// <c>CheckpointConvertUtils</c>'s fp8 passes, which pay the same cost), so a small convolution finishes sooner
/// on one core than it takes to hand out the work. Callers pass the total scalar operations they are about to
/// perform, and anything under <see cref="MinWorkForParallel"/> runs inline.</para></remarks>
public static class CpuParallel
{
    /// <summary>Total scalar operations below which a call runs serially rather than paying dispatch.</summary>
    /// <remarks>64 k multiply-accumulates is roughly 30-60 µs of scalar work, comfortably above the measured
    /// dispatch cost while still catching every layer that matters in a vocoder.</remarks>
    public const long MinWorkForParallel = 64 * 1024;

    /// <summary>Worker cap: the knob when set, otherwise every core.</summary>
    public static int MaxThreads
    {
        get
        {
            int configured = EngineKnobs.CpuThreads.Value;
            return configured > 0 ? Math.Min(configured, Environment.ProcessorCount) : Environment.ProcessorCount;
        }
    }

    /// <summary>Runs <paramref name="body"/> over <c>[0, count)</c>, in parallel only when it is worth it.</summary>
    /// <param name="count">Number of independent work items. Each index must write a disjoint output region —
    /// that is the caller's guarantee, and the reason a scatter kernel has to be re-expressed as a gather before
    /// it can come through here.</param>
    /// <param name="totalWork">Total scalar operations across all items, used against
    /// <see cref="MinWorkForParallel"/>. An estimate is fine; it only picks the branch.</param>
    /// <param name="body">Called once per index.</param>
    /// <remarks>An exception from a worker is rethrown as itself, not wrapped in an
    /// <see cref="AggregateException"/>, so the parallel path keeps the serial path's exception contract and a
    /// caller's <c>catch (ArgumentException)</c> keeps working.</remarks>
    public static void For(int count, long totalWork, Action<int> body)
    {
        ArgumentNullException.ThrowIfNull(body);
        int threads = MaxThreads;
        if (count <= 1 || threads <= 1 || totalWork < MinWorkForParallel)
        {
            for (int i = 0; i < count; i++)
            {
                body(i);
            }
            return;
        }
        try
        {
            Parallel.For(0, count, new ParallelOptions { MaxDegreeOfParallelism = threads }, body);
        }
        catch (AggregateException ex) when (ex.InnerException is not null)
        {
            ExceptionDispatchInfo.Capture(ex.InnerException).Throw();
        }
    }

    /// <summary>Splits <paramref name="length"/> into roughly <paramref name="targetChunks"/> contiguous chunks,
    /// returning the chunk size.</summary>
    /// <remarks>Used by kernels whose natural outer dimension is too small to fill the machine — a final vocoder
    /// convolution has one output channel and tens of thousands of samples, so the time axis has to be split as
    /// well or that layer stays serial while every other one scales.</remarks>
    public static int ChunkSize(int length, int targetChunks)
    {
        if (targetChunks <= 1 || length <= targetChunks)
        {
            return Math.Max(1, length);
        }
        return (length + targetChunks - 1) / targetChunks;
    }

    /// <summary>How many chunks to aim for: a few per worker, so an uneven split still balances.</summary>
    public static int TargetChunks(int rows)
    {
        int threads = MaxThreads;
        if (threads <= 1)
        {
            return 1;
        }
        // Four tasks per worker is the usual load-balancing compromise: enough that a slow chunk is absorbed by
        // its neighbours, few enough that dispatch stays a rounding error.
        int want = threads * 4;
        return rows >= want ? 1 : Math.Max(1, want / Math.Max(1, rows));
    }
}
