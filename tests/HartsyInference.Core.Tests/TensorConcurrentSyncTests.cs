using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Core.Tests;

/// <summary>Guards <c>EnsureCpuData</c>'s primary-slot claim against the double-invoke race: before the fix, the
/// primary sync callback was read/cleared with a plain field access while every other mutator of the same fields
/// (<c>SetGpuBinding</c>/<c>ClearGpuBinding</c>) held <c>lock(this)</c>. Two threads racing <c>.DataPointer</c> on
/// a tensor with a live GPU binding — exactly what CFG-branch parallelism does to a shared weight tensor mid-step
/// if a cross-backend eviction reopens the cache-miss window — could both observe the same non-null callback and
/// both invoke it: a double D2H-sync / double-free of the same GPU pointer.</summary>
public sealed unsafe class TensorConcurrentSyncTests
{
    [Fact]
    public void EnsureCpuData_ConcurrentDataPointerReads_InvokesPrimarySyncExactlyOnce()
    {
        const int iterations = 300;
        const int threadCount = 8;
        for (int iter = 0; iter < iterations; iter++)
        {
            using Tensor tensor = new Tensor(new TensorShape(4), DType.F32, DeviceKind.Cpu);
            int syncCount = 0;
            tensor.SetGpuBinding((nint)1, () => Interlocked.Increment(ref syncCount), () => { });

            using Barrier barrier = new Barrier(threadCount);
            Thread[] threads = new Thread[threadCount];
            for (int t = 0; t < threadCount; t++)
            {
                threads[t] = new Thread(() =>
                {
                    barrier.SignalAndWait();
                    _ = tensor.DataPointer;
                });
                threads[t].Start();
            }
            foreach (Thread th in threads)
            {
                th.Join();
            }

            Assert.Equal(1, syncCount);
        }
    }

    /// <summary>Same race, but with a second backend's overflow binding present too (the CFG-parallel shape: a
    /// weight promoted on both Backend and CfgParallelBackend) — both the primary claim and the overflow-list
    /// drain (already lock-protected via <c>TakeExtraBindings</c>) must survive concurrent readers.</summary>
    [Fact]
    public void EnsureCpuData_ConcurrentDataPointerReads_WithTwoBackendBindings_InvokesEachExactlyOnce()
    {
        const int iterations = 300;
        const int threadCount = 8;
        for (int iter = 0; iter < iterations; iter++)
        {
            using Tensor tensor = new Tensor(new TensorShape(4), DType.F32, DeviceKind.Cpu);
            int syncCountA = 0, syncCountB = 0;
            tensor.SetGpuBinding((nint)1, () => Interlocked.Increment(ref syncCountA), () => { });
            tensor.SetGpuBinding((nint)2, () => Interlocked.Increment(ref syncCountB), () => { });

            using Barrier barrier = new Barrier(threadCount);
            Thread[] threads = new Thread[threadCount];
            for (int t = 0; t < threadCount; t++)
            {
                threads[t] = new Thread(() =>
                {
                    barrier.SignalAndWait();
                    _ = tensor.DataPointer;
                });
                threads[t].Start();
            }
            foreach (Thread th in threads)
            {
                th.Join();
            }

            Assert.Equal(1, syncCountA);
            Assert.Equal(1, syncCountB);
        }
    }
}
