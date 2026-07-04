using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Core.Tests;

/// <summary>Guards the context-partitioned finalizer cleanup queue. Each GPU backend's tensor-finalizer
/// callbacks free against, and mutate the unsynchronized per-context state of, exactly one context. The
/// keyed drain must run ONLY the owning context's callbacks — otherwise, with two GPU backends live, one
/// backend's drain thread runs the other's cleanup concurrently with that backend's own thread, racing on
/// non-thread-safe state (the multi-GPU corruption/leak this partition fixes).</summary>
public sealed class TensorFinalizerCleanupTests
{
    // Distinctive fake context handles that can't collide with a real primary-context pointer.
    private static readonly nint CtxA = 0x7A11_0001;
    private static readonly nint CtxB = 0x7A11_0002;

    [Fact]
    public void KeyedDrain_RunsOnlyOwningContextsCallbacks_InOrder()
    {
        List<string> ran = [];
        Tensor.EnqueueFinalizerGpuCleanup(CtxA, () => ran.Add("a1"));
        Tensor.EnqueueFinalizerGpuCleanup(CtxB, () => ran.Add("b1"));
        Tensor.EnqueueFinalizerGpuCleanup(CtxA, () => ran.Add("a2"));

        // Draining A runs A's two callbacks (FIFO) and leaves B's untouched.
        Tensor.DrainPendingFinalizerGpuCleanup(CtxA);
        Assert.Equal(["a1", "a2"], ran);

        // Idempotent: A's bucket is now empty, so a second drain is a no-op.
        Tensor.DrainPendingFinalizerGpuCleanup(CtxA);
        Assert.Equal(["a1", "a2"], ran);

        // B's callback only runs when B's own context drains.
        Tensor.DrainPendingFinalizerGpuCleanup(CtxB);
        Assert.Equal(["a1", "a2", "b1"], ran);
    }

    [Fact]
    public void KeyedDrain_UnknownContext_IsNoOp()
    {
        bool ran = false;
        Tensor.EnqueueFinalizerGpuCleanup(CtxA, () => ran = true);
        Tensor.DrainPendingFinalizerGpuCleanup((nint)0x7A11_9999); // no bucket for this key
        Assert.False(ran);
        Tensor.DrainPendingFinalizerGpuCleanup(CtxA); // clean up so we don't leak into other tests
        Assert.True(ran);
    }

    [Fact]
    public void DrainAll_RunsEveryContextsCallbacks()
    {
        bool a = false, b = false;
        Tensor.EnqueueFinalizerGpuCleanup(CtxA, () => a = true);
        Tensor.EnqueueFinalizerGpuCleanup(CtxB, () => b = true);
        Tensor.DrainPendingFinalizerGpuCleanup(); // drain-all overload
        Assert.True(a);
        Assert.True(b);
    }
}
