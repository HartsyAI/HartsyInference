using System.Reflection;
using HartsyInference.Core.Memory;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Core.Tests;

public sealed unsafe class TensorDisposeTests
{
    [Fact]
    public void Dispose_PrimaryGpuCleanupFailure_StillReleasesOtherBindingsAndHostBuffer()
    {
        Tensor tensor = new(new TensorShape(16), DType.F32);
        _ = tensor.DataPointer; // force the lazy owned host allocation
        NativeBuffer buffer = Assert.IsType<NativeBuffer>(typeof(Tensor)
            .GetField("_ownedBuffer", BindingFlags.Instance | BindingFlags.NonPublic)!
            .GetValue(tensor));
        int secondaryCleanupCount = 0;
        InvalidOperationException expected = new("injected primary cleanup failure");
        tensor.SetGpuBinding((nint)1, () => { }, () => throw expected);
        tensor.SetGpuBinding((nint)2, () => { }, () => secondaryCleanupCount++);

        InvalidOperationException actual = Assert.Throws<InvalidOperationException>(tensor.Dispose);

        Assert.Same(expected, actual);
        Assert.Equal(1, secondaryCleanupCount);
        Assert.Throws<ObjectDisposedException>(() => { _ = buffer.Pointer; });
        tensor.Dispose(); // idempotent after a faulting first disposal

        // A backend can finish publishing a cache entry just after Tensor disposal begins. The late binding must
        // roll that allocation back immediately instead of being planted on an object whose finalizer is suppressed.
        tensor.SetGpuBinding((nint)3, () => { }, () => secondaryCleanupCount++);
        Assert.Equal(2, secondaryCleanupCount);
    }
}
