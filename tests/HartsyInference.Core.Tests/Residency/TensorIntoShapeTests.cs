using HartsyInference.Core.Backends.Gpu;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Core.Tests.Residency;

/// <summary><see cref="Tensor.IntoShape"/>: the storage, the device buffer and its pin move to the new tensor, with no transfer. The device is a
/// fake residency cache whose buffers are arrays, so this runs without a GPU on the code both GPU backends share.</summary>
public sealed class TensorIntoShapeTests
{
    [Fact]
    public void DeviceResult_MovesWithoutATransfer_AndReadsBackOnceUnderTheNewShape()
    {
        using FakeResidency device = new();
        Tensor source = new(new TensorShape(2, 3), DType.F32);
        float[] buffer = device.Produce(source, [1, 2, 3, 4, 5, 6]);

        using Tensor moved = source.IntoShape(new TensorShape(3, 2));

        Assert.Equal(0, device.D2hSyncCount);
        Assert.False(moved.HostBytesCurrent);
        Assert.Same(buffer, device.CopyToDevice(moved));
        Assert.Equal(0, device.Misses);
        Assert.Equal(new float[] { 1, 2, 3, 4, 5, 6 }, moved.AsReadOnlySpan<float>().ToArray());
        Assert.Equal(1, device.D2hSyncCount);
        Assert.Equal(1, device.Freed);
    }

    [Fact]
    public void Reshape_StillReadsTheDeviceResultBack()
    {
        // The contrast IntoShape exists for; if this stops holding, the remark on IntoShape is out of date.
        using FakeResidency device = new();
        using Tensor source = new(new TensorShape(2, 3), DType.F32);
        device.Produce(source, [1, 2, 3, 4, 5, 6]);

        using Tensor view = source.Reshape(new TensorShape(3, 2));

        Assert.Equal(1, device.D2hSyncCount);
        device.CopyToDevice(view);
        Assert.Equal(1, device.Misses);
    }

    [Fact]
    public void DisposingTheMovedTensor_FreesTheBufferOnce_AndTheSourceIsSpent()
    {
        using FakeResidency device = new();
        Tensor source = new(new TensorShape(4), DType.F32);
        device.Produce(source, [1, 2, 3, 4]);

        Tensor moved = source.IntoShape(new TensorShape(2, 2));
        source.Dispose();
        Assert.Equal(0, device.Freed);
        Assert.Throws<ObjectDisposedException>(() => source.AsSpan<float>().Length);
        Assert.Throws<ObjectDisposedException>(() => source.IntoShape(new TensorShape(4)));

        moved.Dispose();
        Assert.Equal(1, device.Freed);
    }

    [Fact]
    public void APinMovesWithTheBuffer()
    {
        using FakeResidency device = new();
        Tensor source = new(new TensorShape(4), DType.F32);
        device.Produce(source, [1, 2, 3, 4]);
        device.PinActivation(source);

        using Tensor moved = source.IntoShape(new TensorShape(1, 4));
        device.FreeActivations();

        Assert.Equal(0, device.Freed);
        Assert.Equal(new float[] { 1, 2, 3, 4 }, moved.AsReadOnlySpan<float>().ToArray());
    }

    [Fact]
    public void AnUnpinnedMovedTensor_IsFreedByTheActivationSweep()
    {
        using FakeResidency device = new();
        Tensor source = new(new TensorShape(4), DType.F32);
        device.Produce(source, [1, 2, 3, 4]);
        using Tensor moved = source.IntoShape(new TensorShape(2, 2));

        device.FreeActivations();

        Assert.Equal(1, device.Freed);
        Assert.True(moved.HostBytesCurrent);
    }

    [Fact]
    public void HostData_MovesWithoutACopy()
    {
        Tensor source = new(new TensorShape(2, 2), DType.F32);
        float[] values = [1, 2, 3, 4];
        values.CopyTo(source.AsSpan<float>());
        nint pointer = Pointer(source);

        using Tensor moved = source.IntoShape(new TensorShape(4));

        Assert.Equal(pointer, Pointer(moved));
        Assert.Equal(values, moved.AsReadOnlySpan<float>().ToArray());
        Assert.True(moved.OwnsMemory);
    }

    [Fact]
    public void APromotedWeight_KeepsItsHostBytes()
    {
        // A promoted weight's binding drops the device copy and cannot move: the host is authoritative.
        using FakeResidency device = new();
        Tensor source = new(new TensorShape(2, 2), DType.F32);
        float[] values = [1, 2, 3, 4];
        values.CopyTo(source.AsSpan<float>());
        device.Promote(source);

        using Tensor moved = source.IntoShape(new TensorShape(4));

        Assert.Equal(0, device.D2hSyncCount);
        Assert.Equal(1, device.Freed);
        Assert.Equal(values, moved.AsReadOnlySpan<float>().ToArray());
    }

    [Fact]
    public void BorrowedMemory_IsReshapedAndTheSourceStaysValid()
    {
        using Tensor owner = new(new TensorShape(4), DType.F32);
        float[] values = [1, 2, 3, 4];
        values.CopyTo(owner.AsSpan<float>());
        Tensor borrowed = Borrow(owner);

        using Tensor moved = borrowed.IntoShape(new TensorShape(2, 2));

        Assert.Equal(values, moved.AsReadOnlySpan<float>().ToArray());
        Assert.Equal(values, borrowed.AsReadOnlySpan<float>().ToArray());
    }

    [Fact]
    public void AWrongElementCount_IsRefusedAndLeavesTheTensorUsable()
    {
        using FakeResidency device = new();
        using Tensor source = new(new TensorShape(4), DType.F32);
        device.Produce(source, [1, 2, 3, 4]);

        Assert.Throws<HartsyInferenceException>(() => source.IntoShape(new TensorShape(3)));

        Assert.Equal(new float[] { 1, 2, 3, 4 }, source.AsReadOnlySpan<float>().ToArray());
    }

    [Fact]
    public void PerRowQuantization_IsRefusedWhenTheRowsChange_AndLeavesTheTensorUsable()
    {
        using Tensor scale = new(new TensorShape(2), DType.F32);
        using Tensor source = new(new TensorShape(2, 4), DType.I8);
        source.QuantInfo = new QuantWeightInfo { Format = "int8_tensorwise", RowScale = scale };

        Assert.Throws<HartsyInferenceException>(() => source.IntoShape(new TensorShape(4, 2)));

        Assert.Equal(8, source.AsSpan<sbyte>().Length);
    }

    private static unsafe nint Pointer(Tensor tensor) => (nint)tensor.DataPointer;

    private static unsafe Tensor Borrow(Tensor owner) => new(owner.DataPointer, owner.Shape, owner.DType);

    /// <summary>A residency cache over managed arrays.</summary>
    private sealed class FakeResidency : GpuResidencyCache<float[]>
    {
        /// <summary>Device buffers handed back so far.</summary>
        public int Freed { get; private set; }

        /// <summary>Leaves <paramref name="values"/> as an op's output would: on the device only.</summary>
        public float[] Produce(Tensor output, float[] values)
        {
            CacheActivation(output, values, ByteSize(output));
            return values;
        }

        /// <summary>Makes a host tensor a weight the cache promoted on its own.</summary>
        public void Promote(Tensor tensor) => PromoteToWeight(tensor, tensor.AsReadOnlySpan<float>().ToArray());

        protected override float[] AllocateDevice(long bytes) => new float[bytes / sizeof(float)];

        protected override void FreeDevice(float[] buffer, long bytes) => Freed++;

        protected override void Upload(float[] destination, Tensor source, long bytes) => source.AsReadOnlySpan<float>().CopyTo(destination);

        protected override unsafe void DownloadSynced(nint hostDestination, float[] source, long bytes) =>
            source.AsSpan().CopyTo(new Span<float>((void*)hostDestination, source.Length));

        protected override void MakeCurrent() { }
    }
}
