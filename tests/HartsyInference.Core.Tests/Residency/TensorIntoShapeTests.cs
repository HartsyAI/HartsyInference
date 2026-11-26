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
        using ArrayResidency device = new();
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
        using ArrayResidency device = new();
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
        using ArrayResidency device = new();
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
        using ArrayResidency device = new();
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
        using ArrayResidency device = new();
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
        using ArrayResidency device = new();
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
        using ArrayResidency device = new();
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

    [Fact]
    public void ATensorHeldOnTwoDevices_IsReadBackFirst()
    {
        using ArrayResidency first = new();
        using ArrayResidency second = new();
        Tensor source = new(new TensorShape(4), DType.F32);
        float[] values = [1, 2, 3, 4];
        // The first binding is the movable kind, so only the second one stops the move.
        first.Produce(source, [.. values]);
        second.Promote(source, [.. values]);

        using Tensor moved = source.IntoShape(new TensorShape(2, 2));

        Assert.Equal(1, first.D2hSyncCount);
        Assert.Equal(1, first.Freed);
        Assert.Equal(1, second.Freed);
        Assert.Equal(values, moved.AsReadOnlySpan<float>().ToArray());
    }

    [Fact]
    public void AViewTakenEarlier_KeepsTheResultAlive()
    {
        Tensor source = new(new TensorShape(4), DType.F32);
        using Tensor view = source.Reshape(new TensorShape(2, 2));

        using Tensor moved = source.IntoShape(new TensorShape(1, 4));

        Assert.Same(source, view.KeepAliveOwner);
        Assert.Same(moved, source.KeepAliveOwner);
    }

    [Fact]
    public void ReadingAViewWrittenAfterTheMove_MakesTheResultCurrent()
    {
        using ArrayResidency device = new();
        Tensor source = new(new TensorShape(4), DType.F32);
        using Tensor view = source.Reshape(new TensorShape(2, 2));
        using Tensor moved = source.IntoShape(new TensorShape(1, 4));

        device.Produce(view, [1, 2, 3, 4]);

        Assert.Equal(new float[] { 1, 2, 3, 4 }, view.AsReadOnlySpan<float>().ToArray());
        Assert.Equal(new float[] { 1, 2, 3, 4 }, moved.AsReadOnlySpan<float>().ToArray());
    }

#if DEBUG
    [Fact]
    public void AViewWrittenAfterTheMove_MarksTheResult()
    {
        // The view roots the spent tensor, which roots the result: the device write reaches the tensor that now holds the bytes.
        using ArrayResidency device = new();
        Tensor source = new(new TensorShape(4), DType.F32);
        using Tensor view = source.Reshape(new TensorShape(2, 2));
        using Tensor moved = source.IntoShape(new TensorShape(1, 4));

        device.Produce(view, [1, 2, 3, 4]);

        Assert.Contains("wrote a view of it", Assert.Throws<InvalidOperationException>(() => moved.AsReadOnlySpan<float>().Length).Message);
    }

    [Fact]
    public void MovingAnOwnerBehindAViewWrite_IsReported_AndLeavesTheOwner()
    {
        using ArrayResidency device = new();
        using Tensor owner = new(new TensorShape(2, 2), DType.F32);
        using (Tensor view = owner.Reshape(new TensorShape(4)))
            device.Produce(view, [1, 2, 3, 4]);

        Assert.Throws<InvalidOperationException>(() => owner.IntoShape(new TensorShape(4)));

        Assert.Equal(4, owner.AsReadOnlySpan<float>().Length);
    }

    [Fact]
    public void MovingADeviceOwnerBehindALaterViewWrite_IsReported_AndLeavesTheOwner()
    {
        // The owner's own device copy is bound first, so the mark the view write sets is not cleared by a bind before the move.
        using ArrayResidency device = new();
        using Tensor owner = new(new TensorShape(2, 2), DType.F32);
        using Tensor view = owner.Reshape(new TensorShape(4));
        device.Produce(owner, [1, 2, 3, 4]);
        device.Produce(view, [5, 6, 7, 8]);

        Assert.Contains("wrote a view of it", Assert.Throws<InvalidOperationException>(() => owner.IntoShape(new TensorShape(4))).Message);

        Assert.Equal(4, owner.AsReadOnlySpan<float>().Length);
    }
#endif

    private static unsafe nint Pointer(Tensor tensor) => (nint)tensor.DataPointer;

    private static unsafe Tensor Borrow(Tensor owner) => new(owner.DataPointer, owner.Shape, owner.DType);
}
