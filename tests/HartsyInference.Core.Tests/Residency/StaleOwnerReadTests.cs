using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Core.Tests.Residency;

/// <summary>A backend op that writes a view leaves the viewed tensor's host bytes old. Debug builds throw when the owner is read in that state
/// (the SAM 3 and FastFrames decoder bugs, which no CPU test can see); Release builds log it once, so these cases are Debug only.</summary>
public sealed class StaleOwnerReadTests
{
    private static readonly float[] Values = [1, 2, 3, 4];

#if DEBUG
    [Fact]
    public void ReadingTheOwner_AfterTheViewWasWrittenAndDisposed_Throws()
    {
        // The FastFrames decoder shape: the output is written through a `using` view and the owner is returned.
        using ArrayResidency device = new();
        using Tensor owner = new(new TensorShape(2, 2), DType.F32);
        using (Tensor view = owner.Reshape(new TensorShape(4)))
            device.Produce(view, [.. Values]);

        InvalidOperationException error = Assert.Throws<InvalidOperationException>(() => owner.AsReadOnlySpan<float>().Length);
        Assert.Contains("wrote a view of it", error.Message);
    }

    [Fact]
    public void UploadingTheOwner_ForTheNextOp_Throws()
    {
        using ArrayResidency device = new();
        using Tensor owner = new(new TensorShape(2, 2), DType.F32);
        using Tensor view = owner.Reshape(new TensorShape(4));
        device.Produce(view, [.. Values]);

        Assert.Throws<InvalidOperationException>(() => device.CopyToDevice(owner));
    }

    [Fact]
    public void AViewOfAView_AndARowSlice_MarkTheRoot()
    {
        using ArrayResidency device = new();
        using Tensor owner = new(new TensorShape(2, 2), DType.F32);
        using Tensor view = owner.Reshape(new TensorShape(4, 1));
        using Tensor rows = view.SliceRows(2, 2);
        device.Produce(rows, [7, 8]);

        Assert.Throws<InvalidOperationException>(() => owner.AsReadOnlySpan<float>().Length);
    }
#endif

    [Fact]
    public void ReadingTheViewFirst_MakesTheOwnerCurrent()
    {
        using ArrayResidency device = new();
        using Tensor owner = new(new TensorShape(2, 2), DType.F32);
        using Tensor view = owner.Reshape(new TensorShape(4));
        device.Produce(view, [.. Values]);

        Assert.Equal(Values, view.AsReadOnlySpan<float>().ToArray());
        Assert.Equal(Values, owner.AsReadOnlySpan<float>().ToArray());
    }

    [Fact]
    public void WritingAViewInPlaceAndReturningIt_IsAllowed()
    {
        // Sam3DotProductScorer: the view is the result and the owner is never read again.
        using ArrayResidency device = new();
        using Tensor owner = new(new TensorShape(1, 4, 1), DType.F32);
        device.Produce(owner, [9, 9, 9, 9]);
        using Tensor view = owner.Reshape(new TensorShape(1, 4));
        device.Produce(view, [.. Values]);

        Assert.Same(device.CopyToDevice(view), device.CopyToDevice(view));
        Assert.Equal(Values, view.AsReadOnlySpan<float>().ToArray());
    }

    [Fact]
    public void RewritingTheOwner_MakesItCurrentAgain()
    {
        using ArrayResidency device = new();
        using Tensor owner = new(new TensorShape(2, 2), DType.F32);
        using (Tensor view = owner.Reshape(new TensorShape(4)))
            device.Produce(view, [9, 9, 9, 9]);
        device.Produce(owner, [.. Values]);

        Assert.Equal(Values, owner.AsReadOnlySpan<float>().ToArray());
    }

    [Fact]
    public void ReadingAViewOnlyRead_ByABackend_IsAllowed()
    {
        using ArrayResidency device = new();
        using Tensor owner = new(new TensorShape(2, 2), DType.F32);
        Values.CopyTo(owner.AsSpan<float>());
        using Tensor view = owner.Reshape(new TensorShape(4));

        Assert.Equal(Values, device.CopyToDevice(view));
        Assert.Equal(Values, owner.AsReadOnlySpan<float>().ToArray());
    }
}
