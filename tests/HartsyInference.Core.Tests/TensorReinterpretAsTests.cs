using System.Runtime.CompilerServices;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Core.Tests;

/// <summary>Guards <see cref="Tensor.ReinterpretAs"/>, the relabel the resident-nvfp4 path is built on: a U8
/// <c>[N, K/2]</c> checkpoint weight becomes an <see cref="DType.F4E2M1"/> <c>[N, K]</c> view over the same bytes.
/// A byte-count check that let a mismatch through would hand every consumer a weight whose declared inner dimension
/// is wrong, and a view that failed to root its parent would dangle the moment the parent was collected.</summary>
public sealed unsafe class TensorReinterpretAsTests
{
    [Fact]
    public void PackedNvfp4Relabel_KeepsBytesAndReportsTheLogicalShape()
    {
        using Tensor packed = new Tensor(new TensorShape(6, 8), DType.U8);
        byte* source = (byte*)packed.DataPointer;
        for (int i = 0; i < 48; i++) source[i] = (byte)(i * 7 + 1);

        using Tensor view = packed.ReinterpretAs(DType.F4E2M1, new TensorShape(6, 16));

        Assert.Equal(DType.F4E2M1, view.DType);
        Assert.Equal(6L, view.Shape[0]);
        Assert.Equal(16L, view.Shape[1]);
        Assert.True(packed.DataPointer == view.DataPointer, "ReinterpretAs copied instead of viewing.");
        Assert.Equal(48L, view.DType.ComputeByteCount(view.ElementCount));
        byte* seen = (byte*)view.DataPointer;
        for (int i = 0; i < 48; i++) Assert.Equal((byte)(i * 7 + 1), seen[i]);
    }

    [Theory]
    [InlineData(6, 14)]   // 42 bytes of F4E2M1 against 48 bytes of U8
    [InlineData(6, 32)]   // the caller forgot the /2 and asked for twice the elements
    [InlineData(3, 16)]   // right element count per row, wrong row count
    public void ByteCountMismatch_Throws(long rows, long columns)
    {
        using Tensor packed = new Tensor(new TensorShape(6, 8), DType.U8);
        HartsyInferenceException error = Assert.Throws<HartsyInferenceException>(
            () => packed.ReinterpretAs(DType.F4E2M1, new TensorShape(rows, columns)));
        Assert.Contains("Cannot reinterpret", error.Message);
    }

    [Fact]
    public void View_RootsItsParent_AgainstCollection()
    {
        (Tensor view, WeakReference parent) = MakeOrphanedView();

        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();

        // A reachable object is never collected, so this direction of the assertion cannot flake: it fails only if
        // the view stopped rooting the parent, at which point the view's own bytes are freed memory.
        Assert.True(parent.IsAlive, "ReinterpretAs view did not root its parent tensor.");
        Assert.Same(parent.Target, view.KeepAliveOwner);
        byte* seen = (byte*)view.DataPointer;
        for (int i = 0; i < 48; i++) Assert.Equal((byte)(i * 7 + 1), seen[i]);
        view.Dispose();
    }

    /// <summary>Builds the view in a frame of its own so the parent's only remaining reference is the view's.</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static (Tensor View, WeakReference Parent) MakeOrphanedView()
    {
        Tensor packed = new Tensor(new TensorShape(6, 8), DType.U8);
        byte* source = (byte*)packed.DataPointer;
        for (int i = 0; i < 48; i++) source[i] = (byte)(i * 7 + 1);
        return (packed.ReinterpretAs(DType.F4E2M1, new TensorShape(6, 16)), new WeakReference(packed));
    }
}
