using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.PyTorch;
using Xunit;

namespace HartsyInference.ModelAssets.Tests.PyTorch;

public sealed class PickleComplexStorageTests
{
    [Fact]
    public void ViewAsReal_DoublesOffsetAndStridesAndAppendsPairAxis()
    {
        PickleStorage storage = new(DType.F32, "k", 8, IsComplex: true);

        PickleTensor view = PickleMachine.ViewAsReal(storage, 1, [2, 2], [2, 1]);

        Assert.Equal(2L, view.Offset);
        Assert.Equal(new long[] { 2, 2, 2 }, view.Size);
        Assert.Equal(new long[] { 4, 2, 1 }, view.Stride);
        Assert.False(view.Storage.IsComplex);
        Assert.Equal(DType.F32, view.Storage.DType);
    }
}
