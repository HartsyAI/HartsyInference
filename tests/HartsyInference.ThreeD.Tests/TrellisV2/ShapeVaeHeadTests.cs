using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Sparse;
using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Models.TrellisV2;
using Xunit;

namespace HartsyInference.ThreeD.Tests.TrellisV2;

public sealed class ShapeVaeHeadTests
{
    [Fact]
    public void Split_PinsOneHandComputedValuePerHeadFormula()
    {
        float[] values = [1f, 0f, -2f, -1f, 3f, 0.5f, 0f];
        Tensor feats = new(new TensorShape(1, 7), DType.F32);
        values.AsSpan().CopyTo(feats.AsSpan<float>());
        SparseTensor head = new(feats, [0, 2, 3, 4], 16);

        DualGridField field = ShapeVaeHead.Split(head);
        feats.Dispose();

        Assert.Equal([2, 3, 4], field.Voxels.Coords);
        Assert.Equal(0.9621172f, field.Voxels.DualVertices[0], 4);
        Assert.Equal(0.5f, field.Voxels.DualVertices[1], 4);
        Assert.Equal(-0.2615942f, field.Voxels.DualVertices[2], 4);
        Assert.Equal([-1f, 3f, 0.5f], field.IntersectionLogits!);
        Assert.Equal([false, true, true], field.Voxels.Intersected);
        Assert.Equal(MathF.Log(2f), field.SplitWeight[0], 4);
    }

    [Fact]
    public void Split_RejectsAHeadThatIsNotSevenChannels()
    {
        Tensor feats = new(new TensorShape(1, 6), DType.F32);
        SparseTensor head = new(feats, [0, 0, 0, 0], 16);

        Assert.Throws<HartsyInferenceException>(() => ShapeVaeHead.Split(head));
        feats.Dispose();
    }
}
