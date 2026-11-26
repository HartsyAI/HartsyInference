using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Fibo 2.0 RoPE position-id planes: gist <c>(i, 0, 0)</c>, image on frame <c>MaxGist + 1</c> row-major,
/// reference <c>k</c> on frame <c>MaxGist + 2 + k</c>, and the stream order gist-then-image.</summary>
public sealed unsafe class Fibo2PositionIdsTests
{
    [Fact]
    public void Gist_IsOneBasedOnTheFrameAxis()
    {
        using Tensor ids = Fibo2PositionIds.Gist(Fibo2Config.Checkpoint, 3);
        AssertRow(ids, 0, 1, 0, 0);
        AssertRow(ids, 2, 3, 0, 0);
    }

    [Fact]
    public void Image_UsesFrameMaxGistPlusOneAndRowMajorGrid()
    {
        using Tensor ids = Fibo2PositionIds.Image(Fibo2Config.Checkpoint, 2, 3);
        Assert.Equal(6, ids.Shape[0]);
        AssertRow(ids, 0, 513, 0, 0);
        AssertRow(ids, 2, 513, 0, 2);
        AssertRow(ids, 3, 513, 1, 0);
        AssertRow(ids, 5, 513, 1, 2);
    }

    [Fact]
    public void Reference_UsesFrameAfterTheImagePlane()
    {
        using Tensor first = Fibo2PositionIds.Reference(Fibo2Config.Checkpoint, 0, 1, 2);
        using Tensor second = Fibo2PositionIds.Reference(Fibo2Config.Checkpoint, 1, 1, 2);
        AssertRow(first, 1, 514, 0, 1);
        AssertRow(second, 0, 515, 0, 0);
    }

    [Fact]
    public void Combined_PlacesGistBeforeImage()
    {
        using Tensor ids = Fibo2PositionIds.Combined(Fibo2Config.Checkpoint, 2, 1, 2);
        Assert.Equal(4, ids.Shape[0]);
        AssertRow(ids, 0, 1, 0, 0);
        AssertRow(ids, 1, 2, 0, 0);
        AssertRow(ids, 2, 513, 0, 0);
        AssertRow(ids, 3, 513, 0, 1);
    }

    [Fact]
    public void Gist_RejectsCountsOutsideTheSchedule()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => Fibo2PositionIds.Gist(Fibo2Config.Checkpoint, 513));
        Assert.Throws<ArgumentOutOfRangeException>(() => Fibo2PositionIds.Gist(Fibo2Config.Checkpoint, 0));
    }

    private static void AssertRow(Tensor ids, int row, float frame, float y, float x)
    {
        float* p = (float*)ids.DataPointer;
        Assert.Equal(frame, p[row * 3]);
        Assert.Equal(y, p[row * 3 + 1]);
        Assert.Equal(x, p[row * 3 + 2]);
    }
}
