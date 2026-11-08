using HartsyInference.Core.Tensors;
using HartsyInference.Vision.Segmentation.Sam3;
using Xunit;

namespace HartsyInference.Vision.Tests.Segmentation.Sam3;

public sealed class Sam3PositionEncodingTests
{
    [Fact]
    public void ShapeIsDModelByGrid()
    {
        using Tensor encoding = Sam3PositionEncoding.Create(256, 3, 5);

        Assert.Equal(new TensorShape(1, 256, 3, 5), encoding.Shape);
    }

    [Fact]
    public void FirstRowMatchesClosedForm()
    {
        const int dModel = 256;
        const int height = 4;
        const int width = 6;
        const int half = dModel / 2;
        using Tensor encoding = Sam3PositionEncoding.Create(dModel, height, width);
        ReadOnlySpan<float> values = encoding.AsReadOnlySpan<float>();

        double yAngle = 2.0 * Math.PI / (height + 1e-6) * 1.0;
        double xAngle0 = 2.0 * Math.PI / (width + 1e-6) * 1.0;
        double xAngle2 = 2.0 * Math.PI / (width + 1e-6) * 3.0;
        double divisorChannel2 = Math.Pow(10000.0, 2.0 / half);

        // Channel 0 of the y block is sine at position 1 with divisor 1, so it equals sin(2π / (H + eps)).
        Assert.Equal(Math.Sin(yAngle), values[0 * height * width], 4);
        // Channel 1 is cosine of the same angle.
        Assert.Equal(Math.Cos(yAngle), values[1 * height * width], 4);
        // Channel 2 uses divisor 10000^(2/half) and is sine.
        Assert.Equal(Math.Sin(yAngle / divisorChannel2), values[2 * height * width], 4);
        // X block channel 0 at column 0 and column 2 of the first row.
        int xChannel0 = half * height * width;
        Assert.Equal(Math.Sin(xAngle0), values[xChannel0], 4);
        Assert.Equal(Math.Sin(xAngle2), values[xChannel0 + 2], 4);
    }

    [Fact]
    public void RejectsWidthNotMultipleOfFour()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => Sam3PositionEncoding.Create(6, 4, 4));
    }
}
