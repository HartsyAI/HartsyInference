using HartsyInference.Core.Tensors;
using HartsyInference.Vision.DepthPro2;

namespace HartsyInference.Vision.Tests.DepthPro2;

public sealed class DepthPro2PreprocessorTests
{
    [Fact]
    public void Normalize_IsPlanarImageNetAtTheNativeSize()
    {
        // 3 x 2 image: pixel (x, y) has R = 10x + y, G = 100, B = 255.
        int width = 3, height = 2;
        byte[] rgb = new byte[width * height * 3];
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
            {
                int p = (y * width + x) * 3;
                rgb[p] = (byte)(10 * x + y);
                rgb[p + 1] = 100;
                rgb[p + 2] = 255;
            }

        using Tensor pixels = DepthPro2Preprocessor.Normalize(rgb, width, height);

        Assert.Equal(new TensorShape(1, 3, height, width), pixels.Shape);
        ReadOnlySpan<float> v = pixels.AsReadOnlySpan<float>();
        int plane = width * height;
        Assert.Equal((21 / 255f - 0.485f) / 0.229f, v[1 * width + 2], 6);       // R at (2, 1)
        Assert.Equal((100 / 255f - 0.456f) / 0.224f, v[plane + 4], 6);
        Assert.Equal((255 / 255f - 0.406f) / 0.225f, v[2 * plane], 6);
    }

    [Fact]
    public void Normalize_RefusesAWrongByteCount() =>
        Assert.Throws<ArgumentException>(() => DepthPro2Preprocessor.Normalize(new byte[17], 3, 2));

    [Fact]
    public void ToMetric_DividesFocalByWidthTimesInverseAndMarksSkyInvalid()
    {
        float[] inverse = [0.5f, 0f, 2f, 1e-3f];
        float[] depth = new float[4], valid = new float[4];

        DepthPro2Preprocessor.ToMetric(inverse, width: 100, focalLengthPx: 80, depth, valid);

        Assert.Equal(80f / (100 * 0.5f), depth[0]);
        Assert.True(float.IsPositiveInfinity(depth[1]));
        Assert.Equal(80f / (100 * 2f), depth[2]);
        Assert.Equal(80f / (100 * 1e-3f), depth[3]);
        Assert.Equal([1f, 0f, 1f, 1f], valid);
    }

    [Theory]
    [InlineData(0.0)]
    [InlineData(-5.0)]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    public void ToMetric_RefusesAFocalLengthThatIsNotAPositiveNumber(double focal) =>
        Assert.Throws<ArgumentOutOfRangeException>(() => DepthPro2Preprocessor.ToMetric(new float[1], 1, focal, new float[1], new float[1]));
}
