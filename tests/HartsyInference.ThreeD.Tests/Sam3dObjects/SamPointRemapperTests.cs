using HartsyInference.ThreeD.Models.Sam3dObjects;
using Xunit;

namespace HartsyInference.ThreeD.Tests.Sam3dObjects;

public sealed class SamPointRemapperTests
{
    [Fact]
    public void Linear_IsTheIdentity()
    {
        float[] points = [1f, -2f, 3f];

        SamPointRemapper.Apply(points, SamPointRemapType.Linear);

        Assert.Equal([1f, -2f, 3f], points);
    }

    [Fact]
    public void Exp_DividesXyByOnePlusZAndTakesLog1pOfZ()
    {
        float[] points = [2f, 4f, 1f];

        SamPointRemapper.Apply(points, SamPointRemapType.Exp);

        Assert.Equal(1f, points[0], 5);
        Assert.Equal(2f, points[1], 5);
        Assert.Equal(MathF.Log(2f), points[2], 5);
    }

    [Theory]
    [InlineData(SamPointRemapType.Sinh)]
    [InlineData(SamPointRemapType.Exp)]
    [InlineData(SamPointRemapType.SinhExp)]
    [InlineData(SamPointRemapType.ExpDisparity)]
    public void Invert_RoundTrips(SamPointRemapType type)
    {
        float[] original = [0.3f, -0.7f, 1.6f, 1.2f, 0.4f, 0.9f];
        float[] points = (float[])original.Clone();

        SamPointRemapper.Apply(points, type);
        SamPointRemapper.Invert(points, type);

        for (int i = 0; i < original.Length; i++)
            Assert.Equal(original[i], points[i], 4);
    }

    [Fact]
    public void ExpDisparity_UsesDivisionAndLogByZ()
    {
        float[] points = [2f, 6f, 2f];

        SamPointRemapper.Apply(points, SamPointRemapType.ExpDisparity);

        Assert.Equal([1f, 3f], points.Take(2));
        Assert.Equal(MathF.Log(2f), points[2], 5);
    }

    [Fact]
    public void SinhExp_ClampsZBeforeTheLog()
    {
        float[] points = [0f, 0f, 0f];

        SamPointRemapper.Apply(points, SamPointRemapType.SinhExp);

        Assert.Equal(MathF.Log(1e-8f), points[2], 3);
    }

    [Fact]
    public void ParseAndShapeErrors()
    {
        Assert.Equal(SamPointRemapType.SinhExp, SamPointRemapper.Parse("sinh_exp"));
        Assert.Throws<ArgumentException>(() => SamPointRemapper.Parse("cubic"));
        Assert.Throws<ArgumentException>(() => SamPointRemapper.Apply(new float[4], SamPointRemapType.Linear));
    }
}
