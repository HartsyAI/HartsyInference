using HartsyInference.Vision.DepthPro2;

namespace HartsyInference.Vision.Tests.DepthPro2;

public sealed class DepthPro2FocalEstimatorTests
{
    [Fact]
    public void AvgPoolCeil_AveragesOnlyTheCellsAWindowCoversAtTheEdge()
    {
        // 1 channel, 3 x 5: the last row and column of windows are cut by the edge, as avg_pool2d(2, ceil_mode=True) cuts them.
        float[] src = Enumerable.Range(0, 15).Select(i => (float)i).ToArray();
        float[] dst = new float[2 * 3];
        DepthPro2FovHead.AvgPoolCeil(src, 1, 3, 5, 2, dst);
        Assert.Equal([(0 + 1 + 5 + 6) / 4f, (2 + 3 + 7 + 8) / 4f, (4 + 9) / 2f, (10 + 11) / 2f, (12 + 13) / 2f, 14f], dst);
    }

    [Fact]
    public void SinCos2d_ScalesBothAxesByTheLongerSide()
    {
        const int gh = 2, gw = 3, dim = 16;
        float[] code = DepthPro2FovHead.SinCos2d(gh, gw, dim);
        Assert.Equal(gh * gw * dim, code.Length);
        int n = dim / 4;
        for (int y = 0; y < gh; y++)
            for (int x = 0; x < gw; x++)
                for (int i = 0; i < n; i++)
                {
                    double f = Math.PI * Math.Pow(2, 7.0 * i / (n - 1));
                    double cy = (y + 0.5 - gh / 2.0) / gw, cx = (x + 0.5 - gw / 2.0) / gw;
                    int row = (y * gw + x) * dim;
                    Assert.Equal(Math.Sin(cy * f), code[row + i], 4);
                    Assert.Equal(Math.Cos(cy * f), code[row + n + i], 4);
                    Assert.Equal(Math.Sin(cx * f), code[row + 2 * n + i], 4);
                    Assert.Equal(Math.Cos(cx * f), code[row + 3 * n + i], 4);
                }
    }

    [Theory]
    [InlineData(256, 384, 256, 384)]
    [InlineData(250, 370, 224, 320)]
    [InlineData(1080, 1920, 512, 896)]
    [InlineData(20, 40, 32, 64)]
    public void InputSize_FitsTheShortSideTo512AndBothSidesToMultiplesOf32(int h, int w, int expectedH, int expectedW) =>
        Assert.Equal((expectedH, expectedW), DepthPro2FocalEstimator.InputSize(h, w));

    [Fact]
    public void CenterCrop_CutsTheLongSideTo16By9()
    {
        Assert.Equal((256, 384, 0, 0), DepthPro2FocalEstimator.CenterCrop(256, 384));
        Assert.Equal((256, 448, 0, 32), DepthPro2FocalEstimator.CenterCrop(256, 512));
        Assert.Equal((448, 256, 32, 0), DepthPro2FocalEstimator.CenterCrop(512, 256));
    }

    [Fact]
    public void CropAnglesToFull_KeepsTheFocalAndWidensTheCroppedAxis()
    {
        DepthPro2FieldOfView crop = new(60, 40);
        DepthPro2FieldOfView same = DepthPro2FocalEstimator.CropAnglesToFull(crop, 100, 200, 100, 200);
        Assert.Equal(60, same.HorizontalDegrees, 4);
        Assert.Equal(40, same.VerticalDegrees, 4);

        DepthPro2FieldOfView wide = DepthPro2FocalEstimator.CropAnglesToFull(crop, 100, 200, 100, 300);
        double fx = 100 / Math.Tan(Math.PI / 6);
        Assert.Equal(2 * Math.Atan(150 / fx) * 180 / Math.PI, wide.HorizontalDegrees, 4);
        Assert.Equal(40, wide.VerticalDegrees, 4);
    }

    [Fact]
    public void FocalLengthPx_TakesTheLowerMedianOfTheShorterSideAngle()
    {
        DepthPro2FieldOfView[] frames = [new(50, 30), new(70, 90), new(60, 60), new(80, 40)];
        // Landscape: the vertical angles 30, 40, 60, 90 have the lower median 40.
        Assert.Equal(0.5 * 300 / Math.Tan(20 * Math.PI / 180), DepthPro2FocalEstimator.FocalLengthPx(frames, 300, 400), 6);
        // Portrait: the horizontal angles 50, 60, 70, 80 have the lower median 60.
        Assert.Equal(0.5 * 300 / Math.Tan(30 * Math.PI / 180), DepthPro2FocalEstimator.FocalLengthPx(frames, 400, 300), 6);
    }
}
