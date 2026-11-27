using HartsyInference.World.Camera;
using Xunit;
using static HartsyInference.World.Tests.Camera.CameraTestData;

namespace HartsyInference.World.Tests.Camera;

public sealed class CoverCropMappingTests
{
    [Fact]
    public void WideSource_1920x1080_To_1280x704_MatchesHandComputedCrop()
    {
        // Scale 2/3 (width-limited): 1920x1080 -> 1280x720, then crop 8 rows top and bottom.
        CoverCropMapping map = CoverCropMapping.Create(1920, 1080, 1280, 704);
        Assert.Equal(1280, map.ResizedWidth);
        Assert.Equal(720, map.ResizedHeight);
        Assert.Equal(0, map.Left);
        Assert.Equal(8, map.Top);

        (float fx, float fy, float cx, float cy) = map.MapIntrinsics(1000f, 1000f, 960f, 540f);
        AssertClose(666.6667f, fx, 0.01f);
        AssertClose(666.6667f, fy, 0.01f);
        AssertClose(640f, cx, 0.01f);
        AssertClose(352f, cy, 0.01f);
    }

    [Fact]
    public void TallSource_1000x500_To_1280x704_CropsHorizontally()
    {
        // Scale 1.408 (height-limited): 1000x500 -> 1408x704, then crop 64 columns on each side.
        CoverCropMapping map = CoverCropMapping.Create(1000, 500, 1280, 704);
        Assert.Equal(1408, map.ResizedWidth);
        Assert.Equal(704, map.ResizedHeight);
        Assert.Equal(64, map.Left);
        Assert.Equal(0, map.Top);

        (float fx, float fy, float cx, float cy) = map.MapIntrinsics(800f, 800f, 500f, 250f);
        AssertClose(1126.4f, fx, 0.01f);
        AssertClose(1126.4f, fy, 0.01f);
        AssertClose(640f, cx, 0.01f);
        AssertClose(352f, cy, 0.01f);
    }

    [Fact]
    public void SourceCenter_MapsToCropCenter()
    {
        CoverCropMapping map = CoverCropMapping.Create(1920, 1080, 1280, 704);
        (float x, float y) = map.MapPoint(960f, 540f);
        AssertClose(640f, x, 0.01f);
        AssertClose(352f, y, 0.01f);
    }

    [Fact]
    public void Rounding_IsHalfToEvenLikePythonRound()
    {
        // Width-limited scale 0.5: the height 5 * 0.5 = 2.5 rounds to 2, and 7 * 0.5 = 3.5 rounds to 4.
        Assert.Equal(2, CoverCropMapping.Create(2, 5, 1, 1).ResizedHeight);
        Assert.Equal(4, CoverCropMapping.Create(2, 7, 1, 1).ResizedHeight);
    }

    [Fact]
    public void OddOverhang_FloorsTheCropOffset_LikeUpstreamIntegerDivision()
    {
        // Upstream crops at (resized - target) // 2, so an odd overhang puts the extra pixel on the right or bottom.
        // 1700x930 -> 1280x704: scale 704/930 gives width 1286.9, which rounds to 1287, so 7 columns overhang and Left is 3, not 4.
        CoverCropMapping wide = CoverCropMapping.Create(1700, 930, 1280, 704);
        Assert.Equal(1287, wide.ResizedWidth);
        Assert.Equal(704, wide.ResizedHeight);
        Assert.Equal(3, wide.Left);
        Assert.Equal(0, wide.Top);

        // 1280x707 -> 1280x704: no resize, 3 rows overhang, Top is 1, not 2.
        CoverCropMapping tall = CoverCropMapping.Create(1280, 707, 1280, 704);
        Assert.Equal(1280, tall.ResizedWidth);
        Assert.Equal(707, tall.ResizedHeight);
        Assert.Equal(0, tall.Left);
        Assert.Equal(1, tall.Top);
    }

    [Fact]
    public void RandomGeometry_PointsAndIntrinsics_RoundTrip()
    {
        Random rng = new(5);
        for (int trial = 0; trial < 200; trial++)
        {
            int sourceWidth = rng.Next(64, 4096), sourceHeight = rng.Next(64, 4096);
            int targetWidth = rng.Next(16, 1500), targetHeight = rng.Next(16, 1500);
            CoverCropMapping map = CoverCropMapping.Create(sourceWidth, sourceHeight, targetWidth, targetHeight);

            float x = Uniform(rng, 0f, sourceWidth), y = Uniform(rng, 0f, sourceHeight);
            (float mx, float my) = map.MapPoint(x, y);
            (float ux, float uy) = map.UnmapPoint(mx, my);
            AssertClose(x, ux, 1e-2f);
            AssertClose(y, uy, 1e-2f);

            float fx = Uniform(rng, 100f, 2000f), fy = Uniform(rng, 100f, 2000f);
            float cx = Uniform(rng, 0f, sourceWidth), cy = Uniform(rng, 0f, sourceHeight);
            (float fxC, float fyC, float cxC, float cyC) = map.MapIntrinsics(fx, fy, cx, cy);
            (float fxU, float fyU, float cxU, float cyU) = map.UnmapIntrinsics(fxC, fyC, cxC, cyC);
            AssertClose(fx, fxU, 1e-2f);
            AssertClose(fy, fyU, 1e-2f);
            AssertClose(cx, cxU, 1e-2f);
            AssertClose(cy, cyU, 1e-2f);
        }
    }

    [Fact]
    public void Create_RejectsNonPositiveSizes()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => CoverCropMapping.Create(0, 10, 4, 4));
        Assert.Throws<ArgumentOutOfRangeException>(() => CoverCropMapping.Create(10, 10, 4, -1));
    }
}
