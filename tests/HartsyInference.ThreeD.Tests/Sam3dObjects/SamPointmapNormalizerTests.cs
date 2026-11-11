using HartsyInference.ThreeD.Models.Sam3dObjects;
using Xunit;

namespace HartsyInference.ThreeD.Tests.Sam3dObjects;

public sealed class SamPointmapNormalizerTests
{
    private const float NaN = float.NaN;

    // 3 x 2 x 2 channel-first pointmap: x = 0,1,2,3, y = 10 constant, z = 5,6,7,8.
    private static float[] Pointmap() => [0f, 1f, 2f, 3f, 10f, 10f, 10f, 10f, 5f, 6f, 7f, 8f];

    [Fact]
    public void ComputeScaleAndShift_ShiftIsTheLowerMedianOfTheMaskedPointsPerAxis()
    {
        (float[] scale, float[] shift) = SamPointmapNormalizer.ComputeScaleAndShift(Pointmap(), [0f, 1f, 1f, 1f], 2, 2);

        // Masked x = 1, 2, 3 -> 2; y -> 10; z = 6, 7, 8 -> 7.
        Assert.Equal([2f, 10f, 7f], shift);
    }

    [Fact]
    public void ComputeScaleAndShift_ScaleIsTheMedianOverAllPixelsOfTheLargestCentredAxis()
    {
        (float[] scale, _) = SamPointmapNormalizer.ComputeScaleAndShift(Pointmap(), [0f, 1f, 1f, 1f], 2, 2);

        // Centred maxima per pixel: |0−2|,|5−7| -> 2; 1; 0; 1 (the scene, mask-independent). Sorted 0,1,1,2: lower median 1.
        Assert.Equal([1f, 1f, 1f], scale);
    }

    [Fact]
    public void ComputeScaleAndShift_AppliesTheScaleFactor()
    {
        (float[] scale, _) = SamPointmapNormalizer.ComputeScaleAndShift(Pointmap(), [0f, 1f, 1f, 1f], 2, 2, scaleFactor: 3f);

        Assert.Equal([3f, 3f, 3f], scale);
    }

    [Fact]
    public void ComputeScaleAndShift_NaNPixelsAreIgnoredAndEmptyMasksFallBack()
    {
        float[] pointmap = Pointmap();
        pointmap[3] = NaN;

        (float[] scale, float[] shift) = SamPointmapNormalizer.ComputeScaleAndShift(pointmap, [1f, 1f, 1f, 1f], 2, 2);
        (float[] fallbackScale, float[] fallbackShift) = SamPointmapNormalizer.ComputeScaleAndShift(pointmap, [0f, 0f, 0f, 0f], 2, 2);

        Assert.Equal(1f, shift[0]);
        Assert.Equal(3, scale.Length);
        Assert.Equal([1f, 1f, 1f], fallbackScale);
        Assert.Equal([0f, 0f, 0f], fallbackShift);
    }

    [Fact]
    public void NormalizeThenDenormalize_RoundTripsAndKeepsNaN()
    {
        float[] pointmap = Pointmap();
        pointmap[1] = NaN;
        float[] work = (float[])pointmap.Clone();

        SamPointmapNormalizer.Normalize(work, 2, 2, [2f, 4f, 0.5f], [1f, 2f, 3f]);
        Assert.Equal(-0.5f, work[0]);
        Assert.True(float.IsNaN(work[1]));
        SamPointmapNormalizer.Denormalize(work, 2, 2, [2f, 4f, 0.5f], [1f, 2f, 3f]);

        for (int i = 0; i < pointmap.Length; i++)
        {
            if (!float.IsNaN(pointmap[i]))
                Assert.Equal(pointmap[i], work[i], 4);
        }
    }

    [Fact]
    public void Normalize_RejectsZeroScaleAndWrongSize()
    {
        Assert.Throws<ArgumentException>(() => SamPointmapNormalizer.Normalize(Pointmap(), 2, 2, [0f, 1f, 1f], [0f, 0f, 0f]));
        Assert.Throws<ArgumentException>(() => SamPointmapNormalizer.Normalize(new float[5], 2, 2, [1f, 1f, 1f], [0f, 0f, 0f]));
    }

    [Fact]
    public void ResizeNearest_UsesFloorOfTheScaledIndex()
    {
        float[] up = SamPointmapNormalizer.ResizeNearest([1f, 2f, 3f, 4f], 2, 2, 4, 4);
        float[] down = SamPointmapNormalizer.ResizeNearest([1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f], 3, 3, 2, 2);

        Assert.Equal([1f, 1f, 2f, 2f, 1f, 1f, 2f, 2f, 3f, 3f, 4f, 4f, 3f, 3f, 4f, 4f], up);
        Assert.Equal([1f, 2f, 4f, 5f], down);
    }
}
