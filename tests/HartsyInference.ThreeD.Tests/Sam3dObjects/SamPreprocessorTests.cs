using HartsyInference.ThreeD.Models.Sam3dObjects;
using Xunit;

namespace HartsyInference.ThreeD.Tests.Sam3dObjects;

public sealed class SamPreprocessorTests
{
    [Fact]
    public void BilinearAntialias_ShrinkingWidensTheTriangleFilterAndEnlargingIsPlainBilinear()
    {
        // 4 -> 2: output 0 has centre 1.0 and support 2, taps 0..2 with weights .75 .75 .25 (PIL's precompute_coeffs); the ramp is symmetric.
        float[] shrunk = SamImageResize.BilinearAntialias([0f, 1f, 2f, 3f], 1, 1, 4, 1, 2);
        Assert.Equal(1.25 / 1.75, shrunk[0], 5);
        Assert.Equal(3 - 1.25 / 1.75, shrunk[1], 5);

        // 2 -> 4 with half-pixel centres: 0, .25, .75, 1 (the end cells clamp).
        float[] grown = SamImageResize.BilinearAntialias([0f, 1f], 1, 1, 2, 1, 4);
        Assert.Equal([0f, 0.25f, 0.75f, 1f], grown);

        float[] same = SamImageResize.BilinearAntialias([1f, 5f, 2f, 7f], 1, 2, 2, 2, 2);
        Assert.Equal([1f, 5f, 2f, 7f], same);
    }

    [Fact]
    public void BilinearAntialias_KeepsAConstantImageConstantAndResizesEveryChannel()
    {
        float[] image = new float[3 * 9 * 5];
        for (int c = 0; c < 3; c++)
            Array.Fill(image, c + 0.5f, c * 45, 45);
        float[] result = SamImageResize.BilinearAntialias(image, 3, 9, 5, 4, 7);
        Assert.Equal(3 * 4 * 7, result.Length);
        for (int c = 0; c < 3; c++)
            for (int i = 0; i < 28; i++)
                Assert.Equal(c + 0.5f, result[c * 28 + i], 4);
    }

    [Fact]
    public void Nearest_UsesTheFloorRule()
    {
        float[] result = SamImageResize.Nearest([0f, 1f, 2f, 3f, 4f], 1, 1, 5, 1, 3);
        Assert.Equal([0f, 1f, 3f], result);
    }

    private static float[] Rectangle(int h, int w, int x1, int y1, int x2, int y2)
    {
        float[] mask = new float[h * w];
        for (int y = y1; y < y2; y++)
            for (int x = x1; x < x2; x++)
                mask[y * w + x] = 1f;
        return mask;
    }

    [Fact]
    public void Prepare_BuildsBothBranchesAndNormalisesThePointmapWithTheFullMask()
    {
        const int h = 20, w = 32, size = 8;
        float[] rgb = new float[3 * h * w];
        for (int i = 0; i < rgb.Length; i++) rgb[i] = (i % 7) / 7f;
        float[] mask = Rectangle(h, w, 10, 6, 20, 14);
        float[] points = new float[3 * h * w];
        for (int i = 0; i < h * w; i++)
        {
            points[i] = (i % w) * 0.1f;
            points[h * w + i] = (i / w) * 0.1f;
            points[2 * h * w + i] = 2f + (i % 5) * 0.05f;
        }

        points[3] = float.NaN;
        SamPrepared prepared = SamPreprocessor.Prepare(rgb, mask, new SamPointmapInput(points, h, w), h, w, size);

        // Shapes: images 3 x 8 x 8, masks 8 x 8, pointmaps 3 x 8 x 8; the SLAT inputs share the image and mask arrays and carry no pointmap.
        SamConditionInputs s = prepared.Structure;
        Assert.Equal(3 * size * size, s.CroppedImage.Length);
        Assert.Equal(size * size, s.CroppedMask.Length);
        Assert.Equal(3 * size * size, s.FullImage.Length);
        Assert.Equal(size * size, s.FullMask.Length);
        Assert.Equal(3 * size * size, s.CroppedPointmap!.Planar.Length);
        Assert.Equal(size, s.FullPointmap!.Height);
        Assert.Same(s.CroppedImage, prepared.Slat.CroppedImage);
        Assert.Null(prepared.Slat.CroppedPointmap);

        // The scene scale is one number on all axes; the shift is the lower median of the masked points (x runs 1.0 to 1.9 over the 10 masked columns, 8 rows each: the
        // 40th of 80 sorted values is 1.4), and a point is normalised as (p - shift) / scale.
        Assert.Equal(prepared.PointmapScale![0], prepared.PointmapScale[1]);
        Assert.Equal(prepared.PointmapScale[1], prepared.PointmapScale[2]);
        Assert.True(prepared.PointmapScale[0] > 0f);
        Assert.Equal(1.4f, prepared.PointmapShift![0], 4);

        // The full branch pads the 20-row image to 32 rows (6 cells above and below): the first and last of the 8 output rows read pad cells, which are NaN in the pointmap and 0 in the mask.
        Assert.True(float.IsNaN(s.FullPointmap.Planar[0]));
        Assert.True(float.IsNaN(s.FullPointmap.Planar[7 * size]));
        Assert.Equal(0f, s.FullMask[0]);
        Assert.Equal(0f, s.FullMask[7 * size]);

        // The crop is the 10 x 8 mask box scaled by 1.2 (side 12), so the object fills the middle of the cropped mask and the border is background.
        Assert.Equal(1f, s.CroppedMask[4 * size + 4]);
        Assert.Equal(0f, s.CroppedMask[0]);
    }

    [Fact]
    public void Prepare_ResizesAPointmapOfAnotherSizeToTheImageFirst()
    {
        const int h = 8, w = 8;
        float[] rgb = new float[3 * h * w];
        float[] mask = Rectangle(h, w, 2, 2, 6, 6);
        float[] coarse = new float[3 * 4 * 4];
        for (int i = 0; i < coarse.Length; i++) coarse[i] = 1f + i * 0.01f;

        SamPrepared prepared = SamPreprocessor.Prepare(rgb, mask, new SamPointmapInput(coarse, 4, 4), h, w, 4);

        Assert.NotNull(prepared.PointmapScale);
        Assert.Equal(3 * 4 * 4, prepared.Structure.FullPointmap!.Planar.Length);
    }

    [Fact]
    public void Prepare_WithoutAPointmapLeavesThePointmapsAndNormalisationOut()
    {
        float[] mask = Rectangle(8, 8, 2, 2, 6, 6);

        SamPrepared prepared = SamPreprocessor.Prepare(new float[3 * 64], mask, null, 8, 8, 4);

        Assert.Null(prepared.Structure.CroppedPointmap);
        Assert.Null(prepared.PointmapScale);
        Assert.Null(prepared.PointmapShift);
    }

    [Fact]
    public void Prepare_RejectsWrongLengthsAndAnEmptyMask()
    {
        Assert.Throws<ArgumentException>(() => SamPreprocessor.Prepare(new float[3 * 63], new float[64], null, 8, 8, 4));
        Assert.Throws<ArgumentException>(() => SamPreprocessor.Prepare(new float[3 * 64], new float[63], null, 8, 8, 4));
        Assert.Throws<InvalidOperationException>(() => SamPreprocessor.Prepare(new float[3 * 64], new float[64], null, 8, 8, 4));
    }
}
