using HartsyInference.Engine.Features;
using HartsyInference.Engine.Requests;
using Xunit;

namespace HartsyInference.Engine.Tests.Features;

/// <summary>The LongCat init-image resizes against values the reference's own functions produce. The avatar reference prepares its init image with
/// diffusers' resize_mode "crop" (cover-scale, Lanczos, centre crop); the base I2V reference with the default resize (a Lanczos stretch). The
/// engine's Lanczos is not bit-identical to PIL's, so the comparison allows one level per byte.</summary>
[Trait("Category", "SyntheticSmoke")]
public sealed class LongCatInitImageResizeTests
{
    private const int Tolerance = 1;

    [Fact]
    public void AvatarInitImageCoversAndCentreCropsLikeTheReference()
    {
        Assert.True(MaxDiff(FeatureImaging.ResizeCropRgb24Lanczos(Source(5, 3), 4, 4), Crop5x3To4x4) <= Tolerance);
        Assert.True(MaxDiff(FeatureImaging.ResizeCropRgb24Lanczos(Source(7, 9), 10, 6), Crop7x9To10x6) <= Tolerance);
    }

    [Fact]
    public void BaseInitImageStretchesLikeTheReference()
    {
        Assert.True(MaxDiff(FeatureImaging.ResizeRgb24Lanczos(Source(5, 3), 4, 4), Stretch5x3To4x4) <= Tolerance);
        Assert.True(MaxDiff(FeatureImaging.ResizeRgb24Lanczos(Source(7, 9), 10, 6), Stretch7x9To10x6) <= Tolerance);
    }

    // Deterministic ramps (no wrap at these sizes): pixel (x, y, c) = 10x + 12y + 40c.
    private static ImageData Source(int width, int height)
    {
        byte[] rgb = new byte[width * height * 3];
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
                for (int c = 0; c < 3; c++)
                    rgb[(y * width + x) * 3 + c] = (byte)(10 * x + 12 * y + 40 * c);
        return new ImageData { Rgb = rgb, Width = width, Height = height };
    }

    private static int MaxDiff(byte[] actual, byte[] expected)
    {
        Assert.Equal(expected.Length, actual.Length);
        int max = 0;
        for (int i = 0; i < expected.Length; i++)
            max = Math.Max(max, Math.Abs(actual[i] - expected[i]));
        return max;
    }

    // Source 5x3 to 4x4: the reference's values (diffusers 0.35.1 VaeImageProcessor, PIL Lanczos).
    private static readonly byte[] Crop5x3To4x4 = [
            6, 46, 86, 15, 55, 95, 23, 63, 103, 32, 72, 112,
            12, 52, 92, 21, 61, 101, 29, 69, 109, 38, 78, 118,
            26, 66, 106, 35, 75, 115, 43, 83, 123, 52, 92, 132,
            32, 72, 112, 41, 81, 121, 49, 89, 129, 58, 98, 138
    ];
    private static readonly byte[] Stretch5x3To4x4 = [
            0, 40, 80, 13, 53, 93, 25, 65, 105, 38, 78, 118,
            6, 46, 86, 19, 59, 99, 31, 71, 111, 44, 84, 124,
            20, 60, 100, 33, 73, 113, 45, 85, 125, 58, 98, 138,
            26, 66, 106, 39, 79, 119, 51, 91, 131, 64, 104, 144
    ];

    // Source 7x9 to 10x6: the reference's values (diffusers 0.35.1 VaeImageProcessor, PIL Lanczos).
    private static readonly byte[] Crop7x9To10x6 = [
            24, 64, 104, 29, 69, 109, 38, 78, 118, 45, 85, 125,
            52, 92, 132, 58, 98, 138, 65, 105, 145, 72, 112, 152,
            81, 121, 161, 86, 126, 166, 34, 74, 114, 39, 79, 119,
            48, 88, 128, 55, 95, 135, 62, 102, 142, 68, 108, 148,
            75, 115, 155, 82, 122, 162, 91, 131, 171, 96, 136, 176,
            43, 83, 123, 48, 88, 128, 57, 97, 137, 64, 104, 144,
            71, 111, 151, 77, 117, 157, 84, 124, 164, 91, 131, 171,
            100, 140, 180, 105, 145, 185, 51, 91, 131, 56, 96, 136,
            65, 105, 145, 72, 112, 152, 79, 119, 159, 85, 125, 165,
            92, 132, 172, 99, 139, 179, 108, 148, 188, 113, 153, 193,
            60, 100, 140, 65, 105, 145, 74, 114, 154, 81, 121, 161,
            88, 128, 168, 94, 134, 174, 101, 141, 181, 108, 148, 188,
            117, 157, 197, 122, 162, 202, 70, 110, 150, 75, 115, 155,
            84, 124, 164, 91, 131, 171, 98, 138, 178, 104, 144, 184,
            111, 151, 191, 118, 158, 198, 127, 167, 207, 132, 172, 212
    ];
    private static readonly byte[] Stretch7x9To10x6 = [
            2, 42, 82, 7, 47, 87, 16, 56, 96, 23, 63, 103,
            30, 70, 110, 36, 76, 116, 43, 83, 123, 50, 90, 130,
            59, 99, 139, 64, 104, 144, 20, 60, 100, 25, 65, 105,
            34, 74, 114, 41, 81, 121, 48, 88, 128, 54, 94, 134,
            61, 101, 141, 68, 108, 148, 77, 117, 157, 82, 122, 162,
            38, 78, 118, 43, 83, 123, 52, 92, 132, 59, 99, 139,
            66, 106, 146, 72, 112, 152, 79, 119, 159, 86, 126, 166,
            95, 135, 175, 100, 140, 180, 56, 96, 136, 61, 101, 141,
            70, 110, 150, 77, 117, 157, 84, 124, 164, 90, 130, 170,
            97, 137, 177, 104, 144, 184, 113, 153, 193, 118, 158, 198,
            74, 114, 154, 79, 119, 159, 88, 128, 168, 95, 135, 175,
            102, 142, 182, 108, 148, 188, 115, 155, 195, 122, 162, 202,
            131, 171, 211, 136, 176, 216, 92, 132, 172, 97, 137, 177,
            106, 146, 186, 113, 153, 193, 120, 160, 200, 126, 166, 206,
            133, 173, 213, 140, 180, 220, 149, 189, 229, 154, 194, 234
    ];

}
