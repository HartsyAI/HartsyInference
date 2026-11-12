using HartsyInference.ThreeD.Preprocessing;

namespace HartsyInference.ThreeD.Tests.Preprocessing;

public sealed class MeshImageProcessorTests
{
    /// <summary>An image of one colour whose alpha is 255 inside the half-open block and 0 elsewhere.</summary>
    private static (byte[] Rgb, byte[] Alpha) Block(int width, int height, int x0, int y0, int x1, int y1, byte red, byte green, byte blue)
    {
        byte[] rgb = new byte[width * height * 3];
        byte[] alpha = new byte[width * height];
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                rgb[(y * width + x) * 3] = red;
                rgb[(y * width + x) * 3 + 1] = green;
                rgb[(y * width + x) * 3 + 2] = blue;
                alpha[y * width + x] = x >= x0 && x < x1 && y >= y0 && y < y1 ? (byte)255 : (byte)0;
            }
        }
        return (rgb, alpha);
    }

    private static byte[] Pixel(PreparedImage image, int x, int y) => image.Rgb[((y * image.Width + x) * 3)..((y * image.Width + x) * 3 + 3)];

    [Fact]
    public void Hunyuan3D_RecentresTheForegroundWithABorderOnWhite()
    {
        (byte[] rgb, byte[] alpha) = Block(8, 8, 2, 2, 6, 6, 255, 0, 0);

        PreparedImage image = Hunyuan3DImageProcessor.Process(rgb, alpha, 8, 8, size: 8);

        // The upstream box is max - min, so the 4-pixel block measures 3; scale int(8 * 0.85) / 3 = 2 gives a 6 x 6 foreground at offset 1.
        Assert.Equal((8, 8), (image.Width, image.Height));
        for (int y = 0; y < 8; y++)
        {
            for (int x = 0; x < 8; x++)
            {
                bool inside = x is >= 1 and < 7 && y is >= 1 and < 7;
                Assert.Equal(inside ? new byte[] { 255, 0, 0 } : new byte[] { 255, 255, 255 }, Pixel(image, x, y));
            }
        }
    }

    [Fact]
    public void Hunyuan3D_TakesAnImageWithoutAlphaAsAllForegroundAndLosesItsLastRowAndColumnAsUpstreamDoes()
    {
        byte[] rgb = Enumerable.Repeat((byte)0, 4 * 4 * 3).ToArray();

        PreparedImage image = Hunyuan3DImageProcessor.Process(rgb, null, 4, 4, size: 4);

        Assert.Equal(new byte[] { 0, 0, 0 }, Pixel(image, 0, 0));
        Assert.Equal(new byte[] { 0, 0, 0 }, Pixel(image, 2, 2));
        Assert.Equal(new byte[] { 255, 255, 255 }, Pixel(image, 3, 3));
    }

    [Fact]
    public void Hunyuan3D_ResizesTheCanvasToTheModelSize()
    {
        (byte[] rgb, byte[] alpha) = Block(8, 8, 2, 2, 6, 6, 255, 0, 0);

        PreparedImage image = Hunyuan3DImageProcessor.Process(rgb, alpha, 8, 8, size: 16);

        Assert.Equal((16, 16), (image.Width, image.Height));
        Assert.Equal(new byte[] { 255, 0, 0 }, Pixel(image, 8, 8));
        Assert.InRange(Pixel(image, 0, 0)[1], 240, 255);
    }

    [Fact]
    public void Hunyuan3D_PutsAPartiallyTransparentPixelBetweenItsColourAndWhite()
    {
        (byte[] rgb, byte[] alpha) = Block(8, 8, 2, 2, 6, 6, 0, 0, 0);
        alpha[3 * 8 + 3] = 128;

        PreparedImage image = Hunyuan3DImageProcessor.Process(rgb, alpha, 8, 8, size: 8);

        // The pixel sits inside the foreground box; the 2x enlargement blends it with its opaque neighbours, so it is darker than half grey but not black.
        int value = Pixel(image, 3, 3)[0];
        Assert.InRange(value, 1, 254);
    }

    [Fact]
    public void Hunyuan3D_RefusesAnEmptyOrOnePixelWideForeground()
    {
        (byte[] rgb, byte[] empty) = Block(8, 8, 0, 0, 0, 0, 1, 1, 1);
        (_, byte[] thin) = Block(8, 8, 3, 1, 4, 7, 1, 1, 1);

        Assert.Throws<ArgumentException>(() => Hunyuan3DImageProcessor.Process(rgb, empty, 8, 8));
        Assert.Throws<ArgumentException>(() => Hunyuan3DImageProcessor.Process(rgb, thin, 8, 8));
    }

    [Fact]
    public void TripoSr_CropsPadsToASquareAndThenToTheRatioOnGray()
    {
        (byte[] rgb, byte[] alpha) = Block(10, 10, 3, 2, 7, 8, 255, 255, 255);

        PreparedImage image = TripoSrImageProcessor.Process(rgb, alpha, 10, 10, foregroundRatio: 0.5);

        // The inclusive extremes give a 3 x 5 crop (x 3..5, y 2..6); 5 / 0.5 = 10; offsets (5 - 3) / 2 + 2 = 3 across and (5 - 5) / 2 + 2 = 2 down.
        Assert.Equal((10, 10), (image.Width, image.Height));
        for (int y = 0; y < 10; y++)
        {
            for (int x = 0; x < 10; x++)
            {
                bool inside = x is >= 3 and < 6 && y is >= 2 and < 7;
                Assert.Equal(inside ? new byte[] { 255, 255, 255 } : new byte[] { 127, 127, 127 }, Pixel(image, x, y));
            }
        }
    }

    [Fact]
    public void TripoSr_UsesTheUpstreamRatioByDefault()
    {
        (byte[] rgb, byte[] alpha) = Block(40, 40, 10, 10, 30, 30, 255, 255, 255);

        PreparedImage image = TripoSrImageProcessor.Process(rgb, alpha, 40, 40);

        Assert.Equal((int)(19 / 0.85), image.Width);
    }

    [Fact]
    public void TripoSr_RefusesAnImageWithNoForeground()
    {
        (byte[] rgb, byte[] empty) = Block(8, 8, 0, 0, 0, 0, 1, 1, 1);

        Assert.Throws<ArgumentException>(() => TripoSrImageProcessor.Process(rgb, empty, 8, 8));
    }

    [Fact]
    public void Trellis_CropsASquareOfOnePointTwoTimesTheForegroundAndPremultipliesByAlpha()
    {
        (byte[] rgb, byte[] alpha) = Block(20, 20, 5, 5, 15, 15, 255, 255, 255);

        PreparedImage image = TrellisImageProcessor.Process(rgb, alpha, 20, 20, size: 10);

        // Box 5..14, centre 9.5, side int(9 * 1.2) = 10; PIL rounds 4.5 and 14.5 half to even to 4 and 14, so the crop is x 4..13 and column 0 is transparent.
        Assert.Equal((10, 10), (image.Width, image.Height));
        Assert.Equal(new byte[] { 0, 0, 0 }, Pixel(image, 0, 5));
        Assert.Equal(new byte[] { 0, 0, 0 }, Pixel(image, 5, 0));
        Assert.Equal(new byte[] { 255, 255, 255 }, Pixel(image, 5, 5));
        Assert.Equal(new byte[] { 255, 255, 255 }, Pixel(image, 9, 9));
    }

    [Fact]
    public void Trellis_ResizesToTheConditionerSizeAndKeepsTheBackgroundBlack()
    {
        (byte[] rgb, byte[] alpha) = Block(32, 32, 8, 8, 24, 24, 200, 100, 50);

        PreparedImage image = TrellisImageProcessor.Process(rgb, alpha, 32, 32);

        Assert.Equal((518, 518), (image.Width, image.Height));
        Assert.Equal(new byte[] { 0, 0, 0 }, Pixel(image, 0, 0));
        byte[] centre = Pixel(image, 259, 259);
        Assert.InRange(centre[0], 199, 201);
        Assert.InRange(centre[2], 49, 51);
    }

    [Fact]
    public void Trellis_RefusesAnImageWhoseCoverageNeverPassesEightyPercent()
    {
        (byte[] rgb, byte[] alpha) = Block(8, 8, 0, 0, 8, 8, 1, 1, 1);
        Array.Fill(alpha, (byte)100);

        Assert.Throws<ArgumentException>(() => TrellisImageProcessor.Process(rgb, alpha, 8, 8));
    }

    [Fact]
    public void AllProcessorsCheckTheBufferSizes()
    {
        Assert.Throws<ArgumentException>(() => Hunyuan3DImageProcessor.Process(new byte[5], null, 2, 2));
        Assert.Throws<ArgumentException>(() => TripoSrImageProcessor.Process(new byte[12], new byte[3], 2, 2));
        Assert.Throws<ArgumentException>(() => TrellisImageProcessor.Process(new byte[12], new byte[3], 2, 2));
    }
}
