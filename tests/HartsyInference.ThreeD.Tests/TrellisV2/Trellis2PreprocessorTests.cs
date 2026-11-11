using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Exceptions;
using HartsyInference.ThreeD.Models.TrellisV2;
using HartsyInference.ThreeD.Pipelines.Requests;
using HartsyInference.ThreeD.Pipelines.Stages;
using Xunit;

namespace HartsyInference.ThreeD.Tests.TrellisV2;

public sealed class Trellis2PreprocessorTests
{
    private static Trellis2InputImage Image(int width, int height, byte alpha, Func<int, int, byte>? alphaAt = null)
    {
        byte[] rgb = new byte[width * height * 3];
        byte[] a = new byte[width * height];
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int i = y * width + x;
                rgb[i * 3] = 200;
                rgb[i * 3 + 1] = 100;
                rgb[i * 3 + 2] = 50;
                a[i] = alphaAt?.Invoke(x, y) ?? alpha;
            }
        }

        return new Trellis2InputImage { Rgb = rgb, Alpha = a, Width = width, Height = height };
    }

    [Fact]
    public void Crop_FitsTheSquareAroundTheForegroundAndRoundsHalfToEven()
    {
        // Foreground x 2..5, y 3..8: centre (3.5, 5.5), size 5, size // 2 = 2, so the box is (1.5, 3.5, 5.5, 7.5).
        // PIL rounds each edge half to even: (2, 4, 6, 8), a 4x4 crop that drops the foreground's first row.
        byte[] alpha = new byte[10 * 10];
        for (int y = 3; y <= 8; y++)
        {
            for (int x = 2; x <= 5; x++)
                alpha[y * 10 + x] = 255;
        }

        Trellis2PreparedImage prepared = Trellis2ImagePreprocessor.Crop(new byte[10 * 10 * 3], alpha, 10, 10);

        Assert.Equal((4, 4), (prepared.Width, prepared.Height));
    }

    [Theory]
    [InlineData(1.0, 40)]
    [InlineData(1.1, 44)]
    [InlineData(1.25, 50)]
    public void Crop_WidensTheSquareByTruncatingSizeTimesScale(double scale, int expected)
    {
        // Foreground x 10..50 is 40 wide (upstream's max bbox extent); Pixal3D crops int(size * 1.1), TRELLIS.2 int(size * 1).
        byte[] alpha = new byte[100 * 100];
        for (int y = 10; y <= 50; y++)
        {
            for (int x = 10; x <= 50; x++)
                alpha[y * 100 + x] = 255;
        }

        Trellis2PreparedImage prepared = Trellis2ImagePreprocessor.Crop(new byte[100 * 100 * 3], alpha, 100, 100, scale);

        Assert.Equal((expected, expected), (prepared.Width, prepared.Height));
    }

    [Fact]
    public void Crop_WithAnOddWidenedSizeDropsTheTrailingPixelLikePilsHalfSizeBox()
    {
        // Foreground x 10..50 is 40 wide, so a 1.125 scale gives int(45) = 45; size // 2 = 22 puts the box at (center ± 22), a 44 px crop.
        byte[] alpha = new byte[100 * 100];
        for (int y = 10; y <= 50; y++)
        {
            for (int x = 10; x <= 50; x++)
                alpha[y * 100 + x] = 255;
        }

        Trellis2PreparedImage prepared = Trellis2ImagePreprocessor.Crop(new byte[100 * 100 * 3], alpha, 100, 100, 1.125);

        Assert.Equal((44, 44), (prepared.Width, prepared.Height));
    }

    [Fact]
    public void Crop_ThresholdIsStrictlyAboveZeroPointEightOfFullAlpha()
    {
        // The stray pixel has alpha 204 = 0.8 · 255, which is not foreground; had it counted the crop would span the corner.
        byte[] alpha = new byte[8 * 8];
        alpha[0] = 204;
        for (int y = 3; y <= 6; y++)
        {
            for (int x = 3; x <= 6; x++)
                alpha[y * 8 + x] = 255;
        }

        // Foreground 3..6: centre 4.5, size 3, size // 2 = 1, box (3.5, 3.5, 5.5, 5.5) -> (4, 4, 6, 6).
        Trellis2PreparedImage prepared = Trellis2ImagePreprocessor.Crop(new byte[8 * 8 * 3], alpha, 8, 8);

        Assert.Equal((2, 2), (prepared.Width, prepared.Height));
    }

    [Fact]
    public void Crop_ZeroesTheCropAreaThatLiesOutsideTheImageAndPremultipliesTheRest()
    {
        // Foreground x 0..1, y 0..6: centre (0.5, 3), size 6, size // 2 = 3, box (-2.5, 0, 3.5, 6) -> (-2, 0, 4, 6): a 6x6 crop whose
        // first two columns are outside the image (zero), the next two the foreground, the last two transparent pixels (zero).
        byte[] rgb = new byte[8 * 8 * 3];
        byte[] alpha = new byte[64];
        for (int i = 0; i < 64; i++)
        {
            rgb[i * 3] = 200;
            rgb[i * 3 + 1] = 100;
            rgb[i * 3 + 2] = 50;
        }

        for (int y = 0; y <= 6; y++)
        {
            alpha[y * 8] = 255;
            alpha[y * 8 + 1] = 255;
        }

        Trellis2PreparedImage prepared = Trellis2ImagePreprocessor.Crop(rgb, alpha, 8, 8);

        Assert.Equal((6, 6), (prepared.Width, prepared.Height));
        Assert.Equal([0, 0, 0], prepared.Rgb[..3]);
        Assert.Equal([Premultiplied(200, 255), Premultiplied(100, 255), Premultiplied(50, 255)], prepared.Rgb[6..9]);
        Assert.Equal([0, 0, 0], prepared.Rgb[12..15]);
    }

    [Fact]
    public void Crop_ScalesPartialAlphaWithFloat32Truncation()
    {
        byte[] rgb = new byte[3 * 3 * 3];
        for (int i = 0; i < 9; i++)
        {
            rgb[i * 3] = 200;
            rgb[i * 3 + 1] = 100;
            rgb[i * 3 + 2] = 50;
        }

        byte[] alpha = [255, 230, 255, 255, 255, 255, 255, 255, 255];

        // Foreground is all nine pixels: size 2, centre (1, 1), box (0, 0, 2, 2). Crop pixel (1, 0) has alpha 230.
        Trellis2PreparedImage prepared = Trellis2ImagePreprocessor.Crop(rgb, alpha, 3, 3);

        Assert.Equal((2, 2), (prepared.Width, prepared.Height));
        Assert.Equal(Premultiplied(200, 230), prepared.Rgb[3]);
        Assert.Equal(Premultiplied(100, 230), prepared.Rgb[4]);
        Assert.Equal(Premultiplied(200, 255), prepared.Rgb[0]);
    }

    [Fact]
    public void Crop_ASinglePixelWideForegroundIsRejected()
    {
        byte[] alpha = new byte[16];
        alpha[5] = 255;

        Assert.Throws<HartsyInferenceException>(() => Trellis2ImagePreprocessor.Crop(new byte[48], alpha, 4, 4));
    }

    [Fact]
    public void Prepare_UsesTheImagesOwnAlphaWithoutCallingTheMatter()
    {
        Trellis2InputImage image = Image(16, 16, 0, (x, y) => x is >= 4 and <= 11 && y is >= 4 and <= 11 ? (byte)255 : (byte)0);
        ThrowingMatter matter = new();
        using CpuBackend backend = new();

        Trellis2PreparedImage prepared = Trellis2ImagePreprocessor.Prepare(backend, image, matter);

        Assert.Equal(0, matter.Calls);
        Assert.True(prepared.Width > 0 && prepared.Height > 0);
    }

    [Fact]
    public void Prepare_MattesAnOpaqueImageAndTruncatesTheMatteToEightBits()
    {
        Trellis2InputImage opaque = Image(16, 16, 255);
        // Matte 0.999 truncates to 254 (> 204: foreground), 0.5 to 127 (background).
        float[] matte = new float[256];
        for (int y = 4; y <= 11; y++)
        {
            for (int x = 4; x <= 11; x++)
                matte[y * 16 + x] = 0.999f;
        }

        RecordingMatter matter = new(matte);
        using CpuBackend backend = new();

        Trellis2PreparedImage prepared = Trellis2ImagePreprocessor.Prepare(backend, opaque, matter);

        Assert.Equal(1, matter.Calls);
        Assert.Equal((16, 16), (matter.Width, matter.Height));
        // 8x8 foreground x/y 4..11: centre 7.5, size 7, size // 2 = 3, box (4.5, 4.5, 10.5, 10.5) -> (4, 4, 10, 10): a 6x6 crop.
        Assert.Equal((6, 6), (prepared.Width, prepared.Height));
        // alpha 254 = int(0.999 · 255): the matte truncates to 8 bits before the crop and the premultiply.
        Assert.Equal(Premultiplied(200, 254), prepared.Rgb[0]);
    }

    [Fact]
    public void Prepare_ShrinksALongEdgeAbove1024BeforeMatting()
    {
        Trellis2InputImage opaque = Image(2048, 1024, 255);
        RecordingMatter matter = new(new float[1024 * 512]);
        using CpuBackend backend = new();

        // The matte is empty, so preparation fails, but only after the matter saw the image at half size.
        Assert.Throws<HartsyInferenceException>(() => Trellis2ImagePreprocessor.Prepare(backend, opaque, matter));

        Assert.Equal((1024, 512), (matter.Width, matter.Height));
    }

    [Fact]
    public void Prepare_WithoutAlphaOrMatterExplainsWhatToSupply()
    {
        Trellis2InputImage opaque = Image(8, 8, 255);
        using CpuBackend backend = new();

        HartsyInferenceException error = Assert.Throws<HartsyInferenceException>(() => Trellis2ImagePreprocessor.Prepare(backend, opaque, null));

        Assert.Contains("alpha", error.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Prepare_RejectsAnImageWithNoForegroundPixel()
    {
        Trellis2InputImage faint = Image(8, 8, 100);
        using CpuBackend backend = new();

        Assert.Throws<HartsyInferenceException>(() => Trellis2ImagePreprocessor.Prepare(backend, faint, null));
    }

    [Fact]
    public void ConditionerPixels_ResizeToTheResolutionAndNormalizeWithImageNetStatistics()
    {
        byte[] rgb = new byte[4 * 4 * 3];
        Array.Fill(rgb, (byte)255);

        using HartsyInference.Core.Tensors.Tensor pixels = Trellis2ImageConditioner.Pixels(rgb, 4, 4, 16);

        Assert.Equal(new HartsyInference.Core.Tensors.TensorShape(1, 3, 16, 16), pixels.Shape);
        ReadOnlySpan<float> values = pixels.AsReadOnlySpan<float>();
        Assert.Equal((1f - 0.485f) / 0.229f, values[0], 4);
        Assert.Equal((1f - 0.456f) / 0.224f, values[256], 4);
        Assert.Equal((1f - 0.406f) / 0.225f, values[512 + 255], 4);
    }

    [Fact]
    public void InputImage_ValidatesBufferSizes()
    {
        Assert.Throws<ArgumentException>(() => new Trellis2InputImage { Rgb = new byte[5], Width = 2, Height = 2 }.Validate());
        Assert.Throws<ArgumentException>(() => new Trellis2InputImage { Rgb = new byte[12], Alpha = new byte[3], Width = 2, Height = 2 }.Validate());
        Assert.False(new Trellis2InputImage { Rgb = new byte[12], Alpha = [255, 255, 255, 255], Width = 2, Height = 2 }.HasForegroundAlpha);
        Assert.True(new Trellis2InputImage { Rgb = new byte[12], Alpha = [255, 255, 255, 254], Width = 2, Height = 2 }.HasForegroundAlpha);
    }

    // Upstream multiplies float32 rgb/255 by float32 alpha/255, scales by 255 and truncates to uint8.
    private static byte Premultiplied(int channel, int alpha) => (byte)(channel / 255f * (alpha / 255f) * 255f);

    private sealed class ThrowingMatter : IForegroundMatter
    {
        public int Calls { get; private set; }

        public float[] Alpha(IBackend backend, ReadOnlySpan<byte> rgb, int width, int height)
        {
            Calls++;
            throw new InvalidOperationException("The matter must not run when the image has alpha.");
        }
    }

    private sealed class RecordingMatter(float[] matte) : IForegroundMatter
    {
        public int Calls { get; private set; }

        public int Width { get; private set; }

        public int Height { get; private set; }

        public float[] Alpha(IBackend backend, ReadOnlySpan<byte> rgb, int width, int height)
        {
            Calls++;
            Width = width;
            Height = height;
            return matte;
        }
    }
}
