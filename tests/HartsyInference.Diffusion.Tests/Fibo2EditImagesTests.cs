using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Engine.Recipes.Image;
using HartsyInference.Engine.Requests;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Fibo 2.0 edit image preparation: the trained-size search, the mask grey-out and the two copies of each image.</summary>
public sealed class Fibo2EditImagesTests
{
    [Fact]
    public void TrainedSize_Square_IsOneMegapixelSquare()
    {
        Assert.Equal((1024, 1024), Fibo2EditImages.TrainedSize(500, 500));
    }

    [Theory]
    [InlineData(1920, 1080)]
    [InlineData(640, 480)]
    [InlineData(300, 900)]
    [InlineData(1024, 704)]
    public void TrainedSize_IsAGridSizeNearOneMegapixelWithTheLeastCrop(int w, int h)
    {
        (int width, int height) = Fibo2EditImages.TrainedSize(w, h);

        Assert.Equal(0, width % 16);
        Assert.Equal(0, height % 16);
        long area = (long)width * height;
        Assert.InRange(area, 0.9 * 1024 * 1024, 1.1 * 1024 * 1024);
        // No other grid size within the band crops less.
        double aspect = w / (double)h;
        double Crop(int cw, int ch) => 1 - Math.Min(cw / (double)ch / aspect, aspect * ch / cw);
        double best = Crop(width, height);
        for (int cw = 64; cw <= 4096; cw += 16)
        {
            for (int ch = 64; ch <= 4096; ch += 16)
            {
                long a = (long)cw * ch;
                if (a < 0.9 * 1024 * 1024 || a > 1.1 * 1024 * 1024)
                    continue;
                Assert.True(Crop(cw, ch) >= best - 1e-9, $"{cw}x{ch} crops less than {width}x{height}");
            }
        }
    }

    [Fact]
    public void GreyOut_ReplacesOnlyTheMaskedPixelsAtTheBinaryThreshold()
    {
        ImageData image = new() { Rgb = [10, 20, 30, 40, 50, 60, 70, 80, 90, 1, 2, 3], Width = 4, Height = 1 };

        ImageData grey = Fibo2EditImages.GreyOut(image, [255, 127, 128, 0]);

        Assert.Equal(new byte[] { 128, 128, 128, 40, 50, 60, 128, 128, 128, 1, 2, 3 }, grey.Rgb);
        Assert.Equal(new byte[] { 10, 20, 30, 40, 50, 60, 70, 80, 90, 1, 2, 3 }, image.Rgb);
    }

    [Fact]
    public void Prepare_SingleImage_CropsToTheResultSizeAndSizesTheVisionCopyOnItsGrid()
    {
        Qwen3VlVisionConfig config = Qwen3VlVisionConfig.Qwen3Vl8B;
        Qwen3VlImageProcessor processor = new(config, 3136, 1605632, imageMean: [0.5f, 0.5f, 0.5f], imageStd: [0.5f, 0.5f, 0.5f]);
        ImageData image = Solid(2000, 1000);

        using Fibo2EditImages.Prepared prepared = Fibo2EditImages.Prepare([image], 1024, 704, processor, visionFactor: 32);

        Assert.Equal(new long[] { 1, 3, 704, 1024 }, Dims(prepared.Pixels[0]));
        // 1024x704 is thumbnailed to 768x528, then SmartResize puts it on the 32 grid.
        long[] vision = Dims(prepared.Vision[0]);
        Assert.Equal(3, vision[0]);
        Assert.Equal(0, vision[1] % 32);
        Assert.Equal(0, vision[2] % 32);
        Assert.Equal((int)(vision[1] / 32 * (vision[2] / 32)), prepared.PadCounts[0]);
        Assert.True(Math.Max(vision[1], vision[2]) <= 768);
    }

    [Fact]
    public void Prepare_SeveralImages_EachTakesItsOwnTrainedSize()
    {
        Qwen3VlVisionConfig config = Qwen3VlVisionConfig.Qwen3Vl8B;
        Qwen3VlImageProcessor processor = new(config, 3136, 1605632, imageMean: [0.5f, 0.5f, 0.5f], imageStd: [0.5f, 0.5f, 0.5f]);

        using Fibo2EditImages.Prepared prepared = Fibo2EditImages.Prepare([Solid(300, 300), Solid(900, 300)], 1024, 704, processor, 32);

        Assert.Equal(new long[] { 1, 3, 1024, 1024 }, Dims(prepared.Pixels[0]));
        (int w, int h) = Fibo2EditImages.TrainedSize(900, 300);
        Assert.Equal(new long[] { 1, 3, h, w }, Dims(prepared.Pixels[1]));
        Assert.Equal(2, prepared.PadCounts.Length);
    }

    private static long[] Dims(HartsyInference.Core.Tensors.Tensor tensor)
    {
        long[] dims = new long[tensor.Shape.Rank];
        for (int i = 0; i < dims.Length; i++) dims[i] = tensor.Shape[i];
        return dims;
    }

    private static ImageData Solid(int width, int height)
    {
        byte[] rgb = new byte[width * height * 3];
        for (int i = 0; i < rgb.Length; i++) rgb[i] = (byte)(i * 31 % 251);
        return new ImageData { Rgb = rgb, Width = width, Height = height };
    }
}
