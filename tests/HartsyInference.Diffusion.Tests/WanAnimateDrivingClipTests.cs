using HartsyInference.Core.Tensors;
using HartsyInference.Engine.Recipes.Video;
using HartsyInference.Vision.FaceDetection;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Pins the pure math of the Wan-Animate driving-clip build: the frame-count snap-down rule, the
/// truncate-vs-repeat-pad semantics, the <c>[1,3,T,H,W]</c> packing layout with its [-1, 1] normalization, and the
/// face square-resample's out-of-frame mid-gray (114) pad. These break silently — a wrong layout or off-grid frame
/// count still generates a video, just the wrong one.</summary>
public sealed class WanAnimateDrivingClipTests
{
    private const int Step = 4;   // Wan VAE temporal compression

    [Theory]
    [InlineData(81, 200, 81)]    // clip longer than requested → requested count wins
    [InlineData(81, 81, 81)]     // exact fit stays
    [InlineData(81, 50, 49)]     // shorter clip → snapped DOWN onto 4n+1 (50 → 49)
    [InlineData(81, 48, 45)]     // 48 is off-grid → down to 45, never up to 49
    [InlineData(25, 3, 5)]       // tiny clip → hard minimum of 5
    [InlineData(2, 200, 5)]      // tiny request → hard minimum of 5
    [InlineData(5, 200, 5)]      // minimum itself is on-grid (4·1+1)
    public void ResolveDrivingFramesSnapsDownOntoTheTemporalGrid(int requested, int available, int expected)
    {
        Assert.Equal(expected, WanAnimateDrivingResolver.ResolveDrivingFrames(requested, available, Step));
    }

    [Fact]
    public void ResolvedFrameCountIsAlwaysOnTheGridAndAtLeastFive()
    {
        for (int requested = 1; requested <= 60; requested++)
        {
            for (int available = 1; available <= 60; available++)
            {
                int frames = WanAnimateDrivingResolver.ResolveDrivingFrames(requested, available, Step);
                Assert.True(frames >= 5, $"req={requested} avail={available} → {frames} < 5");
                Assert.Equal(1, frames % Step);
                Assert.True(frames <= Math.Max(5, Math.Min(requested, available)),
                    $"req={requested} avail={available} → {frames} exceeds min(requested, available)");
            }
        }
    }

    [Fact]
    public void FitFramesTruncatesALongerClip()
    {
        List<byte[]> frames = [.. Enumerable.Range(0, 10).Select(i => new byte[] { (byte)i })];
        List<byte[]> fitted = WanAnimateDrivingResolver.FitFrames(frames, 5);
        Assert.Equal(5, fitted.Count);
        Assert.Equal([0, 1, 2, 3, 4], fitted.Select(f => (int)f[0]));
    }

    [Fact]
    public void FitFramesRepeatPadsTheLastFrameOfAShorterClip()
    {
        List<byte[]> frames = [.. Enumerable.Range(0, 3).Select(i => new byte[] { (byte)i })];
        List<byte[]> fitted = WanAnimateDrivingResolver.FitFrames(frames, 5);
        Assert.Equal(5, fitted.Count);
        Assert.Equal([0, 1, 2, 2, 2], fitted.Select(f => (int)f[0]));
        Assert.Same(fitted[2], fitted[4]);
    }

    [Fact]
    public void PackRgbFramesToClipUsesChwPerChannelLayoutInMinusOneToOne()
    {
        const int Width = 2, Height = 2, Frames = 2;
        byte[][] rgb = new byte[Frames][];
        for (int f = 0; f < Frames; f++)
        {
            rgb[f] = new byte[Width * Height * 3];
            for (int pix = 0; pix < Width * Height; pix++)
            {
                for (int c = 0; c < 3; c++)
                {
                    rgb[f][pix * 3 + c] = (byte)(f * 100 + pix * 10 + c);
                }
            }
        }

        using Tensor clip = VideoRecipeUtils.PackRgbFramesToClip(rgb, Width, Height);
        Assert.Equal(new TensorShape([1L, 3, Frames, Height, Width]), clip.Shape);
        Span<float> data = clip.AsSpan<float>();
        long perFrame = Width * Height;
        for (int c = 0; c < 3; c++)
        {
            for (int f = 0; f < Frames; f++)
            {
                for (int pix = 0; pix < perFrame; pix++)
                {
                    float expected = rgb[f][pix * 3 + c] / 127.5f - 1f;
                    Assert.Equal(expected, data[(int)((c * (long)Frames + f) * perFrame + pix)], 6);
                }
            }
        }
    }

    [Fact]
    public void PackRgbFramesToClipNormalizesTheByteExtremes()
    {
        byte[][] rgb = [[0, 0, 0, 255, 255, 255]];   // one 2x1 frame: black then white pixels
        using Tensor clip = VideoRecipeUtils.PackRgbFramesToClip(rgb, width: 2, height: 1);
        Span<float> data = clip.AsSpan<float>();
        for (int c = 0; c < 3; c++)
        {
            Assert.Equal(-1f, data[c * 2 + 0], 6);
            Assert.Equal(1f, data[c * 2 + 1], 6);
        }
    }

    [Fact]
    public void FaceResamplePadsOutOfFrameSamplesWithNormalizedMidGray()
    {
        const int Size = 4;
        byte[] rgb = new byte[Size * Size * 3];
        Array.Fill(rgb, (byte)200);

        // Crop entirely off-image: every bilinear tap reads the 114 letterbox fill.
        PoseFaceCrop.Rect offImage = new PoseFaceCrop.Rect(-100f, -100f, 10f);
        float[] chw = WanAnimateFaceClipBuilder.SampleSquareChw(rgb, Size, Size, offImage, outSize: 8);
        float expectedPad = 114f / 127.5f - 1f;
        Assert.Equal(3 * 8 * 8, chw.Length);
        Assert.All(chw, v => Assert.Equal(expectedPad, v, 5));
    }

    [Fact]
    public void FaceResampleOfAnInteriorCropReadsTheSourceValues()
    {
        const int Size = 8;
        byte[] rgb = new byte[Size * Size * 3];
        Array.Fill(rgb, (byte)200);

        PoseFaceCrop.Rect interior = new PoseFaceCrop.Rect(2f, 2f, 4f);
        float[] chw = WanAnimateFaceClipBuilder.SampleSquareChw(rgb, Size, Size, interior, outSize: 4);
        float expected = 200f / 127.5f - 1f;
        Assert.All(chw, v => Assert.Equal(expected, v, 5));
    }

    [Fact]
    public void CenterSquareCropTakesTheShorterSideCentered()
    {
        PoseFaceCrop.Rect crop = WanAnimateFaceClipBuilder.CenterSquareCrop(width: 10, height: 6);
        Assert.Equal(6f, crop.Size);
        Assert.Equal(2f, crop.X);
        Assert.Equal(0f, crop.Y);
    }
}
