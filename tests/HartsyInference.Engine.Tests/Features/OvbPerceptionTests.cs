using System.Text.RegularExpressions;
using HartsyInference.Engine.Features;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Vision.OpenVisionBanana;

namespace HartsyInference.Engine.Tests.Features;

/// <summary>Open Vision Banana's perception adapter with a fake picture source: the fake paints a known answer for whatever
/// legend the prompt carries, the way a perfect model would, so what is under test is the request it is sent, the resize back
/// to the photo's size, the snap, the vote and the decode, not the model.</summary>
public sealed partial class OvbPerceptionTests
{
    private const int SourceWidth = 40;
    private const int SourceHeight = 30;
    private const int PaintedWidth = 20;
    private const int PaintedHeight = 15;

    [GeneratedRegex("\"([^\"]+)\": \\[(\\d+),(\\d+),(\\d+)\\]")]
    private static partial Regex LegendEntry();

    [Fact]
    public async Task SegmentationSendsTheVerbatimPromptWithThePhotoAsAReference()
    {
        List<ImageRequest> sent = [];
        ImageData photo = Photo();

        await OvbPerception.SegmentAsync(Painter(sent, TruthByHalf), photo, ["road", "sky"], legends: 1, seed: 5, OvbSemsegStyle.Quickstart, default);

        ImageRequest request = Assert.Single(sent);
        Assert.StartsWith("[TASK: SEMANTIC_SEGMENTATION] Generate a semantic segmentation visualization of the input image, using this color mapping: {\"road\": [", request.Prompt);
        Assert.Equal(Img2ImgMode.Reference, request.Img2Img!.Mode);
        Assert.Same(photo, request.Img2Img.InitImage);
        Assert.Equal(5, request.Seed);
    }

    [Fact]
    public async Task SegmentationReturnsLabelsAtThePhotosSizeFromAHalfSizePicture()
    {
        OvbSegmentation result = await OvbPerception.SegmentAsync(Painter([], TruthByHalf), Photo(), ["road", "sky"], 1, 0, OvbSemsegStyle.Quickstart, default);

        Assert.Equal((SourceWidth, SourceHeight), (result.Width, result.Height));
        Assert.Equal(SourceWidth * SourceHeight, result.Labels.Length);
        // Top half is class 0 ("road"), bottom half class 1 ("sky"), after the nearest-neighbour resize.
        Assert.All(Enumerable.Range(0, SourceWidth), x => Assert.Equal(0, result.Labels[x]));
        Assert.All(Enumerable.Range(0, SourceWidth), x => Assert.Equal(1, result.Labels[(SourceHeight - 1) * SourceWidth + x]));
        Assert.All(result.Agreement, a => Assert.Equal(1f, a));
    }

    [Fact]
    public async Task LegendEnsemblingUsesADifferentLegendPerDrawAndOutvotesOneBadPass()
    {
        List<ImageRequest> sent = [];
        int call = 0;
        // The second pass paints everything black: it abstains, and the two good passes carry the vote.
        Func<ImageRequest, CancellationToken, Task<ImageResult>> generate = (request, cancel) =>
        {
            sent.Add(request);
            return call++ == 1 ? Task.FromResult(Flat(0, 0, 0)) : Painter([], TruthByHalf)(request, cancel);
        };

        OvbSegmentation result = await OvbPerception.SegmentAsync(generate, Photo(), ["road", "sky"], legends: 3, seed: 1, OvbSemsegStyle.Quickstart, default);

        Assert.Equal(3, sent.Count);
        Assert.Equal(3, sent.Select(r => r.Prompt).Distinct().Count());
        Assert.Equal(3, sent.Select(r => r.Seed).Distinct().Count());
        Assert.Equal(0, result.Labels[0]);
        Assert.Equal(1, result.Labels[(SourceHeight - 1) * SourceWidth]);
        Assert.All(result.Agreement, a => Assert.Equal(1f, a));
    }

    [Fact]
    public async Task ReferringSegmentationMasksTheRedRegion()
    {
        (byte[] mask, _) = await OvbPerception.ReferAsync((request, cancel) =>
        {
            Assert.Contains("\"table in the room\"", request.Prompt);
            byte[] rgb = new byte[PaintedWidth * PaintedHeight * 3];
            for (int y = 0; y < PaintedHeight; y++)
                for (int x = 0; x < PaintedWidth; x++)
                    if (x < PaintedWidth / 2)
                        (rgb[(y * PaintedWidth + x) * 3], rgb[(y * PaintedWidth + x) * 3 + 1], rgb[(y * PaintedWidth + x) * 3 + 2]) = (222, 28, 31);
            return Task.FromResult(new ImageResult { Rgb = rgb, Width = PaintedWidth, Height = PaintedHeight });
        }, Photo(), "table in the room", 0, default);

        Assert.Equal(SourceWidth * SourceHeight, mask.Length);
        Assert.Equal(255, mask[0]);
        Assert.Equal(0, mask[SourceWidth - 1]);
    }

    [Fact]
    public async Task DepthIsResizedBilinearlyAndReadAsLuma()
    {
        List<ImageRequest> sent = [];
        OvbDepth depth = await OvbPerception.DepthAsync((request, cancel) =>
        {
            sent.Add(request);
            byte[] rgb = new byte[PaintedWidth * PaintedHeight * 3];
            for (int y = 0; y < PaintedHeight; y++)
                for (int x = 0; x < PaintedWidth; x++)
                {
                    byte v = (byte)(x * 255 / (PaintedWidth - 1));
                    (rgb[(y * PaintedWidth + x) * 3], rgb[(y * PaintedWidth + x) * 3 + 1], rgb[(y * PaintedWidth + x) * 3 + 2]) = (v, v, v);
                }
            return Task.FromResult(new ImageResult { Rgb = rgb, Width = PaintedWidth, Height = PaintedHeight });
        }, Photo(), 0, default);

        Assert.Equal(OvbPrompts.Depth, Assert.Single(sent).Prompt);
        Assert.Equal((SourceWidth, SourceHeight), (depth.Width, depth.Height));
        Assert.Equal(SourceWidth * SourceHeight, depth.InverseDepth.Length);
        Assert.All(depth.InverseDepth, v => Assert.InRange(v, 0f, 1f));
        Assert.True(depth.InverseDepth[SourceWidth - 1] > depth.InverseDepth[0], "brighter on the right");
        for (int x = 1; x < SourceWidth; x++)
            Assert.True(depth.InverseDepth[x] >= depth.InverseDepth[x - 1] - 1e-6f, $"monotone at {x}");
    }

    [Fact]
    public async Task NormalsFacingTheCameraDecodeToPlusZ()
    {
        OvbNormals normals = await OvbPerception.NormalsAsync((request, cancel) =>
        {
            Assert.Equal(OvbPrompts.Normals, request.Prompt);
            // R=(1-x)/2, G=(1+y)/2, B=(1+z)/2 for (x,y,z) = (0,0,1): about (128, 128, 255).
            return Task.FromResult(Flat(128, 128, 255));
        }, Photo(), 0, default);

        Assert.Equal((SourceWidth, SourceHeight), (normals.Width, normals.Height));
        Assert.All(normals.Valid, v => Assert.True(v));
        Assert.InRange(normals.Normals[0], -0.02f, 0.02f);
        Assert.InRange(normals.Normals[1], -0.02f, 0.02f);
        Assert.InRange(normals.Normals[2], 0.99f, 1.0001f);
    }

    [Fact]
    public async Task ARequestForTooManyOrNoLegendsIsRefused()
    {
        await Assert.ThrowsAsync<ArgumentOutOfRangeException>(() =>
            OvbPerception.SegmentAsync(Painter([], TruthByHalf), Photo(), ["a"], 0, 0, OvbSemsegStyle.Quickstart, default));
        await Assert.ThrowsAsync<ArgumentOutOfRangeException>(() =>
            OvbPerception.SegmentAsync(Painter([], TruthByHalf), Photo(), [], 1, 0, OvbSemsegStyle.Quickstart, default));
    }

    private static int TruthByHalf(int x, int y) => y < PaintedHeight / 2 ? 0 : 1;

    private static ImageData Photo()
    {
        byte[] rgb = new byte[SourceWidth * SourceHeight * 3];
        Array.Fill(rgb, (byte)90);
        return new ImageData { Rgb = rgb, Width = SourceWidth, Height = SourceHeight };
    }

    private static ImageResult Flat(byte r, byte g, byte b)
    {
        byte[] rgb = new byte[PaintedWidth * PaintedHeight * 3];
        for (int i = 0; i < rgb.Length; i += 3) (rgb[i], rgb[i + 1], rgb[i + 2]) = (r, g, b);
        return new ImageResult { Rgb = rgb, Width = PaintedWidth, Height = PaintedHeight };
    }

    /// <summary>A picture source that paints <paramref name="truth"/>(x, y) in the colour the prompt's legend gives that class.</summary>
    private static Func<ImageRequest, CancellationToken, Task<ImageResult>> Painter(List<ImageRequest> sent, Func<int, int, int> truth) =>
        (request, cancel) =>
        {
            sent.Add(request);
            List<(byte R, byte G, byte B)> colors = LegendEntry().Matches(request.Prompt)
                .Select(m => ((byte)int.Parse(m.Groups[2].Value), (byte)int.Parse(m.Groups[3].Value), (byte)int.Parse(m.Groups[4].Value))).ToList();
            byte[] rgb = new byte[PaintedWidth * PaintedHeight * 3];
            for (int y = 0; y < PaintedHeight; y++)
                for (int x = 0; x < PaintedWidth; x++)
                    (rgb[(y * PaintedWidth + x) * 3], rgb[(y * PaintedWidth + x) * 3 + 1], rgb[(y * PaintedWidth + x) * 3 + 2]) = colors[truth(x, y)];
            return Task.FromResult(new ImageResult { Rgb = rgb, Width = PaintedWidth, Height = PaintedHeight });
        };
}
