using System.Text.Json;
using HartsyInference.Core.Imaging;
using HartsyInference.LLM.Models.MossVl;

namespace HartsyInference.LLM.Tests;

/// <summary>MOSS-VL's batch pixel budget and config reading, on synthetic images (no weights).</summary>
public sealed class MossVlImageProcessorTests
{
    private static RawRgbFrame Gradient(int width, int height)
    {
        byte[] rgb = new byte[width * height * 3];
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
            {
                int i = (y * width + x) * 3;
                rgb[i] = (byte)(x * 255 / Math.Max(1, width - 1));
                rgb[i + 1] = (byte)(y * 255 / Math.Max(1, height - 1));
                rgb[i + 2] = 128;
            }
        return new RawRgbFrame(rgb, width, height);
    }

    [Fact]
    public void TwoImagesShareTheBatchBudgetByArea()
    {
        // Budget 160,000 over areas 1:3 → shares 40,000 and 120,000, each below the per-image ceiling.
        MossVlImageProcessor processor = new(new VisionPatchSpec(PatchSize: 16, MergeSize: 2, TemporalPatchSize: 1),
            [0.5f, 0.5f, 0.5f], [0.5f, 0.5f, 0.5f], minPixels: 4096, maxPixels: 16_777_216, batchBudget: 160_000);
        RawRgbFrame small = Gradient(400, 400), large = Gradient(1200, 400);

        VisionPatchGrid[] batch = processor.PreprocessBatch([small, large]);
        try
        {
            using VisionPatchGrid smallAlone = processor.Preprocess(small, 40_000);
            using VisionPatchGrid largeAlone = processor.Preprocess(large, 120_000);
            Assert.Equal((smallAlone.GridH, smallAlone.GridW), (batch[0].GridH, batch[0].GridW));
            Assert.Equal((largeAlone.GridH, largeAlone.GridW), (batch[1].GridH, batch[1].GridW));
            Assert.True(batch[0].GridH * batch[0].GridW * 256 <= 40_000);
            Assert.True(batch[1].GridH * batch[1].GridW * 256 <= 120_000);
            Assert.Equal(smallAlone.PixelValues.AsSpan<float>().ToArray(), batch[0].PixelValues.AsSpan<float>().ToArray());

            // On its own, an image gets the whole budget, so it is resized larger than its batch share.
            using VisionPatchGrid smallSolo = processor.Preprocess(small);
            Assert.True(smallSolo.GridH * smallSolo.GridW > batch[0].GridH * batch[0].GridW);
        }
        finally
        {
            foreach (VisionPatchGrid g in batch) g.Dispose();
        }
    }

    [Fact]
    public void PreprocessAtKeepsTheGivenSize()
    {
        using VisionPatchGrid grid = MossVlImageProcessor.Default.PreprocessAt(Gradient(300, 200), 192, 288);
        // 192×288 at patch 16 is a 12×18 patch grid; no smart-resize runs, so the grid is exactly that.
        Assert.Equal((1, 12, 18), (grid.GridT, grid.GridH, grid.GridW));
    }

    [Fact]
    public void TopLevelPixelBoundsOverrideSize()
    {
        using JsonDocument doc = JsonDocument.Parse("""
            {"size": {"shortest_edge": 4096, "longest_edge": 16777216}, "min_pixels": 8192, "max_pixels": 1048576,
             "multi_image_max_pixels": 2000000}
            """);
        MossVlImageProcessor processor = MossVlImageProcessor.FromJson(doc.RootElement);
        Assert.Equal((8192L, 1_048_576L, 2_000_000L), (processor.MinPixels, processor.MaxPixels, processor.BatchBudget));
    }
}
