using System.Text.Json;
using HartsyInference.Core.Imaging;
using HartsyInference.LLM.Models.MossVl;

namespace HartsyInference.LLM.Tests;

/// <summary>MOSS-VL image preprocessing against <c>transformers</c>' Qwen2-VL fast processor with MOSS's settings
/// (<c>tests/python-reference/dump_moss_vl_preprocess.py</c>). The fixtures are committed: synthetic 8-bit images and the
/// processor's patch rows, covering a downscale, a tall image, a binding pixel cap and an upscale below the minimum.</summary>
public sealed class MossVlPreprocessParityTests
{
    private static readonly string Dir = Path.Combine(AppContext.BaseDirectory, "Fixtures", "MossVl", "Preprocess");

    /// <summary>One 8-bit level of the normalized range (2/255). Measured: 0.2–0.3% of values differ, each by exactly one level
    /// (rounding of the 8-bit resize); without the per-pass rounding nearly every value differs.</summary>
    private const float OneLevel = 2f / 255f;

    public static TheoryData<int> Cases() => new() { 0, 1, 2, 3 };

    [Theory]
    [MemberData(nameof(Cases))]
    public void PatchRowsMatchTheHuggingFaceProcessor(int index)
    {
        using JsonDocument meta = JsonDocument.Parse(File.ReadAllText(Path.Combine(Dir, "meta.json")));
        JsonElement c = meta.RootElement.GetProperty("cases")[index];
        int width = c.GetProperty("width").GetInt32(), height = c.GetProperty("height").GetInt32();
        long maxPixels = c.GetProperty("max_pixels").GetInt64();
        int[] grid = c.GetProperty("grid").EnumerateArray().Select(e => e.GetInt32()).ToArray();
        byte[] rgb = File.ReadAllBytes(Path.Combine(Dir, $"image_{index}.rgb"));
        (long[] refShape, float[] reference) = NpyReader.ReadFloat32(Path.Combine(Dir, $"pixel_values_{index}.npy"));

        using VisionPatchGrid actual = MossVlImageProcessor.Default.Preprocess(new RawRgbFrame(rgb, width, height), maxPixels);

        Assert.Equal(grid, new[] { actual.GridT, actual.GridH, actual.GridW });
        Assert.Equal(refShape, new[] { actual.PixelValues.Shape[0], actual.PixelValues.Shape[1] });
        ReadOnlySpan<float> rows = actual.PixelValues.AsSpan<float>();
        double maxDiff = 0;
        int offByMoreThanALevel = 0, differing = 0;
        for (int i = 0; i < reference.Length; i++)
        {
            double d = Math.Abs(rows[i] - reference[i]);
            maxDiff = Math.Max(maxDiff, d);
            if (d > OneLevel * 1.01) offByMoreThanALevel++;
            if (d > 1e-6) differing++;
        }
        Console.WriteLine($"MOSS-VL preprocess case {index} ({width}x{height} -> grid {string.Join("x", grid)}): max|diff| {maxDiff:G4}, " +
                          $"values differing: {differing}, off by more than one 8-bit level: {offByMoreThanALevel}, of {reference.Length}.");
        Assert.Equal(0, offByMoreThanALevel);
        // torch resizes 8-bit images with fixed-point weights; float weights round a few pixels the other way.
        Assert.True(differing <= reference.Length / 100, $"{differing} of {reference.Length} values differ; more than 1% means the resize changed.");
        Assert.True(maxDiff <= OneLevel * 1.01, $"max|diff| {maxDiff:G4} exceeds one 8-bit level ({OneLevel:G4}).");
    }
}
