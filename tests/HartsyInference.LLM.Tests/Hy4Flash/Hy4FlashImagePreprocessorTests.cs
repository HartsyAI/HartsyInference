using System.Globalization;
using System.Text.Json;
using HartsyInference.LLM.Hy4Flash;

namespace HartsyInference.LLM.Tests.Hy4Flash;

/// <summary>The Hy4 vision preprocessing against the released <c>HYV4VLImageProcessor</c> (transformers eb1d5db): the smart resize table exactly, and the
/// patch rows of aligned images exactly (no resample). <c>dump_image_preprocess.py</c> writes the references.</summary>
public sealed class Hy4FlashImagePreprocessorTests
{
    private static string Dir
    {
        get
        {
            string? dir = AppContext.BaseDirectory;
            while (dir is not null && !Directory.Exists(Path.Combine(dir, "tests", "python-reference", "hy4_flash", "data", "image_preprocess"))) dir = Path.GetDirectoryName(dir);
            return Path.Combine(dir ?? throw new DirectoryNotFoundException("tests/python-reference/hy4_flash/data/image_preprocess not found."),
                "tests", "python-reference", "hy4_flash", "data", "image_preprocess");
        }
    }

    [Fact]
    public void SmartResize_MatchesTheProcessorTableExactly()
    {
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(Path.Combine(Dir, "smart_resize.json")));
        JsonElement root = document.RootElement;
        int factor = root.GetProperty("factor").GetInt32();
        long min = root.GetProperty("min_pixels").GetInt64(), max = root.GetProperty("max_pixels").GetInt64();
        int checkedCases = 0;
        foreach (JsonElement entry in root.GetProperty("cases").EnumerateArray())
        {
            int[] input = [.. entry.GetProperty("in").EnumerateArray().Select(static e => e.GetInt32())];
            int[] expected = [.. entry.GetProperty("out").EnumerateArray().Select(static e => e.GetInt32())];
            (int h, int w) = Hy4FlashImagePreprocessor.SmartResize(input[0], input[1], factor, min, max);
            Assert.True(h == expected[0] && w == expected[1], $"smart_resize({input[0]}, {input[1]}) gave ({h}, {w}), the processor gave ({expected[0]}, {expected[1]})");
            checkedCases++;
        }
        Assert.Equal(938, checkedCases);
    }

    [Theory]
    [InlineData("aligned_96x128")]
    [InlineData("aligned_256x160")]
    public void AlignedImages_PatchRowsMatchTheProcessorExactly(string name)
    {
        (int h, int w, int gridH, int gridW) = ReadGrid(name);
        byte[] rgb = File.ReadAllBytes(Path.Combine(Dir, name + "__rgb.u8"));
        (long[] shape, float[] expected) = NpyReader.ReadFloat32(Path.Combine(Dir, name + "__pixel_values.npy"));
        Assert.Equal(2, shape.Length);
        Assert.Equal(gridH * gridW, shape[0]);
        Assert.Equal(Hy4FlashImagePreprocessor.FeatureDim, shape[1]);

        Hy4FlashPatchedImage actual = Hy4FlashImagePreprocessor.Preprocess(rgb, w, h);
        Assert.Equal(gridH, actual.GridH);
        Assert.Equal(gridW, actual.GridW);
        Assert.Equal(expected.Length, actual.PixelValues.Length);
        float maxDifference = 0;
        for (int i = 0; i < expected.Length; i++) maxDifference = MathF.Max(maxDifference, MathF.Abs(expected[i] - actual.PixelValues[i]));
        Assert.True(maxDifference <= 1e-6f, $"{name}: max difference {maxDifference.ToString(CultureInfo.InvariantCulture)}");
    }

    [Theory]
    [InlineData("unaligned_37x53")]
    [InlineData("unaligned_300x200")]
    [InlineData("unaligned_17x1000")]
    public void UnalignedImages_GridAndSizeMatchTheProcessor(string name)
    {
        (int h, int w, int gridH, int gridW) = ReadGrid(name);
        (int resizedH, int resizedW) = ReadResized(name);
        byte[] rgb = File.ReadAllBytes(Path.Combine(Dir, name + "__rgb.u8"));
        Hy4FlashPatchedImage actual = Hy4FlashImagePreprocessor.Preprocess(rgb, w, h);
        Assert.Equal(resizedH, actual.ResizedHeight);
        Assert.Equal(resizedW, actual.ResizedWidth);
        Assert.Equal(gridH, actual.GridH);
        Assert.Equal(gridW, actual.GridW);

        // The resampler differs from the processor's, so the values are bounded, not equal. Measured on these three images: 0.007843 and 0.015686,
        // one and two 8-bit grey levels after normalisation (a level is 1/127.5). The bound is two levels.
        (long[] shape, float[] expected) = NpyReader.ReadFloat32(Path.Combine(Dir, name + "__pixel_values.npy"));
        Assert.Equal(gridH * gridW, shape[0]);
        float maxDifference = 0;
        for (int i = 0; i < expected.Length; i++) maxDifference = MathF.Max(maxDifference, MathF.Abs(expected[i] - actual.PixelValues[i]));
        Assert.True(maxDifference <= MaxUnalignedDifference, $"{name}: max difference {maxDifference.ToString("R", CultureInfo.InvariantCulture)} exceeds {MaxUnalignedDifference}");
    }

    private const float MaxUnalignedDifference = 0.0157f;

    private static (int Height, int Width, int GridH, int GridW) ReadGrid(string name)
    {
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(Path.Combine(Dir, name + "__grid.json")));
        JsonElement root = document.RootElement;
        JsonElement input = root.GetProperty("input");
        JsonElement grid = root.GetProperty("image_grid_thw");
        return (input[0].GetInt32(), input[1].GetInt32(), grid[1].GetInt32(), grid[2].GetInt32());
    }

    private static (int Height, int Width) ReadResized(string name)
    {
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(Path.Combine(Dir, name + "__grid.json")));
        JsonElement resized = document.RootElement.GetProperty("resized");
        return (resized[0].GetInt32(), resized[1].GetInt32());
    }
}
