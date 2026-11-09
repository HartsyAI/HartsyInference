using System.Text.Json;
using HartsyInference.Core.Imaging;

namespace HartsyInference.LLM.Models.MossVl;

/// <summary>MOSS-VL-Realtime's image processor: Qwen2-VL's fast processor (exact <c>smart_resize</c>, PIL bicubic on 8-bit pixels,
/// mean/std 0.5, merge-block patch rows) plus MOSS's batch pixel budget, which splits <c>multi_image_max_pixels</c> across the
/// images of one request in proportion to their areas. Realtime frames drained in one cycle form one batch.</summary>
public sealed class MossVlImageProcessor
{
    /// <summary>The checkpoint's <c>size.shortest_edge</c>: the smallest area an image is resized to.</summary>
    public const long DefaultMinPixels = 4096;

    /// <summary>The checkpoint's <c>size.longest_edge</c>: the largest area one image is resized to.</summary>
    public const long DefaultMaxPixels = 16_777_216;

    /// <summary>The checkpoint's <c>multi_image_max_pixels</c>: the area budget one batch of images shares.</summary>
    public const long DefaultBatchBudget = 65_536_000;

    private readonly VisionPatchPreprocessor _pre;

    /// <summary>Creates the processor with explicit settings.</summary>
    public MossVlImageProcessor(VisionPatchSpec spec, IReadOnlyList<float> mean, IReadOnlyList<float> std,
        long minPixels = DefaultMinPixels, long maxPixels = DefaultMaxPixels, long batchBudget = DefaultBatchBudget)
    {
        if (minPixels <= 0 || maxPixels < minPixels || batchBudget <= 0)
            throw new ArgumentException($"Invalid pixel bounds: min {minPixels}, max {maxPixels}, batch {batchBudget}.");
        _pre = new VisionPatchPreprocessor(spec, mean, std, ResampleOptions.PilBicubic);
        MinPixels = minPixels;
        MaxPixels = maxPixels;
        BatchBudget = batchBudget;
    }

    /// <summary>The released checkpoint's settings: patch 16, merge 2, temporal patch 1, mean/std 0.5.</summary>
    public static MossVlImageProcessor Default { get; } =
        new(new VisionPatchSpec(PatchSize: 16, MergeSize: 2, TemporalPatchSize: 1), [0.5f, 0.5f, 0.5f], [0.5f, 0.5f, 0.5f]);

    /// <summary>The smallest area an image is resized to.</summary>
    public long MinPixels { get; }

    /// <summary>The largest area one image is resized to.</summary>
    public long MaxPixels { get; }

    /// <summary>The area one batch of images shares.</summary>
    public long BatchBudget { get; }

    /// <summary>The patch geometry the processor produces.</summary>
    public VisionPatchSpec Spec => _pre.Spec;

    /// <summary>Reads a checkpoint's <c>preprocessor_config.json</c> (or the video one, whose budget key is
    /// <c>video_max_pixels</c>). Missing fields keep the released checkpoint's values.</summary>
    public static MossVlImageProcessor FromJson(JsonElement root)
    {
        int Int(string name, int fallback) => root.TryGetProperty(name, out JsonElement v) && v.TryGetInt32(out int i) ? i : fallback;
        long Long(JsonElement e, string name, long fallback) => e.TryGetProperty(name, out JsonElement v) && v.TryGetInt64(out long l) ? l : fallback;
        float[] Triple(string name) => root.TryGetProperty(name, out JsonElement v) && v.ValueKind == JsonValueKind.Array
            ? v.EnumerateArray().Select(x => x.GetSingle()).ToArray()
            : [0.5f, 0.5f, 0.5f];

        long min = DefaultMinPixels, max = DefaultMaxPixels;
        if (root.TryGetProperty("size", out JsonElement size))
        {
            min = Long(size, "shortest_edge", min);
            max = Long(size, "longest_edge", max);
        }
        // Qwen2-VL's processor lets top-level min_pixels / max_pixels override size.
        min = Long(root, "min_pixels", min);
        max = Long(root, "max_pixels", max);
        long budget = Long(root, "multi_image_max_pixels", Long(root, "video_max_pixels", DefaultBatchBudget));
        VisionPatchSpec spec = new(Int("patch_size", 16), Int("merge_size", 2), Int("temporal_patch_size", 1));
        return new MossVlImageProcessor(spec, Triple("image_mean"), Triple("image_std"), min, max, budget);
    }

    /// <summary>Reads <paramref name="path"/> with <see cref="FromJson"/>.</summary>
    public static MossVlImageProcessor FromFile(string path)
    {
        using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(path));
        return FromJson(doc.RootElement);
    }

    /// <summary>The resized (height, width) of one image under <paramref name="maxPixels"/>.</summary>
    public (int Height, int Width) TargetSize(int height, int width, long maxPixels) =>
        SmartResize.Qwen2Vl(height, width, Spec.Factor, MinPixels, maxPixels);

    /// <summary>Preprocesses one image on its own: its budget is the whole batch budget, capped at <see cref="MaxPixels"/>.</summary>
    public VisionPatchGrid Preprocess(RawRgbFrame image) => PreprocessBatch([image])[0];

    /// <summary>Preprocesses the images of one request together: each gets its share of <see cref="BatchBudget"/>, in proportion to
    /// its area, clamped to [<see cref="MinPixels"/>, <see cref="MaxPixels"/>]. Results are in input order; the caller owns them.</summary>
    public VisionPatchGrid[] PreprocessBatch(IReadOnlyList<RawRgbFrame> images)
    {
        ArgumentNullException.ThrowIfNull(images);
        (int, int)[] sizes = images.Select(i => (i.Height, i.Width)).ToArray();
        long[] shares = SmartResize.ShareBudget(sizes, BatchBudget, MinPixels, MaxPixels);
        VisionPatchGrid[] grids = new VisionPatchGrid[images.Count];
        try
        {
            for (int i = 0; i < images.Count; i++)
            {
                (int h, int w) = TargetSize(images[i].Height, images[i].Width, shares[i]);
                grids[i] = _pre.Preprocess(images[i], h, w);
            }
            return grids;
        }
        catch
        {
            foreach (VisionPatchGrid? g in grids) g?.Dispose();
            throw;
        }
    }

    /// <summary>Preprocesses one image smart-resized under an explicit pixel cap (a still image; a video frame sized by
    /// <see cref="VideoFrameSampling.FrameSize"/> must use <see cref="PreprocessAt"/>: smart-resize is not idempotent, so resizing
    /// that size's area again can land one merge block smaller).</summary>
    public VisionPatchGrid Preprocess(RawRgbFrame image, long maxPixels)
    {
        (int h, int w) = TargetSize(image.Height, image.Width, maxPixels);
        return _pre.Preprocess(image, h, w);
    }

    /// <summary>Preprocesses one image resized to exactly (<paramref name="targetHeight"/>, <paramref name="targetWidth"/>), which must be
    /// multiples of the patch block.</summary>
    public VisionPatchGrid PreprocessAt(RawRgbFrame image, int targetHeight, int targetWidth) => _pre.Preprocess(image, targetHeight, targetWidth);
}
