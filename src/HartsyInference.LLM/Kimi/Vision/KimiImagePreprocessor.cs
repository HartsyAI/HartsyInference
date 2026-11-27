using System.Text.Json;
using HartsyInference.Core.Imaging;
using HartsyInference.Core.Tensors;

namespace HartsyInference.LLM.Kimi.Vision;

/// <summary>The image geometry of a Kimi K3.1 checkpoint's <c>media_proc_cfg</c> (<c>preprocessor_config.json</c>): the patch and merge sizes, the
/// patch budget and side limit the resize reads, and the normalization.</summary>
public sealed record KimiImageSpec
{
    /// <summary>Square patch edge in pixels.</summary>
    public required int PatchSize { get; init; }

    /// <summary>Square merge kernel: each vision token covers <c>MergeKernelSize²</c> patches.</summary>
    public required int MergeKernelSize { get; init; }

    /// <summary>Most patches an image may hold before it is shrunk.</summary>
    public required int InPatchLimit { get; init; }

    /// <summary>Most patches along one side before it is shrunk.</summary>
    public required int PatchLimitOnOneSide { get; init; }

    /// <summary>Per-channel mean, applied after rescaling to [0, 1].</summary>
    public required float[] ImageMean { get; init; }

    /// <summary>Per-channel standard deviation.</summary>
    public required float[] ImageStd { get; init; }

    /// <summary>Reads a <c>media_proc_cfg</c> object.</summary>
    /// <exception cref="InvalidDataException">A field is missing or malformed.</exception>
    /// <exception cref="NotSupportedException"><c>fixed_output_tokens</c> is set, so the token count no longer follows the grid.</exception>
    public static KimiImageSpec Read(JsonElement cfg)
    {
        if (cfg.TryGetProperty("fixed_output_tokens", out JsonElement fixedTokens) && fixedTokens.ValueKind != JsonValueKind.Null)
            throw new NotSupportedException("fixed_output_tokens is set, so the token count does not follow the image grid.");
        return new KimiImageSpec
        {
            PatchSize = Int(cfg, "patch_size"),
            MergeKernelSize = Int(cfg, "merge_kernel_size"),
            InPatchLimit = Int(cfg, "in_patch_limit"),
            PatchLimitOnOneSide = Int(cfg, "patch_limit_on_one_side"),
            ImageMean = Channels(cfg, "image_mean"),
            ImageStd = Channels(cfg, "image_std"),
        };
    }

    private static int Int(JsonElement cfg, string name)
    {
        if (!cfg.TryGetProperty(name, out JsonElement value) || !value.TryGetInt32(out int number))
            throw new InvalidDataException($"media_proc_cfg has no integer '{name}'.");
        return number;
    }

    private static float[] Channels(JsonElement cfg, string name)
    {
        if (!cfg.TryGetProperty(name, out JsonElement value) || value.ValueKind != JsonValueKind.Array || value.GetArrayLength() != 3)
            throw new InvalidDataException($"media_proc_cfg has no three-value '{name}'.");
        float[] values = new float[3];
        int i = 0;
        foreach (JsonElement item in value.EnumerateArray()) values[i++] = (float)item.GetDouble();
        return values;
    }
}

/// <summary>One image's resize: the size it is resized to, the zero pad that rounds it to whole merge blocks, and the vision tokens it becomes.</summary>
public readonly record struct KimiImageResize(int NewWidth, int NewHeight, int PadWidth, int PadHeight, int Tokens)
{
    /// <summary>The padded width, a whole number of merge blocks.</summary>
    public int Width => NewWidth + PadWidth;

    /// <summary>The padded height, a whole number of merge blocks.</summary>
    public int Height => NewHeight + PadHeight;
}

/// <summary>Turns an RGB image into the MoonViT-V2 tower's input the way the Kimi K3.1 processor does: shrink to the patch budget, resize with PIL
/// bicubic, pad with black to whole merge blocks, and normalize.</summary>
/// <remarks>Ports <c>navit_resize_image</c> and <c>image_to_np</c> of the checkpoint's <c>media_utils.py</c>. The transparent-background fill is not
/// ported: the engine's images carry no alpha, and the route refuses an image that does. The resize is verified against the PIL kernel on
/// synthetic images only.</remarks>
public static class KimiImagePreprocessor
{
    /// <summary>Works out the resize of a <paramref name="width"/>×<paramref name="height"/> image.</summary>
    /// <exception cref="NotSupportedException">The side limit is not a whole number of merge blocks, so the padded image would overrun it.</exception>
    public static KimiImageResize Resize(int width, int height, KimiImageSpec spec)
    {
        ArgumentNullException.ThrowIfNull(spec);
        if (width < 1 || height < 1) throw new ArgumentOutOfRangeException(nameof(width), $"An image needs positive sides, got {width}x{height}.");
        int patch = spec.PatchSize, side = spec.PatchLimitOnOneSide;
        double patchBudget = spec.InPatchLimit / (Math.Max(1, width / patch) * (double)Math.Max(1, height / patch));
        double scale = Math.Min(1.0, Math.Min(Math.Sqrt(patchBudget),
            Math.Min((double)side * patch / width, (double)side * patch / height)));
        int newWidth = Math.Min(Math.Max(1, (int)(width * scale)), side * patch);
        int newHeight = Math.Min(Math.Max(1, (int)(height * scale)), side * patch);

        int factor = spec.MergeKernelSize * patch;
        int padHeight = (factor - newHeight % factor) % factor;
        int padWidth = (factor - newWidth % factor) % factor;
        int tokenHeight = (newHeight + padHeight) / factor, tokenWidth = (newWidth + padWidth) / factor;
        // The clamp keeps the resized side within the limit, but the pad rounds it up to a whole merge block, so a limit that is not a multiple
        // of the merge size can be overrun.
        if (tokenHeight * spec.MergeKernelSize > side || tokenWidth * spec.MergeKernelSize > side)
        {
            throw new NotSupportedException($"The {width}x{height} image pads past the {side}-patch side limit, which is not a whole number of "
                + $"{spec.MergeKernelSize}-patch merge blocks.");
        }
        return new KimiImageResize(newWidth, newHeight, padWidth, padHeight, tokenHeight * tokenWidth);
    }

    /// <summary>Preprocesses interleaved RGB (<c>width·height·3</c> bytes, row-major) into a normalized <c>[1, 3, H, W]</c> F32 tensor, with H and W the
    /// padded size. The caller owns the tensor.</summary>
    public static Tensor Preprocess(ReadOnlySpan<byte> rgb, int width, int height, KimiImageSpec spec)
    {
        ArgumentNullException.ThrowIfNull(spec);
        if (rgb.Length < (long)width * height * 3)
            throw new ArgumentException($"rgb has {rgb.Length} bytes, expected {(long)width * height * 3} for {width}x{height}.", nameof(rgb));
        KimiImageResize resize = Resize(width, height, spec);
        float[] resized = new float[resize.NewWidth * resize.NewHeight * 3];
        Resample.BicubicHwc8(rgb, width, height, 3, resized, resize.NewWidth, resize.NewHeight, -0.5f, antialias: true, quantizeU8: true);

        int h = resize.Height, w = resize.Width, plane = h * w;
        Tensor pixels = new(new TensorShape(1, 3, h, w), DType.F32);
        Span<float> dst = pixels.AsSpan<float>();
        for (int c = 0; c < 3; c++)
        {
            float mean = spec.ImageMean[c], invStd = 1f / spec.ImageStd[c];
            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    // The pad is raw black before normalization, so it lands at -mean/std, as the processor's own pad does.
                    float v = y < resize.NewHeight && x < resize.NewWidth ? resized[(y * resize.NewWidth + x) * 3 + c] : 0f;
                    dst[c * plane + y * w + x] = (v / 255f - mean) * invStd;
                }
            }
        }
        return pixels;
    }
}
