using System.Globalization;
using HartsyInference.Engine.Features;

namespace HartsyInference.Engine.Recipes.Video;

/// <summary>The source geometry of LongCat-Video's video continuation, as the pinned reference computes it (meituan-longcat/LongCat-Video @ 58eb336):
/// the demo's frame stride to 15 fps (<c>run_demo_video_continuation.py</c> lines 89-92, 96), the 480p aspect bucket the pipeline generates at
/// (<c>pipeline_longcat_video.py</c> lines 352-366 and 907-910, <c>longcat_video/utils/bukcet_config.py</c> ASPECT_RATIO_627), and the
/// source resize to that bucket (<c>pipeline_longcat_video.py</c> line 996, diffusers 0.35.1 <c>image_processor.py</c> <c>resize</c>, PIL lanczos).</summary>
internal static class LongCatContinuation
{
    /// <summary>The demo's target frame rate (run_demo_video_continuation.py line 89); the output is written at this rate (line 116).</summary>
    internal const int TargetFps = 15;

    /// <summary>The reference's <c>ASPECT_RATIO_627</c> buckets in its listed order, each key with its <c>[height, width]</c>.</summary>
    private static readonly (string Key, int Height, int Width)[] Buckets =
    [
        ("0.26", 320, 1216), ("0.31", 352, 1120), ("0.38", 384, 1024), ("0.43", 416, 960), ("0.52", 448, 864), ("0.58", 480, 832),
        ("0.67", 512, 768), ("0.74", 544, 736), ("0.86", 576, 672), ("0.95", 608, 640), ("1.05", 640, 608), ("1.17", 672, 576),
        ("1.29", 704, 544), ("1.35", 736, 544), ("1.50", 768, 512), ("1.67", 800, 480), ("1.73", 832, 480), ("2.00", 896, 448),
        ("2.31", 960, 416), ("2.58", 992, 384), ("2.75", 1056, 384), ("3.09", 1088, 352), ("3.70", 1184, 320), ("3.80", 1216, 320),
        ("3.90", 1248, 320), ("4.00", 1280, 320),
    ];

    /// <summary>The frame stride from the source's frame rate to <see cref="TargetFps"/>: <c>max(1, round(fps / 15))</c>, with Python's
    /// round-half-to-even (<c>run_demo_video_continuation.py</c> lines 91-92). An unknown rate (0) gives a stride of 1.</summary>
    internal static int Stride(double sourceFps)
    {
        if (!(sourceFps > 0)) return 1;
        return Math.Max(1, (int)Math.Round(sourceFps / TargetFps, MidpointRounding.ToEven));
    }

    /// <summary>The generation size for a source of <paramref name="width"/>×<paramref name="height"/>: the bucket whose key is closest to
    /// <c>height / width</c>, the first such key on a tie (Python's stable <c>sorted(...)[0]</c>, <c>pipeline_longcat_video.py</c> lines 362-365).
    /// Returns <c>(height, width)</c>.</summary>
    internal static (int Height, int Width) Bucket(int width, int height)
    {
        if (width <= 0 || height <= 0) throw new ArgumentOutOfRangeException(nameof(width), $"The source must have a size; got {width}x{height}.");
        double ratio = (double)height / width;
        double bestDistance = double.PositiveInfinity;
        (int Height, int Width) best = (0, 0);
        foreach ((string key, int h, int w) in Buckets)
        {
            double distance = Math.Abs(double.Parse(key, CultureInfo.InvariantCulture) - ratio);
            if (distance < bestDistance)
            {
                bestDistance = distance;
                best = (h, w);
            }
        }
        return best;
    }

    /// <summary>The source as the continuation uses it: the strided frames at the source size (the output's first part), and the last
    /// <paramref name="contextFrames"/> of them resized to the generation size (the frames the VAE encodes).</summary>
    internal sealed record Source(byte[][] Frames, byte[][] ContextFrames, int GenerationHeight, int GenerationWidth, int Width, int Height);

    /// <summary>Strides a decoded clip to the target rate and resizes its context to the generation size. <paramref name="decoded"/> is RGB24 at
    /// <paramref name="width"/>×<paramref name="height"/>. Only the context frames are resized: the reference resizes the whole clip
    /// (<c>pipeline_longcat_video.py</c> line 996), but prepare_latents encodes only its last 13 (line 269), so the rest is never read.</summary>
    internal static Source Prepare(byte[][] decoded, int width, int height, double sourceFps, int contextFrames)
    {
        ArgumentNullException.ThrowIfNull(decoded);
        int stride = Stride(sourceFps);
        List<byte[]> strided = [];
        for (int i = 0; i < decoded.Length; i += stride) strided.Add(decoded[i]);
        if (strided.Count < contextFrames)
            throw new ArgumentException($"A continuation needs at least {contextFrames} source frames after the {stride}-frame stride; the clip has {strided.Count}.");
        (int genHeight, int genWidth) = Bucket(width, height);
        byte[][] context = new byte[contextFrames][];
        for (int i = 0; i < contextFrames; i++)
        {
            byte[] frame = strided[strided.Count - contextFrames + i];
            context[i] = FeatureImaging.ResizeRgb24Lanczos(new HartsyInference.Engine.Requests.ImageData { Rgb = frame, Width = width, Height = height }, genWidth, genHeight);
        }
        return new Source([.. strided], context, genHeight, genWidth, width, height);
    }
}
