using HartsyInference.Diffusion.Utilities;

namespace HartsyInference.Video.Pipelines;

/// <summary>SeedVR2 input conditioning chain (reference <c>inference_seedvr2_3b.py</c> video_transform +
/// cut_videos). The bicubic area-resize is part of the MODEL CONTRACT, not hygiene: SeedVR2 restores detail
/// at the target resolution — the upscaling itself is this plain resize, and the window partition normalizes
/// token budgets with the same area formula. Skipping or mis-scaling it moves the token grid outside the
/// training regime and silently degrades output.</summary>
public static class SeedVr2Preprocess
{
    /// <summary>Preprocessed clip: fp32 CTHW in [-1,1], frame count padded to the causal-VAE contract.</summary>
    /// <param name="Data">C×T×H×W, C=3.</param>
    /// <param name="Frames">Padded frame count T ((T−1) % 4 == 0, or 1 for stills).</param>
    /// <param name="OriginalFrames">Input frame count before padding — trim the output back to this.</param>
    public sealed record Result(float[] Data, int Frames, int Height, int Width, int OriginalFrames);

    /// <summary>Runs resize → clamp → divisible-crop → normalize → layout → pad on RGB24 frames.</summary>
    /// <param name="frames">Input frames, each H×W×3 interleaved bytes (engine RGB24 convention).</param>
    /// <param name="targetArea">res_h*res_w token budget; reference default 1280*720. Aspect is preserved —
    /// this is an AREA target, not an output size.</param>
    public static Result Run(IReadOnlyList<byte[]> frames, int width, int height, long targetArea)
    {
        if (frames.Count == 0)
            throw new ArgumentException("At least one frame is required.");
        if (width <= 0 || height <= 0)
            throw new ArgumentException($"Frame dims must be positive, got {width}x{height}.");
        long expected = (long)width * height * 3;
        for (int i = 0; i < frames.Count; i++)
            if (frames[i].Length != expected)
                throw new ArgumentException($"Frame {i} is {frames[i].Length} bytes, expected {expected}.");

        // AreaResize: scale = sqrt(max_area / (h*w)), Python round() = banker's. downsample_only=False —
        // small inputs are UPSCALED ("model only trained for high res").
        double scale = Math.Sqrt(targetArea / (height * (double)width));
        int resizedH = (int)Math.Round(height * scale, MidpointRounding.ToEven);
        int resizedW = (int)Math.Round(width * scale, MidpointRounding.ToEven);

        // DivisibleCrop(16): center crop, top/left offset uses Python round() (banker's on the .5 halves).
        int cropH = resizedH - resizedH % 16;
        int cropW = resizedW - resizedW % 16;
        int top = (int)Math.Round((resizedH - cropH) / 2.0, MidpointRounding.ToEven);
        int left = (int)Math.Round((resizedW - cropW) / 2.0, MidpointRounding.ToEven);

        int t = frames.Count;
        int paddedT = PaddedFrameCount(t);
        float[] output = new float[3L * paddedT * cropH * cropW];

        float[] srcPlanes = new float[3 * height * width];
        float[] resized = new float[3 * resizedH * resizedW];
        for (int f = 0; f < t; f++)
        {
            byte[] rgb = frames[f];
            for (int y = 0; y < height; y++)
            {
                int rowBase = y * width;
                for (int x = 0; x < width; x++)
                {
                    int px = (rowBase + x) * 3;
                    int planeIdx = rowBase + x;
                    srcPlanes[planeIdx] = rgb[px] * (1f / 255f);
                    srcPlanes[height * width + planeIdx] = rgb[px + 1] * (1f / 255f);
                    srcPlanes[2 * height * width + planeIdx] = rgb[px + 2] * (1f / 255f);
                }
            }

            TorchResize.BicubicAntialiasChw(srcPlanes, 3, height, width, resized, resizedH, resizedW);

            for (int c = 0; c < 3; c++)
            {
                long outPlane = ((long)c * paddedT + f) * cropH * cropW;
                int inPlane = c * resizedH * resizedW;
                for (int y = 0; y < cropH; y++)
                {
                    int srcRow = inPlane + (top + y) * resizedW + left;
                    long dstRow = outPlane + (long)y * cropW;
                    for (int x = 0; x < cropW; x++)
                    {
                        // clamp [0,1] then Normalize(0.5, 0.5) -> [-1,1].
                        float v = Math.Clamp(resized[srcRow + x], 0f, 1f);
                        output[dstRow + x] = (v - 0.5f) * 2f;
                    }
                }
            }
        }

        // cut_videos: repeat the last real frame until (T-1) % 4 == 0 (stills exempt). Trim after decode.
        for (int f = t; f < paddedT; f++)
            for (int c = 0; c < 3; c++)
            {
                long src = ((long)c * paddedT + (t - 1)) * cropH * cropW;
                long dst = ((long)c * paddedT + f) * cropH * cropW;
                Array.Copy(output, src, output, dst, (long)cropH * cropW);
            }

        return new Result(output, paddedT, cropH, cropW, t);
    }

    /// <summary>Causal-VAE frame contract: 1 for stills, otherwise the next count with (T−1) % 4 == 0.</summary>
    public static int PaddedFrameCount(int t)
    {
        if (t <= 0)
            throw new ArgumentException($"Frame count must be positive, got {t}.");
        if (t == 1)
            return 1;
        if (t <= 4)
            return 5;
        return (t - 1) % 4 == 0 ? t : t + 4 - (t - 1) % 4;
    }
}
