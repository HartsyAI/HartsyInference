using HartsyInference.Core.Numerics;

namespace HartsyInference.Audio.Models.ControlFoley;

/// <summary>Port of the official <c>load_video</c> / <c>extract_video_segments</c>: samples the decoded frames at the CLIP,
/// CAV-MAE and Synchformer rates by presentation time, resizes (and centre-crops) them with torch's uint8 bicubic kernel and
/// normalises, then truncates every stream to the shortest one.</summary>
public static class ControlFoleyVideoPreprocessor
{
    private static readonly float[] ZeroMean = [0f, 0f, 0f];
    private static readonly float[] OneStd = [1f, 1f, 1f];
    private static readonly float[] ImageNetMean = [0.4850f, 0.4560f, 0.4060f];
    private static readonly float[] ImageNetStd = [0.2290f, 0.2240f, 0.2250f];
    private static readonly float[] HalfMean = [0.5f, 0.5f, 0.5f];

    /// <summary>Prepares up to <paramref name="durationSeconds"/> of <paramref name="video"/>; shorter clips shrink
    /// <see cref="ControlFoleyVideoInput.TotalDuration"/>.</summary>
    public static ControlFoleyVideoInput Prepare(ControlFoleyRawVideo video, double durationSeconds, ControlFoleyVideoOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(video);
        options ??= ControlFoleyVideoOptions.Default;
        if (durationSeconds <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(durationSeconds), "The duration must be positive.");
        }

        if (video.Frames.Count != video.Times.Count || video.Frames.Count == 0 || video.Width <= 0 || video.Height <= 0)
        {
            throw new ArgumentException("The video needs one timestamp per frame and at least one frame.", nameof(video));
        }

        foreach (byte[] frame in video.Frames)
        {
            if (frame.Length != video.Width * video.Height * 3)
            {
                throw new ArgumentException("Every frame must be Width x Height x 3 bytes.", nameof(video));
            }
        }

        double[] fps = [options.ClipFps, options.VisualFps, options.SyncFps];
        List<int>[] selected = [[], [], []];
        double[] next = [0.0, 0.0, 0.0];
        for (int i = 0; i < video.Frames.Count; i++)
        {
            double time = video.Times[i];
            if (time < 0 || time > durationSeconds)
            {
                continue;
            }

            for (int s = 0; s < 3; s++)
            {
                while (time >= next[s])
                {
                    selected[s].Add(i);
                    next[s] += 1.0 / fps[s];
                }
            }
        }

        if (selected[0].Count == 0 || selected[1].Count == 0 || selected[2].Count == 0)
        {
            throw new InvalidDataException("The video has no frames inside the requested duration.");
        }

        double duration = durationSeconds;
        for (int s = 0; s < 3; s++)
        {
            duration = Math.Min(duration, selected[s].Count / fps[s]);
        }

        int[] counts = [(int)(fps[0] * duration), (int)(fps[1] * duration), (int)(fps[2] * duration)];
        if (counts[0] == 0 || counts[1] == 0 || counts[2] == 0)
        {
            throw new InvalidDataException($"A usable duration of {duration:0.###} s leaves no frames in one stream.");
        }

        float[] clip = Transform(video, selected[0], counts[0], options.ClipSize, options.ClipSize, crop: false, ZeroMean, OneStd);
        float[] visual = Transform(video, selected[1], counts[1], options.VisualSize, options.VisualSize, crop: true, ImageNetMean, ImageNetStd);
        float[] sync = Transform(video, selected[2], counts[2], options.SyncSize, options.SyncSize, crop: true, HalfMean, [0.5f, 0.5f, 0.5f]);
        return new ControlFoleyVideoInput
        {
            TotalDuration = duration, Clip = clip, ClipFrames = counts[0], Visual = visual, VisualFrames = counts[1], Sync = sync,
            SyncFrames = counts[2],
        };
    }

    private static float[] Transform(ControlFoleyRawVideo video, List<int> selected, int count, int outHeight, int outWidth, bool crop,
        float[] mean, float[] std)
    {
        int h = video.Height, w = video.Width, rh = outHeight, rw = outWidth;
        if (crop)
        {
            int shortSide = Math.Min(h, w);
            rh = h <= w ? outHeight : (int)((long)outHeight * h / shortSide);
            rw = h <= w ? (int)((long)outWidth * w / shortSide) : outWidth;
            if (shortSide == outHeight)
            {
                rh = h;
                rw = w;
            }
        }

        int plane = outHeight * outWidth, perFrame = 3 * plane;
        float[] result = new float[(long)count * perFrame];
        Dictionary<int, int> firstUse = [];
        int[] source = new int[count];
        for (int f = 0; f < count; f++)
        {
            source[f] = selected[f];
            firstUse.TryAdd(selected[f], f);
        }

        int[] unique = [.. firstUse.Values];
        CpuParallel.For(unique.Length, (long)unique.Length * h * w * 12, u =>
        {
            int f = unique[u];
            byte[] resized = rh == h && rw == w ? video.Frames[source[f]] : ControlFoleyBicubicResize.Resize(video.Frames[source[f]], h, w, rh, rw);
            int top = crop ? (int)Math.Round((rh - outHeight) / 2.0) : 0, left = crop ? (int)Math.Round((rw - outWidth) / 2.0) : 0;
            Span<float> dst = result.AsSpan(f * perFrame, perFrame);
            for (int y = 0; y < outHeight; y++)
            {
                for (int x = 0; x < outWidth; x++)
                {
                    int px = ((y + top) * rw + x + left) * 3;
                    for (int ch = 0; ch < 3; ch++)
                    {
                        dst[ch * plane + y * outWidth + x] = (resized[px + ch] / 255f - mean[ch]) / std[ch];
                    }
                }
            }
        });

        for (int f = 0; f < count; f++)
        {
            int first = firstUse[source[f]];
            if (first != f)
            {
                result.AsSpan(first * perFrame, perFrame).CopyTo(result.AsSpan(f * perFrame, perFrame));
            }
        }

        return result;
    }
}
