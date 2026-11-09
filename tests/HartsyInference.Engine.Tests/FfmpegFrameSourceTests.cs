using System.Diagnostics;
using HartsyInference.Core.Imaging;
using HartsyInference.Video.Encoding;

namespace HartsyInference.Engine.Tests;

/// <summary>Skips a test when <c>ffmpeg</c>/<c>ffprobe</c> are not on PATH; a skip is reported, never counted as a pass.</summary>
public sealed class FfmpegFactAttribute : FactAttribute
{
    public FfmpegFactAttribute()
    {
        if (!OnPath("ffmpeg") || !OnPath("ffprobe"))
            Skip = "SKIPPED: ffmpeg/ffprobe are not on PATH; the video frame source did not run.";
    }

    private static bool OnPath(string tool)
    {
        try
        {
            using Process p = Process.Start(new ProcessStartInfo(tool, "-version") { RedirectStandardOutput = true, RedirectStandardError = true, UseShellExecute = false })!;
            p.WaitForExit(10_000);
            return p.ExitCode == 0;
        }
        catch
        {
            return false;
        }
    }
}

/// <summary>The ffmpeg frame source on a clip ffmpeg itself generates: the sampled indices and timestamps follow the plan, and
/// frames come at source size.</summary>
public sealed class FfmpegFrameSourceTests
{
    [FfmpegFact]
    public async Task SamplesOneFramePerSecondAtSourceSize()
    {
        string clip = Path.Combine(Path.GetTempPath(), $"hartsy-testsrc-{Guid.NewGuid():N}.mp4");
        using (Process gen = Process.Start(new ProcessStartInfo("ffmpeg",
                   $"-v error -f lavfi -i testsrc=duration=5:size=160x120:rate=10 -pix_fmt yuv420p \"{clip}\"") { UseShellExecute = false })!)
            await gen.WaitForExitAsync();
        try
        {
            using FfmpegFrameSource source = new(clip, new VideoFrameSamplingOptions(Fps: 1.0));
            List<TimedRgbFrame> frames = [];
            await foreach (TimedRgbFrame frame in source.ReadAsync()) frames.Add(frame);

            // 50 frames at 10 fps, 1 fps sampling: ceil(5) = 5 frames at int(linspace(0, 49, 5)) = 0, 12, 24, 36, 49.
            Assert.Equal([0.0, 1.2, 2.4, 3.6, 4.9], frames.Select(f => Math.Round(f.TimestampSeconds, 6)).ToArray());
            Assert.All(frames, f => Assert.Equal((160, 120, 160 * 120 * 3), (f.Frame.Width, f.Frame.Height, f.Frame.Rgb.Length)));
            Assert.NotEqual(frames[0].Frame.Rgb.ToArray(), frames[^1].Frame.Rgb.ToArray());
        }
        finally
        {
            File.Delete(clip);
        }
    }
}
