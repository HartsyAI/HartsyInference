using System.Diagnostics;
using HartsyInference.Video.Encoding;
using Xunit;

namespace HartsyInference.Video.Tests;

public sealed class FfmpegProcessDecoderTests
{
    [Fact]
    [Trait("Category", "Integration")]
    public async Task DecodeAsync_MaxSeconds_StopsDecodingAfterThatMuchInput()
    {
        string clip = Path.Combine(Path.GetTempPath(), $"hartsy-ffmpeg-{Guid.NewGuid():N}.mp4");
        try
        {
            if (!TryRun("ffmpeg", $"-v error -f lavfi -i testsrc=size=64x48:rate=10 -t 3 -pix_fmt yuv420p \"{clip}\""))
            {
                return;
            }

            FfmpegProcessDecoder decoder = new();
            FfmpegProcessDecoder.Result full = await decoder.DecodeFileAsync(clip);
            FfmpegProcessDecoder.Result capped = await decoder.DecodeAsync(
                await File.ReadAllBytesAsync(clip), "mp4", maxFrames: null, scaleWidth: null, scaleHeight: null, maxSeconds: 1.0);

            Assert.Equal(30, full.Frames.Count);
            Assert.Equal(10, capped.Frames.Count);
            Assert.Equal(64, capped.Width);
            Assert.Equal(48, capped.Height);
            Assert.Equal(10.0, capped.Fps, 3);
        }
        finally
        {
            File.Delete(clip);
        }
    }

    private static bool TryRun(string binary, string arguments)
    {
        try
        {
            using Process? process = Process.Start(new ProcessStartInfo(binary, arguments) { UseShellExecute = false, RedirectStandardError = true });
            process?.WaitForExit();
            return process is { ExitCode: 0 };
        }
        catch (System.ComponentModel.Win32Exception)
        {
            return false;
        }
    }
}
