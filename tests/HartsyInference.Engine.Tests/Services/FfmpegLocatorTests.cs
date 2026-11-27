using HartsyInference.Video.Encoding;

namespace HartsyInference.Engine.Tests.Services;

/// <summary>Where the engine looks for ffmpeg and ffprobe: the <c>paths.ffmpegDir</c> folder first, then PATH.</summary>
public sealed class FfmpegLocatorTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "hartsy-ffmpeg-" + Guid.NewGuid().ToString("N"));

    public FfmpegLocatorTests() => Directory.CreateDirectory(_dir);

    public void Dispose() => Directory.Delete(_dir, recursive: true);

    private static string Executable(string tool) => OperatingSystem.IsWindows() ? tool + ".exe" : tool;

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("   ")]
    public void NoFolderMeansThePathLookup(string? folder) => Assert.Equal("ffmpeg", FfmpegLocator.Resolve("ffmpeg", folder));

    [Fact]
    public void AToolInTheFolderIsUsedByFullPath()
    {
        string ffmpeg = Path.Combine(_dir, Executable("ffmpeg"));
        File.WriteAllBytes(ffmpeg, [0]);
        Assert.Equal(ffmpeg, FfmpegLocator.Resolve("ffmpeg", _dir));
        // ffprobe is not there, so it still falls back to PATH rather than naming a file that does not exist.
        Assert.Equal("ffprobe", FfmpegLocator.Resolve("ffprobe", _dir));
    }

    [Fact]
    public void AFolderWithoutTheToolFallsBackToPath() =>
        Assert.Equal("ffmpeg", FfmpegLocator.Resolve("ffmpeg", Path.Combine(_dir, "does-not-exist")));
}
