using System.Text.Json;
using HartsyInference.Cli.Dispatch;
using HartsyInference.Cli.Infra;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using Xunit;

namespace HartsyInference.Cli.Tests;

/// <summary>The CLI's structured vision output: each array lands as a <c>.npy</c> whose payload is the API's wire bytes, maps
/// get PNG previews, <c>result.json</c> carries the non-array fields, and a frame run indexes every frame's exact time.</summary>
public sealed class VisionOutputWriterTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "hartsy-vision-writer-" + Path.GetRandomFileName());

    public VisionOutputWriterTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        if (Directory.Exists(_dir))
            Directory.Delete(_dir, recursive: true);
    }

    [Fact]
    public void Result_WritesNpyPreviewsAndASidecar()
    {
        VisionArray depth = new VisionArray([1f, 2f, 3f, 4f], 2, 2);
        VisionArray normals = new VisionArray(Enumerable.Repeat(0f, 12).ToArray(), 2, 2, 3);
        VisionResult result = new VisionResult
        {
            DepthMap = new VisionDepthMap { Values = depth, Kind = VisionDepthKind.MetricMeters },
            Normals = normals,
            Camera = new VisionCamera { Fx = 10, Fy = 11, Cx = 1, Cy = 1, Width = 2, Height = 2 },
            Poses = [new VisionPose { Box = new VisionBox(0, 0, 2, 2), Score = 0.9f, Keypoints = [1, 1, 0.8f], Schema = "test-1" }],
        };

        IReadOnlyList<string> files = VisionOutputWriter.WriteResult(result, _dir);

        Assert.Equal(VisionArrayCodec.ToLittleEndianBytes(depth), NpyPayload(Path.Combine(_dir, "depth.npy")));
        Assert.Equal(depth.Data, VisionArrayCodec.FromNpy(File.ReadAllBytes(Path.Combine(_dir, "depth.npy"))).Data);
        Assert.True(File.Exists(Path.Combine(_dir, "depth.png")));
        Assert.True(File.Exists(Path.Combine(_dir, "normals.npy")));
        Assert.True(File.Exists(Path.Combine(_dir, "normals.png")));
        Assert.Equal("result.json", Path.GetFileName(files[^1]));

        using JsonDocument sidecar = JsonDocument.Parse(File.ReadAllText(Path.Combine(_dir, "result.json")));
        Assert.Equal("metricMeters", sidecar.RootElement.GetProperty("depthKind").GetString());
        Assert.Equal(10, sidecar.RootElement.GetProperty("camera").GetProperty("fx").GetDouble());
        Assert.Equal("test-1", sidecar.RootElement.GetProperty("poses")[0].GetProperty("schema").GetString());
        Assert.Equal(0.5, sidecar.RootElement.GetProperty("conventions").GetProperty("pixelCenter").GetDouble());
    }

    [Fact]
    public void Result_ViewsArePrefixedPerView()
    {
        VisionResult result = new VisionResult
        {
            Views = [Depth(1f), Depth(2f)],
        };

        VisionOutputWriter.WriteResult(result, _dir);

        Assert.Equal([1f], VisionArrayCodec.FromNpy(File.ReadAllBytes(Path.Combine(_dir, "view-00.depth.npy"))).Data);
        Assert.Equal([2f], VisionArrayCodec.FromNpy(File.ReadAllBytes(Path.Combine(_dir, "view-01.depth.npy"))).Data);
    }

    [Fact]
    public async Task Frames_IndexEachFramesTimeAndSource()
    {
        int written = await VisionOutputWriter.WriteFramesAsync(Frames(), _dir, CancellationToken.None);

        Assert.Equal(3, written);
        Assert.True(File.Exists(Path.Combine(_dir, "frame-00001.image.png")));
        using JsonDocument index = JsonDocument.Parse(File.ReadAllText(Path.Combine(_dir, "frames.json")));
        JsonElement middle = index.RootElement[1];
        Assert.Equal(1.0 / 48, middle.GetProperty("timestampSeconds").GetDouble());
        Assert.False(middle.TryGetProperty("sourceIndex", out _));
        Assert.Equal(1, index.RootElement[2].GetProperty("sourceIndex").GetInt32());
    }

    [Theory]
    [InlineData("metric-depth", VisionMode.MetricDepth)]
    [InlineData("geometry", VisionMode.Geometry)]
    [InlineData("pose", VisionMode.Pose)]
    [InlineData("matting", VisionMode.Matting)]
    [InlineData("face", VisionMode.Face)]
    [InlineData("intrinsic", VisionMode.Intrinsic)]
    [InlineData("track", VisionMode.Track)]
    [InlineData("INTERPOLATE", VisionMode.Interpolate)]
    [InlineData("removebg", VisionMode.BackgroundRemoval)]
    public void ModeNames_MapToTheEngineModes(string name, VisionMode mode)
    {
        Assert.Equal(mode, GenerationDispatch.ParseVisionMode(name));
    }

    [Fact]
    public void BatchFolder_TakesOnlyTheFormatsTheDecoderReads()
    {
        foreach (string name in new[] { "b.png", "a.bmp", "c.jpg", "d.jpeg", "notes.txt", "e.PNG" })
            File.WriteAllBytes(Path.Combine(_dir, name), [0]);

        string[] inputs = GenerationDispatch.VisionBatchInputs(_dir).Select(Path.GetFileName).ToArray()!;

        Assert.Equal(["a.bmp", "b.png", "e.PNG"], inputs);
    }

    [Fact]
    public void BatchFolder_WithNoReadableImagesIsRefused()
    {
        File.WriteAllBytes(Path.Combine(_dir, "only.jpg"), [0]);

        ArgumentException ex = Assert.Throws<ArgumentException>(() => GenerationDispatch.VisionBatchInputs(_dir));
        Assert.Contains(".png or .bmp", ex.Message);
    }

    private static VisionResult Depth(float value) =>
        new VisionResult { DepthMap = new VisionDepthMap { Values = new VisionArray([value], 1, 1), Kind = VisionDepthKind.Relative } };

    private static async IAsyncEnumerable<VisionFrameResult> Frames()
    {
        ImageData frame = new ImageData { Rgb = new byte[2 * 2 * 3], Width = 2, Height = 2 };
        yield return new VisionFrameResult { Index = 0, TimestampSeconds = 0, SourceIndex = 0, Result = new VisionResult { Image = frame } };
        yield return new VisionFrameResult { Index = 1, TimestampSeconds = 1.0 / 48, Result = new VisionResult { Image = frame } };
        yield return new VisionFrameResult { Index = 2, TimestampSeconds = 1.0 / 24, SourceIndex = 1, Result = new VisionResult { Image = frame } };
        await Task.CompletedTask;
    }

    private static byte[] NpyPayload(string path)
    {
        byte[] file = File.ReadAllBytes(path);
        return file[(10 + BitConverter.ToUInt16(file, 8))..];
    }
}
