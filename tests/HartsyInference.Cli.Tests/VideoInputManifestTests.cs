using HartsyInference.Cli.Infra;
using HartsyInference.Engine;
using HartsyInference.Engine.Requests;
using Xunit;

namespace HartsyInference.Cli.Tests;

public sealed class VideoInputManifestTests
{
    [Fact]
    public void GuideObjectManifestResolvesRelativePathsAndMergesSignedFramePayloads()
    {
        using TestDirectory files = new();
        files.WriteBytes("guide.mp4", [1, 2, 3]);
        files.WriteBytes("guide.wav", [4, 5]);
        string manifest = files.WriteText("guides.json",
            """{"guides":[{"frame":-1,"video":"guide.mp4","audio":"guide.wav","fit":"contain"}]}""");
        ParamState parameters = VideoParameters();
        parameters.Put("guides-manifest", manifest);

        VideoGuide guide = Assert.Single(VideoInputManifest.Guides(parameters)!);

        Assert.Equal(-1, guide.FrameIndex);
        Assert.Equal(VideoGuideFitMode.Contain, guide.FitMode);
        Assert.Equal([1, 2, 3], guide.Video!.Data);
        Assert.Equal([4, 5], guide.Audio!.Data);
    }

    [Fact]
    public void GuideBareArrayManifestAndRepeatableFlagsAreBothAccepted()
    {
        using TestDirectory files = new();
        files.WriteBytes("manifest.mp4", [8]);
        files.WriteBytes("flag.wav", [9]);
        string manifest = files.WriteText("guides.json",
            """[{"frame":4,"video":"manifest.mp4","fit":"stretch"}]""");
        ParamState parameters = VideoParameters();
        parameters.Put("guides-manifest", manifest);
        parameters.Put("guide-audios", $"-2={files.PathOf("flag.wav")}");

        IReadOnlyList<VideoGuide> guides = VideoInputManifest.Guides(parameters)!;

        Assert.Equal([-2, 4], guides.Select(guide => guide.FrameIndex));
        Assert.Equal(VideoGuideFitMode.Stretch, guides[1].FitMode);
    }

    [Fact]
    public void DuplicateGuidePayloadsFailBeforeMediaDecode()
    {
        ParamState parameters = VideoParameters();
        parameters.Put("guide-videos", "3=first.mp4\n3=second.mp4");

        ArgumentException error = Assert.Throws<ArgumentException>(() => VideoInputManifest.Guides(parameters));

        Assert.Contains("duplicate video", error.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void ControlManifestResolvesEveryRelativePathAndStringEnum()
    {
        using TestDirectory files = new();
        files.WriteBytes("control.safetensors", [1]);
        files.WriteBytes("control.mp4", [2]);
        files.WriteBytes("visible.mp4", [3]);
        files.WriteBytes("source.mp4", [4]);
        string manifest = files.WriteText("controls.json", """
            {"controls":[{"model":"control.safetensors","video":"control.mp4","kind":"inpaint",
              "strength":0.75,"start":0.2,"end":0.8,"visibilityMask":"visible.mp4","maskedSource":"source.mp4"}]}
            """);
        ParamState parameters = VideoParameters();
        parameters.Put("controls-manifest", manifest);

        VideoControl control = Assert.Single(VideoInputManifest.Controls(parameters)!);

        Assert.Equal(files.PathOf("control.safetensors"), control.Model);
        Assert.Equal(VideoControlKind.Inpaint, control.Kind);
        Assert.Equal(0.75, control.Strength);
        Assert.Equal(0.2, control.Start);
        Assert.Equal(0.8, control.End);
        Assert.Equal([2], control.Video.Data);
        Assert.Equal([3], control.VisibilityMask!.Data);
        Assert.Equal([4], control.MaskedSource!.Data);
    }

    [Fact]
    public void SimpleInpaintRequiresTheManifestPayloadContract()
    {
        ParamState parameters = VideoParameters();
        parameters.Put("control-model", "control.safetensors");
        parameters.Put("control-video", "control.mp4");
        parameters.Put("control-kind", "inpaint");

        ArgumentException error = Assert.Throws<ArgumentException>(() => VideoInputManifest.Controls(parameters));

        Assert.Contains("--controls-manifest", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AudioMaskParsesDelimitedValuesRateAndSource()
    {
        using TestDirectory files = new();
        string mask = files.WriteText("mask.txt", "0, 0.5 1");
        files.WriteBytes("source.wav", [7, 8]);
        ParamState parameters = VideoParameters();
        parameters.Put("audio-denoise-mask", mask);
        parameters.Put("audio-mask-source", files.PathOf("source.wav"));
        parameters.Put("audio-mask-rate", "20");

        AudioDenoiseMask parsed = VideoInputManifest.AudioMask(parameters)!;

        Assert.Equal([0f, 0.5f, 1f], parsed.Values);
        Assert.Equal(20f, parsed.Rate);
        Assert.Equal([7, 8], parsed.Source!.Data);
    }

    private static ParamState VideoParameters() => new(Modality.Video);

    private sealed class TestDirectory : IDisposable
    {
        private readonly string _path = System.IO.Path.Combine(System.IO.Path.GetTempPath(),
            $"hartsy-cli-tests-{Guid.NewGuid():N}");

        internal TestDirectory() => Directory.CreateDirectory(_path);

        internal string PathOf(string name) => System.IO.Path.GetFullPath(System.IO.Path.Combine(_path, name));

        internal void WriteBytes(string name, byte[] value) => File.WriteAllBytes(PathOf(name), value);

        internal string WriteText(string name, string value)
        {
            string path = PathOf(name);
            File.WriteAllText(path, value);
            return path;
        }

        public void Dispose() => Directory.Delete(_path, recursive: true);
    }
}
