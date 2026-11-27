using HartsyInference.Cli.Dispatch;
using HartsyInference.Cli.Infra;
using HartsyInference.Engine;
using HartsyInference.Engine.Requests;
using Xunit;

namespace HartsyInference.Cli.Tests;

/// <summary>The CLI's speaker options as the engine's request.</summary>
public sealed class LongCatSpeakerOptionTests : IDisposable
{
    private readonly string _wav = Path.Combine(Path.GetTempPath(), "longcat-speaker-" + Guid.NewGuid().ToString("N") + ".wav");

    public LongCatSpeakerOptionTests() => File.WriteAllBytes(_wav, [1, 2, 3]);

    public void Dispose() => File.Delete(_wav);

    private static ParamState Parameters(params (string Key, string Value)[] values)
    {
        ParamState parameters = new(Modality.Video);
        foreach ((string key, string value) in values)
            parameters.Put(key, value);
        return parameters;
    }

    [Fact]
    public void NoSpeakerOptionMeansNoSpeakers() =>
        Assert.Null(GenerationDispatch.BuildSpeakers(Parameters(("driving-audio", _wav))));

    [Fact]
    public void EveryOptionReachesTheRequest()
    {
        VideoSpeakers speakers = GenerationDispatch.BuildSpeakers(Parameters(("speaker1-audio", _wav), ("speaker1-box", "10, 20,300,400"),
            ("speaker2-box", "1,500,600,700"), ("speaker-timing", "Add"), ("silent-boxes", "0,0,5,5\n6,7,8,9")))!;

        Assert.Equal([1, 2, 3], speakers.First.Audio!.Data);
        Assert.Null(speakers.Second.Audio);
        Assert.Equal(new VideoSpeakerBox { YMin = 10, XMin = 20, YMax = 300, XMax = 400 }, speakers.First.Box);
        Assert.Equal(new VideoSpeakerBox { YMin = 1, XMin = 500, YMax = 600, XMax = 700 }, speakers.Second.Box);
        Assert.Equal(VideoSpeakerTiming.Sequential, speakers.Timing);
        Assert.Equal([new VideoSpeakerBox { YMin = 0, XMin = 0, YMax = 5, XMax = 5 },
            new VideoSpeakerBox { YMin = 6, XMin = 7, YMax = 8, XMax = 9 }], speakers.SilentBoxes);
    }

    [Theory]
    [InlineData("parallel", VideoSpeakerTiming.Parallel)]
    [InlineData("para", VideoSpeakerTiming.Parallel)]
    [InlineData("sequential", VideoSpeakerTiming.Sequential)]
    public void TimingTakesEitherName(string text, VideoSpeakerTiming expected) =>
        Assert.Equal(expected, GenerationDispatch.BuildSpeakers(Parameters(("speaker2-audio", _wav), ("speaker-timing", text)))!.Timing);

    [Theory]
    [InlineData("speaker-timing", "together")]
    [InlineData("speaker1-box", "1,2,3")]
    [InlineData("speaker1-box", "1,2,3,x")]
    [InlineData("silent-boxes", "1;2;3;4")]
    public void BadOptionTextIsRefused(string key, string value) =>
        Assert.Throws<ArgumentException>(() => GenerationDispatch.BuildSpeakers(Parameters((key, value))));

    [Fact]
    public void AMissingAudioFileIsRefused() =>
        Assert.Throws<FileNotFoundException>(() => GenerationDispatch.BuildSpeakers(Parameters(("speaker1-audio", _wav + ".missing"))));
}
