using Xunit;
using HartsyInference.Engine.Audio;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Covers the generated-audio return path: <see cref="AudioBuffer"/> channel math and the
/// <see cref="VideoAudioResolver"/> precedence that decides which track ships with a generation.</summary>
public class VideoAudioContractTests
{
    private static float[] Ramp(int n, float scale = 1f)
    {
        float[] data = new float[n];
        for (int i = 0; i < n; i++)
        {
            data[i] = i * scale;
        }
        return data;
    }

    private static AudioBuffer Stereo(int frames, int sampleRate = 48_000) =>
        new AudioBuffer { Channels = [Ramp(frames), Ramp(frames, 2f)], SampleRate = sampleRate };

    private static VideoGenerationResult Frames(int count) =>
        VideoGenerationResult.FromFrames(
            [.. Enumerable.Range(0, count).Select(i => new VideoFrame { Index = i, Width = 2, Height = 2, Rgb = new byte[12] })]);

    [Fact]
    public void EmptyBufferReportsNoAudio()
    {
        Assert.True(AudioBuffer.Empty.IsEmpty);
        Assert.Equal(0, AudioBuffer.Empty.FrameCount);
        Assert.Equal(0d, AudioBuffer.Empty.Seconds);
        Assert.True(AudioBuffer.FromChannels(null, 48_000).IsEmpty);
        Assert.True(AudioBuffer.FromChannels([], 48_000).IsEmpty);
        Assert.True(AudioBuffer.FromChannels([[1f]], 0).IsEmpty);
    }

    [Fact]
    public void StereoAndMonoConversionsAreConsistent()
    {
        AudioBuffer stereo = Stereo(4);
        Assert.Equal(2, stereo.ChannelCount);
        (float[] left, float[] right) = stereo.ToStereo();
        Assert.Equal(1f, left[1]);
        Assert.Equal(2f, right[1]);
        // Mono is the average of the two ramps: (i + 2i) / 2.
        Assert.Equal(1.5f, stereo.ToMono()[1]);

        AudioBuffer mono = new AudioBuffer { Channels = [Ramp(4)], SampleRate = 48_000 };
        (float[] monoLeft, float[] monoRight) = mono.ToStereo();
        Assert.Same(monoLeft, monoRight);
        Assert.Same(mono.Channels[0], mono.ToMono());
    }

    [Fact]
    public void SecondsAndTrimUseSampleRate()
    {
        AudioBuffer buffer = Stereo(48_000);
        Assert.Equal(1d, buffer.Seconds);
        AudioBuffer trimmed = buffer.TrimTo(0.5d);
        Assert.Equal(24_000, trimmed.FrameCount);
        Assert.Equal(2, trimmed.ChannelCount);
        // Trimming past the end is a no-op rather than a pad.
        Assert.Same(buffer, buffer.TrimTo(10d));
        Assert.True(buffer.TrimTo(0d).IsEmpty);
    }

    [Fact]
    public void RaggedChannelsClampToTheShortestRatherThanThrowing()
    {
        AudioBuffer ragged = new AudioBuffer { Channels = [Ramp(100), Ramp(60)], SampleRate = 1_000 };
        // Playable length is the shortest channel, so trimming above it is a no-op and below it clamps every channel.
        Assert.Equal(60, ragged.FrameCount);
        Assert.Same(ragged, ragged.TrimTo(0.08d));
        AudioBuffer trimmed = ragged.TrimTo(0.05d);
        Assert.Equal(50, trimmed.Channels[0].Length);
        Assert.Equal(50, trimmed.Channels[1].Length);
        Assert.Equal(60, ragged.ToMono().Length);
        Assert.True(AudioBuffer.FromChannels([[], []], 48_000).IsEmpty);
    }

    [Fact]
    public void GeneratedAudioWinsOverRequestPassThrough()
    {
        AudioBuffer generated = Stereo(48_000);
        VideoGenerationResult result = Frames(24) with { Audio = generated };
        VideoRequest request = new VideoRequest
        {
            Prompt = "x",
            VideoAudioInput = new AudioClip { Data = AudioClipCodec.EncodeWav(Stereo(96_000)), Format = "wav" },
        };
        VideoGenerationResult resolved = VideoAudioResolver.Resolve(result, request, videoSeconds: 1d);
        Assert.NotNull(resolved.Audio);
        Assert.Equal(48_000, resolved.Audio!.FrameCount);
    }

    [Fact]
    public void RequestPassThroughFillsInWhenModelIsSilent()
    {
        VideoRequest request = new VideoRequest
        {
            Prompt = "x",
            VideoAudioInput = new AudioClip { Data = AudioClipCodec.EncodeWav(Stereo(48_000)), Format = "wav" },
        };
        VideoGenerationResult resolved = VideoAudioResolver.Resolve(Frames(24), request, videoSeconds: 1d);
        Assert.NotNull(resolved.Audio);
        Assert.Equal(2, resolved.Audio!.ChannelCount);
    }

    [Fact]
    public void ConditioningReferenceIsNotMuxedByDefault()
    {
        VideoRequest request = new VideoRequest
        {
            Prompt = "x",
            VideoAudioReference = new AudioClip { Data = AudioClipCodec.EncodeWav(Stereo(48_000)), Format = "wav" },
        };
        Assert.Null(VideoAudioResolver.Resolve(Frames(24), request, videoSeconds: 1d).Audio);
    }

    [Fact]
    public void ResolvedTrackIsTrimmedToVideoLength()
    {
        VideoGenerationResult result = Frames(12) with { Audio = Stereo(48_000) };
        VideoRequest request = new VideoRequest { Prompt = "x" };
        VideoGenerationResult resolved = VideoAudioResolver.Resolve(result, request, VideoAudioResolver.VideoSeconds(12, 24));
        Assert.Equal(24_000, resolved.Audio!.FrameCount);
    }

    [Fact]
    public void ShortTrackIsPaddedSoTheMuxerCannotDropVideoFrames()
    {
        // Real LTX-2.3 case: 25 frames @24fps = 1.0417s of video against a 1.010s soundtrack. ffmpeg -shortest
        // cut the 25th frame until the track was padded to match.
        double videoSeconds = VideoAudioResolver.VideoSeconds(25, 24);
        AudioBuffer shortTrack = Stereo(48_480);
        Assert.True(shortTrack.Seconds < videoSeconds);
        VideoGenerationResult resolved = VideoAudioResolver.Resolve(
            Frames(25) with { Audio = shortTrack }, new VideoRequest { Prompt = "x" }, videoSeconds);
        Assert.True(resolved.Audio!.Seconds >= videoSeconds);
        Assert.Equal(2, resolved.Audio.ChannelCount);
        // Padding is silence appended to the tail, not a resample of the original.
        Assert.Equal(shortTrack.Channels[0][100], resolved.Audio.Channels[0][100]);
        Assert.Equal(0f, resolved.Audio.Channels[0][^1]);
    }

    [Fact]
    public void SilentGenerationStaysSilent()
    {
        VideoRequest request = new VideoRequest { Prompt = "x" };
        Assert.Null(VideoAudioResolver.Resolve(Frames(24), request, videoSeconds: 1d).Audio);
    }

    [Fact]
    public void WavRoundTripPreservesRateAndChannels()
    {
        AudioBuffer source = Stereo(1_000, 44_100);
        AudioBuffer decoded = AudioClipCodec.DecodeNative(
            new AudioClip { Data = AudioClipCodec.EncodeWav(source), Format = "wav" });
        Assert.Equal(44_100, decoded.SampleRate);
        Assert.Equal(2, decoded.ChannelCount);
        Assert.Equal(1_000, decoded.FrameCount);
    }
}
