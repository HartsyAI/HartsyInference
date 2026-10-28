using HartsyInference.Engine;
using HartsyInference.Engine.Audio;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Requests;
using HartsyInference.Tests.Common;
using HartsyInference.Voice.Tests.Fakes;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Voice.Tests;

/// <summary>Speaker diarization on a clip longer than one Whisper window: spans must cover the whole clip, not only its
/// first window. Uses the clip named by HARTSY_LONGFORM_CLIP when set, else three JFK repeats across 4 s of silence.</summary>
[Trait("Category", "Integration")]
[Trait("Category", "RealWeights")]
public sealed class LongFormDiarizationTests
{
    private const int SampleRate = 16_000;

    private readonly ITestOutputHelper _output;

    public LongFormDiarizationTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public async Task DiarizedSpansCoverTheWholeLongClip()
    {
        if (!RealWeightGate.Require(_output.WriteLine,
            [VoiceAssets.Jfk, .. VoiceAssets.WhisperFiles(WhisperTinyTurnSttTests.WhisperTiny)]))
        {
            return;
        }
        string? override_ = Environment.GetEnvironmentVariable("HARTSY_LONGFORM_CLIP");
        float[] clip;
        if (!string.IsNullOrEmpty(override_))
        {
            clip = AudioClipCodec.DecodeMono(new AudioClip { Data = File.ReadAllBytes(override_), Format = "wav" }, SampleRate);
        }
        else
        {
            float[] jfk = VoiceAssets.Jfk16k();
            float[] gap = new float[4 * SampleRate];
            clip = [.. jfk, .. gap, .. jfk, .. gap, .. jfk];
        }
        double seconds = clip.Length / (double)SampleRate;
        ModelSpec tiny = ModelResolver.Resolve("whisper:" + WhisperTinyTurnSttTests.WhisperTiny, null, Modality.Transcribe);

        using InferenceEngine engine = new("cpu");
        TranscriptResult result = await engine.Transcribe.RunAsync(tiny, new AudioRequest
        {
            Audio = new AudioClip { Data = AudioClipCodec.EncodeWav(clip, null, SampleRate), Format = "wav" },
            Diarization = true,
        });

        IReadOnlyList<DiarizedSegment> speakers = result.Speakers ?? [];
        double covered = speakers.Count == 0 ? 0 : speakers.Max(s => s.End);
        _output.WriteLine($"{seconds:0.0} s, {result.WindowCount} windows; {speakers.Select(s => s.Speaker).Distinct().Count()} speaker(s), "
            + $"{speakers.Count} span(s), spans reach {covered:0.0} s");
        Assert.NotNull(result.Speakers);
        Assert.NotEmpty(speakers);
        Assert.True(covered >= seconds - 3.0, $"spans stop at {covered:0.0} s of a {seconds:0.0} s clip");
    }
}
