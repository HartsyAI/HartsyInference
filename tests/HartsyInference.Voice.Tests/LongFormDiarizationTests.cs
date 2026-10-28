using HartsyInference.Engine;
using HartsyInference.Engine.Audio;
using HartsyInference.Engine.Audio.Wake.Speakers;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Requests;
using HartsyInference.Tests.Common;
using HartsyInference.Voice.Tests.Fakes;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Voice.Tests;

/// <summary>Speaker diarization on a clip longer than one Whisper window: spans must cover the whole clip, not only its
/// first window. Uses the clip named by HARTSY_LONGFORM_CLIP when set, else three JFK repeats across 4 s of silence.
/// A clip named there must be WAV, longer than one window, and end in speech: spans have to reach within 3 s of its end.
/// Needs the CAM++ speaker checkpoint as well as Whisper tiny.</summary>
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
        string? overridePath = Environment.GetEnvironmentVariable("HARTSY_LONGFORM_CLIP");
        string clipPath = string.IsNullOrEmpty(overridePath) ? VoiceAssets.Jfk : overridePath;
        // The CAM++ checkpoint is gated like the other weights: SpeakerDiarizer never downloads it and throws when it is absent.
        if (!RealWeightGate.Require(_output.WriteLine,
            [clipPath, CamPlusCheckpoint(), .. VoiceAssets.WhisperFiles(WhisperTinyTurnSttTests.WhisperTiny)]))
        {
            return;
        }
        float[] clip;
        if (!string.IsNullOrEmpty(overridePath))
        {
            clip = AudioClipCodec.DecodeMono(new AudioClip { Data = File.ReadAllBytes(overridePath), Format = "wav" }, SampleRate);
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
        Assert.True(result.WindowCount >= 2, $"a {seconds:0.0} s clip must be planned into more than one window; got {result.WindowCount}");
        Assert.NotNull(result.Speakers);
        Assert.NotEmpty(speakers);
        Assert.True(covered >= seconds - 3.0, $"spans stop at {covered:0.0} s of a {seconds:0.0} s clip");
    }

    /// <summary>The checkpoint SpeakerDiarizer loads (CamPlusEmbedder.LocateWeights searches the same names and folders), or
    /// the folder its error message names when none is placed, so the gate's SKIPPED line says where to put it.</summary>
    private static string CamPlusCheckpoint() =>
        CamPlusEmbedder.LocateWeights()
        ?? Path.Combine(RepoPaths.ModelsRoot(), "audio", "speaker", "campplus", "campplus_cn_common.bin");
}
