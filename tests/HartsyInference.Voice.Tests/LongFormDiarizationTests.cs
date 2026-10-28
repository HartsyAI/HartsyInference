using HartsyInference.Engine;
using HartsyInference.Engine.Audio;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Features;
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
        // The CAM++ checkpoint is gated like the other weights: SpeakerDiarizer never downloads it and throws when it is absent.
        if (!RealWeightGate.Require(_output.WriteLine,
            [string.IsNullOrEmpty(overridePath) ? VoiceAssets.Jfk : overridePath, CamPlusCheckpoint(), .. VoiceAssets.WhisperFiles(WhisperTinyTurnSttTests.WhisperTiny)]))
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

    /// <summary>The checkpoint <c>SpeakerDiarizer.Load</c> would pick: the same four names under the same folders, in the
    /// same order. When none is placed, the conventional default path, so the gate's SKIPPED line names what is missing.
    /// SpeakerDiarizer is internal to the Engine, so its name list is repeated here.</summary>
    private static string CamPlusCheckpoint()
    {
        string[] names = ["campplus_cn_common.bin", "campplus_cn_common.safetensors", "campplus.safetensors", "s3gen.safetensors"];
        foreach (string name in names)
        {
            string? found = ModelFileLocator.Find(name, Path.Combine("audio", "speaker"), "audio");
            if (found is not null)
            {
                return found;
            }
        }
        return Path.Combine(RepoPaths.ModelsRoot(), "audio", "speaker", names[0]);
    }
}
