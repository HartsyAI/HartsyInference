using System.Runtime.InteropServices;
using System.Text.Json;
using HartsyInference.Audio.Models.LongCat;
using Xunit;

namespace HartsyInference.Audio.Tests.Parity;

/// <summary>The two speakers' audio layout against the reference's <c>audio_prepare_multi</c>, sample for sample (the dump of
/// tests/python-reference/longcat_avatar/dump_longcat_avatar_multi.py: upstream pull request 162's version). Skipped, and so not evidence, until
/// HARTSY_LONGCAT_AVATAR_MULTI_DIR names the dump.</summary>
[Trait("Category", "Integration")]
public sealed class LongCatSpeakerAudioParityTests
{
    [LongCatSpeakerDumpFact]
    public void BothTimingsMatchTheReferenceSampleForSample()
    {
        string dir = Environment.GetEnvironmentVariable(LongCatSpeakerDumpFactAttribute.Variable)!;
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(Path.Combine(dir, "meta.json")));
        float[] Read(string name) => [.. MemoryMarshal.Cast<byte, float>(File.ReadAllBytes(Path.Combine(dir, name + ".f32")))];
        int index = 0;
        foreach (JsonElement item in document.RootElement.GetProperty("audio_cases").EnumerateArray())
        {
            LongCatSpeakerSpeech? Speaker(int s) => item.GetProperty("present")[s].GetBoolean()
                ? new LongCatSpeakerSpeech(Read($"audio.{index}.vocals{s}"), Read($"audio.{index}.raw{s}"))
                : null;
            LongCatSpeakerTiming timing = item.GetProperty("mode").GetString() == "add"
                ? LongCatSpeakerTiming.Sequential
                : LongCatSpeakerTiming.Parallel;

            LongCatSpeakerAudio audio = LongCatDrivingAudio.Combine(Speaker(0), Speaker(1), timing, item.GetProperty("duration").GetDouble());

            Assert.Equal(Read($"audio.{index}.track0"), audio.Tracks[0]);
            Assert.Equal(Read($"audio.{index}.track1"), audio.Tracks[1]);
            Assert.Equal(Read($"audio.{index}.mix"), audio.Soundtrack);
            index++;
        }
        Assert.True(index >= 9, "The dump should hold every timing and absent-speaker case.");
    }

    /// <summary>A fact skipped, never passed, when the multi-speaker dump is absent.</summary>
    private sealed class LongCatSpeakerDumpFactAttribute : FactAttribute
    {
        public const string Variable = "HARTSY_LONGCAT_AVATAR_MULTI_DIR";

        public LongCatSpeakerDumpFactAttribute()
        {
            string? dir = Environment.GetEnvironmentVariable(Variable);
            if (string.IsNullOrWhiteSpace(dir) || !File.Exists(Path.Combine(dir, "meta.json")))
                Skip = $"SKIPPED: {Variable} is unset or has no meta.json; the speaker audio reference comparison did not run.";
        }
    }
}
