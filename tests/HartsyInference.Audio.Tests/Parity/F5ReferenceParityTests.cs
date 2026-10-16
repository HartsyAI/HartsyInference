using System.Security.Cryptography;
using System.Text.Json;
using HartsyInference.Audio.Dsp;
using HartsyInference.Audio.Models.F5Tts;
using Xunit;

namespace HartsyInference.Audio.Tests.Parity;

/// <summary>F5-TTS's reference preparation against upstream <c>preprocess_ref_audio_text</c> (fixture from
/// tools/parity/f5_reference_dump.py). The processed clip must match upstream's 16-bit samples exactly (length and
/// SHA-1), and the RMS that scales the reference and the output must match to float precision.</summary>
public sealed class F5ReferenceParityTests
{
    private const int SampleRate = 24000;
    private const int Block = SampleRate / 100;

    [Fact]
    public void Prepare_MatchesUpstreamSamplesAndRms()
    {
        JsonElement root = F5Fixture();
        int checkedCases = 0;
        foreach (JsonElement entry in root.GetProperty("cases").EnumerateArray())
        {
            string name = entry.GetProperty("name").GetString()!;
            string spec = entry.GetProperty("spec").GetString()!;
            float[] input = spec.StartsWith("file:", StringComparison.Ordinal)
                ? ReadWav24k(Path.Combine(AppContext.BaseDirectory, "Fixtures", spec["file:".Length..]))
                : Signal(spec).Select(s => s / 32768f).ToArray();
            Assert.Equal(entry.GetProperty("in_len").GetInt32(), input.Length);

            float[] processed = F5ReferenceAudio.Prepare(input, SampleRate);
            short[] pcm = PydubSilence.Quantize(processed);
            int rate = entry.GetProperty("rate").GetInt32();
            double rms = F5ReferenceAudio.Rms(processed);
            Assert.True(Math.Abs(rms - entry.GetProperty("rms").GetDouble()) < 1e-6, $"{name}: rms {rms}");

            if (rate != SampleRate)
            {
                // A clip with no speech left: pydub's empty segment is 11.025 kHz, so upstream's 50 ms of silence is
                // 551 samples at that rate. The engine works at 24 kHz, where the same 50 ms is the resampled count.
                Assert.Equal(PydubSilence.ResampledSilenceFrames(50, rate), entry.GetProperty("out_len").GetInt32());
                Assert.Equal(PydubSilence.ResampledSilenceFrames(50, SampleRate), pcm.Length);
            }
            else
            {
                Assert.True(pcm.Length == entry.GetProperty("out_len").GetInt32(),
                    $"{name}: want {entry.GetProperty("out_len").GetInt32()} samples, got {pcm.Length}");
                Assert.Equal(entry.GetProperty("out_sha1").GetString(), Sha1(pcm));
            }
            checkedCases++;
        }
        Assert.True(checkedCases >= 9, $"fixture has only {checkedCases} cases");
    }

    [Fact]
    public void TailSilence_IsPydubsFiftyMillisecondsAtTheClipRate()
    {
        float[] silent = new float[SampleRate];
        float[] processed = F5ReferenceAudio.Prepare(silent, SampleRate);
        Assert.Equal(PydubSilence.ResampledSilenceFrames(50, SampleRate), processed.Length);
    }

    private static JsonElement F5Fixture() =>
        JsonDocument.Parse(File.ReadAllText(
            Path.Combine(AppContext.BaseDirectory, "Fixtures", "f5_reference_parity.json"))).RootElement;

    // The synthetic signals of f5_reference_dump.py: S, Q, L, T, U and V blocks of 10 ms, with g the sample index.
    private static short[] Signal(string spec)
    {
        List<short> samples = new();
        long total = 0;
        string[] tokens = spec.Split(' ');
        for (int i = 0; i < tokens.Length; i += 2)
        {
            char kind = tokens[i][0];
            int count = int.Parse(tokens[i + 1]);
            int size = count * Block;
            for (int k = 0; k < size; k++)
            {
                long g = total + k;
                long v = kind switch
                {
                    'L' => g * 7919 % 2001 + 2000,
                    'Q' => g * 7919 % 201 - 100,
                    'S' => g * 131 % 61 - 30,
                    'T' => 103,
                    'U' => 104,
                    'V' => k % Block < 192 ? 104 : 103,
                    _ => throw new ArgumentException($"unknown block '{kind}'"),
                };
                samples.Add((short)v);
            }
            total += size;
        }
        return [.. samples];
    }

    // A 16-bit mono 24 kHz WAV written by Python's wave module (a 44-byte header).
    private static float[] ReadWav24k(string path)
    {
        byte[] raw = File.ReadAllBytes(path);
        int n = (raw.Length - 44) / 2;
        float[] audio = new float[n];
        for (int i = 0; i < n; i++) audio[i] = BitConverter.ToInt16(raw, 44 + i * 2) / 32768f;
        return audio;
    }

    private static string Sha1(short[] pcm)
    {
        byte[] bytes = new byte[pcm.Length * 2];
        Buffer.BlockCopy(pcm, 0, bytes, 0, bytes.Length);
        return Convert.ToHexString(SHA1.HashData(bytes)).ToLowerInvariant();
    }
}
