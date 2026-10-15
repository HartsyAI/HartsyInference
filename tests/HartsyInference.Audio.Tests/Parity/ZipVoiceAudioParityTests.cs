using System.Security.Cryptography;
using System.Text.Json;
using HartsyInference.Audio.Dsp;
using Xunit;

namespace HartsyInference.Audio.Tests.Parity;

/// <summary>ZipVoice's audio post-processing against upstream (fixture from tools/parity/zipvoice_frontend_dump.py): the 0.1 s
/// cross-fade of <c>cross_fade_concat</c> and pydub's <c>remove_silence</c> on synthetic signals built from the same specs,
/// where a spec is blocks of 10 ms (S silent, L loud, T and U just under and over the -50 dBFS threshold, V a mix whose
/// float RMS is over it and integer RMS under it) or raw loud samples (R).</summary>
public sealed class ZipVoiceAudioParityTests
{
    private const int SampleRate = 24000;
    private const int Block = SampleRate / 100;

    [Fact]
    public void CrossFade_MatchesUpstream()
    {
        foreach (JsonElement entry in ZipVoiceFixture.Root.GetProperty("crossfade").EnumerateArray())
        {
            int sr = entry.GetProperty("sr").GetInt32();
            List<float[]> waves = new();
            int index = 0;
            foreach (JsonElement length in entry.GetProperty("lens").EnumerateArray()) waves.Add(Wave(index++, length.GetInt32()));
            float[] got = WaveCrossFade.Join(waves, (int)(entry.GetProperty("fade").GetDouble() * sr));
            float[] want = entry.GetProperty("out").EnumerateArray().Select(e => (float)e.GetDouble()).ToArray();
            string name = entry.GetProperty("name").GetString()!;
            Assert.True(got.Length == want.Length, $"{name}: want {want.Length} samples, got {got.Length}");
            for (int i = 0; i < want.Length; i++) Assert.True(Math.Abs(got[i] - want[i]) < 1e-6f, $"{name}[{i}]: want {want[i]}, got {got[i]}");
        }
    }

    [Fact]
    public void RemoveSilence_MatchesPydubOnSyntheticSignals()
    {
        foreach (JsonElement entry in ZipVoiceFixture.Root.GetProperty("silence").EnumerateArray())
        {
            string name = entry.GetProperty("name").GetString()!;
            short[] pcm = Signal(entry.GetProperty("spec").GetString()!);
            Assert.Equal(entry.GetProperty("in_len").GetInt32(), pcm.Length);
            float[] input = pcm.Select(s => s / 32768f).ToArray();

            float[] got = PydubSilence.RemoveSilence(input, SampleRate, entry.GetProperty("only_edge").GetBoolean(), entry.GetProperty("trail").GetInt32());

            Assert.True(got.Length == entry.GetProperty("out_len").GetInt32(), $"{name}: want {entry.GetProperty("out_len").GetInt32()} samples, got {got.Length}");
            byte[] bytes = new byte[got.Length * 2];
            for (int i = 0; i < got.Length; i++) BitConverter.TryWriteBytes(bytes.AsSpan(i * 2), (short)MathF.Round(got[i] * 32768f));
            Assert.Equal(entry.GetProperty("sha1").GetString(), Convert.ToHexString(SHA1.HashData(bytes)).ToLowerInvariant());
        }
    }

    [Theory]
    [InlineData(0, 0)]
    [InlineData(100, 2397)]
    [InlineData(200, 4798)]
    [InlineData(333, 7990)]
    [InlineData(1000, 23998)]
    public void TrailingSilence_IsPydubsElevenKilohertzSilenceResampledTo24k(int ms, int expected) =>
        Assert.Equal(expected, PydubSilence.ResampledSilenceFrames(ms, SampleRate));

    private static float[] Wave(int index, int length)
    {
        float[] wave = new float[length];
        for (int i = 0; i < length; i++) wave[i] = (float)((i * 37 + index * 101) % 211) / 211f - 0.5f;
        return wave;
    }

    private static short[] Signal(string spec)
    {
        List<short> samples = new();
        long n = 0;
        foreach (string token in spec.Split(' '))
        {
            char kind = token[0];
            int count = int.Parse(token[1..]);
            int size = kind == 'R' ? count : count * Block;
            for (int k = 0; k < size; k++, n++)
            {
                long v = kind switch
                {
                    'L' or 'R' => n * 7919 % 2001 + 2000,
                    'S' => n * 131 % 61 - 30,
                    'T' => 103,
                    'U' => 104,
                    'V' => k % Block < 192 ? 104 : 103,
                    _ => throw new ArgumentException($"unknown block '{kind}'"),
                };
                samples.Add((short)v);
            }
        }
        return [.. samples];
    }
}
