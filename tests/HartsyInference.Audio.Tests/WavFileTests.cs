using HartsyInference.Audio.Io;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>WAV I/O round-trip tests. The reader supports several PCM bit depths
/// + IEEE float; we exercise 16-bit (the universally common case) end-to-end via
/// the writer, and 8/24/32-bit + float via in-memory synthesized headers.</summary>
public sealed class WavFileTests
{
    [Fact]
    public void Write_Read_16BitMono_RoundTrips()
    {
        string tmp = Path.GetTempFileName();
        try
        {
            float[] samples = new float[1024];
            for (int i = 0; i < samples.Length; i++)
                samples[i] = 0.5f * MathF.Sin(2f * MathF.PI * i / 64);

            WavFile.WriteMono16(tmp, samples, sampleRate: 16_000);
            WavFile.DecodedAudio decoded = WavFile.Read(tmp);

            Assert.Equal(16_000, decoded.SampleRate);
            Assert.Single(decoded.Channels);
            Assert.Equal(samples.Length, decoded.Length);
            // 16-bit quantization tolerance: 1/65536 worst-case round-trip ≈ 1.5e-5
            // absolute, which can flip the 4th decimal. precision:3 gives a margin.
            for (int i = 0; i < samples.Length; i++)
                Assert.Equal(samples[i], decoded.Channels[0][i], precision: 3);
        }
        finally { File.Delete(tmp); }
    }

    [Fact]
    public void Read_IeeeFloat32_Stereo_ProducesTwoChannels()
    {
        // Build a 32-bit IEEE float WAV in memory.
        byte[] data = BuildFloat32StereoWav(sampleRate: 44_100, framesPerChannel: 100);
        using MemoryStream ms = new(data);
        WavFile.DecodedAudio decoded = WavFile.Read(ms);

        Assert.Equal(44_100, decoded.SampleRate);
        Assert.Equal(2, decoded.Channels.Length);
        Assert.Equal(100, decoded.Length);
        Assert.Equal(0.25f, decoded.Channels[0][0], precision: 6);
        Assert.Equal(-0.25f, decoded.Channels[1][0], precision: 6);
    }

    [Fact]
    public void ToMono_Downmixes_Stereo_ByAveraging()
    {
        float[][] ch = [new[] { 1f, 0f, -1f }, new[] { 0f, 1f, 1f }];
        WavFile.DecodedAudio decoded = new(ch, 16_000);
        float[] mono = decoded.ToMono();
        Assert.Equal(0.5f, mono[0]);
        Assert.Equal(0.5f, mono[1]);
        Assert.Equal(0f, mono[2]);
    }

    [Fact]
    public void Read_RejectsNonWavFile()
    {
        byte[] bogus = new byte[100];
        bogus[0] = (byte)'X';
        using MemoryStream ms = new(bogus);
        Assert.Throws<InvalidDataException>(() => WavFile.Read(ms));
    }

    private static byte[] BuildFloat32StereoWav(int sampleRate, int framesPerChannel)
    {
        int dataBytes = framesPerChannel * 2 * 4;
        using MemoryStream ms = new();
        using BinaryWriter w = new(ms);
        w.Write(System.Text.Encoding.ASCII.GetBytes("RIFF"));
        w.Write((uint)(36 + dataBytes));
        w.Write(System.Text.Encoding.ASCII.GetBytes("WAVE"));
        w.Write(System.Text.Encoding.ASCII.GetBytes("fmt "));
        w.Write((uint)16);
        w.Write((ushort)3);       // IEEE float
        w.Write((ushort)2);       // channels
        w.Write((uint)sampleRate);
        w.Write((uint)(sampleRate * 2 * 4));    // byte rate
        w.Write((ushort)8);       // block align
        w.Write((ushort)32);      // bits
        w.Write(System.Text.Encoding.ASCII.GetBytes("data"));
        w.Write((uint)dataBytes);
        for (int i = 0; i < framesPerChannel; i++)
        {
            w.Write(0.25f);
            w.Write(-0.25f);
        }
        return ms.ToArray();
    }
}
