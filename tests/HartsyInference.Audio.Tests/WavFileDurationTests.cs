using HartsyInference.Audio.Io;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>The header-only duration read of <see cref="WavFile.TryReadDurationSeconds"/>.</summary>
public sealed class WavFileDurationTests
{
    [Theory]
    [InlineData(24_000, 1.0)]
    [InlineData(24_000, 21.5)]
    [InlineData(44_100, 3.0)]
    public void ASizedWav_GivesItsDuration(int rate, double seconds)
    {
        byte[] wav = Wav(rate, (int)(seconds * rate));
        Assert.True(WavFile.TryReadDurationSeconds(wav, out double got));
        Assert.Equal(seconds, got, 3);
    }

    [Fact]
    public void AChunkBeforeData_IsSkipped()
    {
        byte[] plain = Wav(16_000, 32_000);
        byte[] list = [.. "LIST"u8.ToArray(), 4, 0, 0, 0, .. "INFO"u8.ToArray()];
        byte[] wav = [.. plain[..36], .. list, .. plain[36..]];
        Assert.True(WavFile.TryReadDurationSeconds(wav, out double seconds));
        Assert.Equal(2.0, seconds, 3);
    }

    [Fact]
    public void ATruncatedData_ReportsTheDeclaredLength()
    {
        byte[] wav = Wav(24_000, 48_000);
        byte[] truncated = wav[..(44 + 24_000 * 2)];
        Assert.True(WavFile.TryReadDurationSeconds(truncated, out double seconds));
        Assert.Equal(2.0, seconds, 3);
        Assert.Equal(48_000, WavFile.Read(new MemoryStream(truncated)).Length);
    }

    [Fact]
    public void AnythingTheHeaderCannotSize_IsRefused()
    {
        byte[] streaming = Wav(24_000, 24_000);
        BitConverter.GetBytes(uint.MaxValue).CopyTo(streaming, 40);
        Assert.False(WavFile.TryReadDurationSeconds(streaming, out _));
        Assert.False(WavFile.TryReadDurationSeconds([], out _));
        Assert.False(WavFile.TryReadDurationSeconds("RIFF....WAVE"u8, out _));
        Assert.False(WavFile.TryReadDurationSeconds("not a wav file at all"u8, out _));
        Assert.False(WavFile.TryReadDurationSeconds(Wav(24_000, 24_000)[..30], out _));
    }

    private static byte[] Wav(int rate, int frames)
    {
        using MemoryStream stream = new();
        WavFile.WriteMono16(stream, new float[frames], rate);
        return stream.ToArray();
    }
}
