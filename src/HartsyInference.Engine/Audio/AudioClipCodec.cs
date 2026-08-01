using System.Text;
using HartsyInference.Audio.Io;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Logging;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Audio;

/// <summary>Decodes request <see cref="AudioClip"/>s to float PCM and encodes generated waveforms back to WAV. Only
/// RIFF/WAVE is decodable in-engine: the extension shelled out to ffmpeg for compressed containers, which is a host
/// dependency the Engine deliberately does not take, so anything else is refused with a message naming the limit.</summary>
public static class AudioClipCodec
{
    /// <summary>Decodes a clip to mono float PCM in [-1, 1] at <paramref name="targetSampleRate"/>; empty for no input.</summary>
    public static float[] DecodeMono(AudioClip? clip, int targetSampleRate)
    {
        if (clip is null || clip.Data.Length == 0)
        {
            return [];
        }
        WavFile.DecodedAudio decoded = Decode(clip);
        float[] mono = decoded.ToMono();
        return decoded.SampleRate == targetSampleRate
            ? mono
            : Resampler.Create(decoded.SampleRate, targetSampleRate).Resample(mono);
    }

    /// <summary>Decodes a clip to a stereo pair at <paramref name="targetSampleRate"/> (mono sources duplicate the
    /// single channel), for the models that need true stereo input such as Demucs.</summary>
    public static (float[] Left, float[] Right) DecodeStereo(AudioClip? clip, int targetSampleRate)
    {
        if (clip is null || clip.Data.Length == 0)
        {
            return ([], []);
        }
        WavFile.DecodedAudio decoded = Decode(clip);
        float[] left = decoded.Channels.Length > 0 ? decoded.Channels[0] : [];
        float[] right = decoded.Channels.Length > 1 ? decoded.Channels[1] : left;
        if (decoded.SampleRate == targetSampleRate)
        {
            return (left, right);
        }
        // Separate resampler instances: the polyphase filter carries per-stream state.
        float[] outLeft = Resampler.Create(decoded.SampleRate, targetSampleRate).Resample(left);
        float[] outRight = ReferenceEquals(left, right)
            ? outLeft
            : Resampler.Create(decoded.SampleRate, targetSampleRate).Resample(right);
        return (outLeft, outRight);
    }

    /// <summary>Decodes a clip at its native rate and channel count, for pass-through paths that must not resample.</summary>
    public static AudioBuffer DecodeNative(AudioClip? clip)
    {
        if (clip is null || clip.Data.Length == 0)
        {
            return AudioBuffer.Empty;
        }
        WavFile.DecodedAudio decoded = Decode(clip);
        return AudioBuffer.FromChannels(decoded.Channels, decoded.SampleRate);
    }

    /// <summary>Encodes a mono (<paramref name="right"/> null) or stereo waveform as a 16-bit PCM WAV container.</summary>
    public static byte[] EncodeWav(float[] left, float[]? right, int sampleRate)
    {
        ArgumentNullException.ThrowIfNull(left);
        if (right is null)
        {
            using MemoryStream mono = new MemoryStream();
            WavFile.WriteMono16(mono, left, sampleRate);
            return mono.ToArray();
        }
        int frames = Math.Min(left.Length, right.Length);
        int dataBytes = frames * 2 * sizeof(short);
        using MemoryStream stream = new MemoryStream(44 + dataBytes);
        using (BinaryWriter writer = new BinaryWriter(stream, Encoding.ASCII, leaveOpen: true))
        {
            writer.Write("RIFF"u8.ToArray());
            writer.Write(36 + dataBytes);
            writer.Write("WAVE"u8.ToArray());
            writer.Write("fmt "u8.ToArray());
            writer.Write(16);
            writer.Write((short)1);
            writer.Write((short)2);
            writer.Write(sampleRate);
            writer.Write(sampleRate * 2 * sizeof(short));
            writer.Write((short)(2 * sizeof(short)));
            writer.Write((short)16);
            writer.Write("data"u8.ToArray());
            writer.Write(dataBytes);
            for (int i = 0; i < frames; i++)
            {
                writer.Write(ToPcm16(left[i]));
                writer.Write(ToPcm16(right[i]));
            }
            writer.Flush();
        }
        return stream.ToArray();
    }

    /// <summary>Encodes a buffer as a 16-bit PCM WAV container, mono when it carries a single channel.</summary>
    public static byte[] EncodeWav(AudioBuffer audio)
    {
        ArgumentNullException.ThrowIfNull(audio);
        if (audio.IsEmpty)
        {
            return EncodeWav([], null, Math.Max(1, audio.SampleRate));
        }
        (float[] left, float[] right) = audio.ToStereo();
        return audio.ChannelCount == 1
            ? EncodeWav(left, null, audio.SampleRate)
            : EncodeWav(left, right, audio.SampleRate);
    }

    /// <summary>The clip's duration in seconds at <paramref name="sampleRate"/>.</summary>
    public static double Seconds(int sampleCount, int sampleRate) => sampleRate <= 0 ? 0d : sampleCount / (double)sampleRate;

    private static WavFile.DecodedAudio Decode(AudioClip clip)
    {
        try
        {
            using MemoryStream stream = new MemoryStream(clip.Data, writable: false);
            return WavFile.Read(stream);
        }
        catch (Exception ex)
        {
            Logs.Error($"[Audio] Failed to decode an input clip (format hint '{clip.Format ?? "none"}').", ex);
            throw new HartsyInferenceException(
                $"The Engine decodes RIFF/WAVE PCM audio only (format hint '{clip.Format ?? "none"}'). "
                + "Convert the clip to WAV before submitting it — the Engine takes no ffmpeg dependency.", ex);
        }
    }

    private static short ToPcm16(float value) =>
        (short)Math.Clamp((int)MathF.Round(value * 32767f), short.MinValue, short.MaxValue);
}
