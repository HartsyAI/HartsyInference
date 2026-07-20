using System.Text;
using HartsyInference.Audio.Io;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Logging;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Audio;

/// <summary>Decodes request <see cref="AudioClip"/>s to float PCM and encodes generated waveforms back to WAV. Only
/// RIFF/WAVE is decodable in-engine: the extension shelled out to ffmpeg for compressed containers, which is a host
/// dependency the Engine deliberately does not take, so anything else is refused with a message naming the limit.</summary>
internal static class AudioClipCodec
{
    /// <summary>Decodes a clip to mono float PCM in [-1, 1] at <paramref name="targetSampleRate"/>; empty for no input.</summary>
    internal static float[] DecodeMono(AudioClip? clip, int targetSampleRate)
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
    internal static (float[] Left, float[] Right) DecodeStereo(AudioClip? clip, int targetSampleRate)
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

    /// <summary>Encodes a mono (<paramref name="right"/> null) or stereo waveform as a 16-bit PCM WAV container.</summary>
    internal static byte[] EncodeWav(float[] left, float[]? right, int sampleRate)
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

    /// <summary>The clip's duration in seconds at <paramref name="sampleRate"/>.</summary>
    internal static double Seconds(int sampleCount, int sampleRate) => sampleRate <= 0 ? 0d : sampleCount / (double)sampleRate;

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
