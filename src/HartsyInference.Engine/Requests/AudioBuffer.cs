namespace HartsyInference.Engine.Requests;

/// <summary>Engine-native raw audio: planar float PCM in [-1, 1] plus its sample rate. The decoded counterpart to
/// <see cref="AudioClip"/> (encoded container bytes in) and <see cref="AudioResult"/> (encoded container bytes out) —
/// use this whenever a waveform moves between engine components without a container around it.</summary>
public sealed record AudioBuffer
{
    /// <summary>Planar per-channel samples; index 0 is left, 1 is right. Borrowed, not copied — accessors hand back these arrays.</summary>
    public required float[][] Channels { get; init; }

    /// <summary>Samples per second.</summary>
    public required int SampleRate { get; init; }

    /// <summary>A buffer carrying no samples.</summary>
    public static AudioBuffer Empty { get; } = new AudioBuffer { Channels = [], SampleRate = 0 };

    /// <summary>Number of channels present.</summary>
    public int ChannelCount => Channels.Length;

    /// <summary>Samples per channel; the shortest channel wins so a ragged set can still be played.</summary>
    public int FrameCount
    {
        get
        {
            if (Channels.Length == 0)
                return 0;
            int shortest = Channels[0].Length;
            for (int c = 1; c < Channels.Length; c++)
            {
                shortest = Math.Min(shortest, Channels[c].Length);
            }
            return shortest;
        }
    }

    /// <summary>True when there is nothing to play.</summary>
    public bool IsEmpty => FrameCount == 0 || SampleRate <= 0;

    /// <summary>Duration in seconds.</summary>
    public double Seconds => SampleRate <= 0 ? 0d : FrameCount / (double)SampleRate;

    /// <summary>Downmixes to a single channel by averaging; empty for an empty buffer.</summary>
    public float[] ToMono()
    {
        if (Channels.Length == 0)
            return [];
        if (Channels.Length == 1)
            return Channels[0];
        int frames = FrameCount;
        float[] mono = new float[frames];
        float inv = 1f / Channels.Length;
        for (int c = 0; c < Channels.Length; c++)
        {
            float[] source = Channels[c];
            for (int i = 0; i < frames; i++)
            {
                mono[i] += source[i] * inv;
            }
        }
        return mono;
    }

    /// <summary>Splits into a left/right pair; a mono buffer returns the same array for both channels.</summary>
    public (float[] Left, float[] Right) ToStereo()
    {
        if (Channels.Length == 0)
            return ([], []);
        float[] left = Channels[0];
        return (left, Channels.Length > 1 ? Channels[1] : left);
    }

    /// <summary>Truncates every channel to at most <paramref name="seconds"/>; returns this buffer when already shorter.</summary>
    public AudioBuffer TrimTo(double seconds)
    {
        if (IsEmpty || seconds <= 0d)
            return Empty;
        int keep = (int)Math.Floor(seconds * SampleRate);
        if (keep >= FrameCount)
            return this;
        if (keep <= 0)
            return Empty;
        float[][] trimmed = new float[Channels.Length][];
        for (int c = 0; c < Channels.Length; c++)
        {
            trimmed[c] = Channels[c][..Math.Min(keep, Channels[c].Length)];
        }
        return this with { Channels = trimmed };
    }

    /// <summary>Silence-pads every channel up to <paramref name="seconds"/>; returns this buffer when already that long.</summary>
    public AudioBuffer PadTo(double seconds)
    {
        if (IsEmpty || seconds <= 0d)
            return this;
        int target = (int)Math.Ceiling(seconds * SampleRate);
        if (target <= FrameCount)
            return this;
        float[][] padded = new float[Channels.Length][];
        for (int c = 0; c < Channels.Length; c++)
        {
            padded[c] = new float[target];
            Array.Copy(Channels[c], padded[c], Math.Min(Channels[c].Length, target));
        }
        return this with { Channels = padded };
    }

    /// <summary>Wraps planar channel arrays, normalizing a null/empty set to <see cref="Empty"/>.</summary>
    public static AudioBuffer FromChannels(float[][]? channels, int sampleRate)
    {
        if (channels is null || channels.Length == 0 || sampleRate <= 0)
            return Empty;
        AudioBuffer buffer = new AudioBuffer { Channels = channels, SampleRate = sampleRate };
        return buffer.IsEmpty ? Empty : buffer;
    }
}
