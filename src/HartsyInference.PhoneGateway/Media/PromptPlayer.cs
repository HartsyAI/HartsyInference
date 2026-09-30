using System.Reflection;

namespace HartsyInference.PhoneGateway.Media;

/// <summary>Plays 8 kHz PCM16 prompts into the outbound ring: the two embedded ones (<see cref="PromptKind"/>) and raw
/// PCM files for greetings and the <c>play_prompt</c> tool. The embedded prompts are synthesized tone patterns (see
/// <c>docs/Research/PHONE_GATEWAY.md</c>); they are what a caller hears when the voice host is unreachable.</summary>
public sealed class PromptPlayer
{
    /// <summary>Longest prompt file accepted, in samples (60 s at 8 kHz).</summary>
    public const int MaxFileSamples = 60 * ClockedAudioSource.SampleRate;

    private const string ResourcePrefix = "HartsyInference.PhoneGateway.Assets.";

    private readonly short[] _oneMoment = LoadEmbedded("one-moment.pcm");
    private readonly short[] _goodbye = LoadEmbedded("goodbye.pcm");

    /// <summary>Duration of an embedded prompt in milliseconds.</summary>
    public int DurationMs(PromptKind kind) => Samples(kind).Length * 1000 / ClockedAudioSource.SampleRate;

    /// <summary>Queues an embedded prompt on <paramref name="path"/> and returns its duration in milliseconds.</summary>
    public int Play(OutboundAudioPath path, PromptKind kind)
    {
        ArgumentNullException.ThrowIfNull(path);
        short[] samples = Samples(kind);
        path.WritePrompt(samples);
        return samples.Length * 1000 / ClockedAudioSource.SampleRate;
    }

    /// <summary>Queues a raw 8 kHz PCM16 little-endian file and returns its duration in milliseconds.</summary>
    public int PlayFile(OutboundAudioPath path, string filePath)
    {
        ArgumentNullException.ThrowIfNull(path);
        short[] samples = LoadFile(filePath);
        path.WritePrompt(samples);
        return samples.Length * 1000 / ClockedAudioSource.SampleRate;
    }

    /// <summary>Reads a raw 8 kHz PCM16 file, refusing an odd length, an empty file or one over <see cref="MaxFileSamples"/>.</summary>
    public static short[] LoadFile(string filePath)
    {
        FileInfo info = new(filePath);
        if (!info.Exists)
        {
            throw new FileNotFoundException("Prompt file not found.", filePath);
        }
        if (info.Length == 0 || (info.Length & 1) != 0 || info.Length / 2 > MaxFileSamples)
        {
            throw new InvalidDataException($"Prompt {filePath} must be 2..{MaxFileSamples * 2} bytes of PCM16; it is {info.Length} bytes.");
        }
        byte[] bytes = File.ReadAllBytes(filePath);
        return ToSamples(bytes);
    }

    private short[] Samples(PromptKind kind) => kind switch
    {
        PromptKind.OneMoment => _oneMoment,
        PromptKind.Goodbye => _goodbye,
        _ => throw new ArgumentOutOfRangeException(nameof(kind), kind, "Unknown prompt."),
    };

    private static short[] LoadEmbedded(string name)
    {
        using Stream? stream = Assembly.GetExecutingAssembly().GetManifestResourceStream(ResourcePrefix + name)
            ?? throw new InvalidOperationException($"Embedded prompt {name} is missing from the gateway assembly.");
        using MemoryStream copy = new();
        stream.CopyTo(copy);
        return ToSamples(copy.ToArray());
    }

    private static short[] ToSamples(byte[] bytes)
    {
        short[] samples = new short[bytes.Length / 2];
        for (int i = 0; i < samples.Length; i++)
        {
            samples[i] = (short)(bytes[2 * i] | (bytes[2 * i + 1] << 8));
        }
        return samples;
    }
}
