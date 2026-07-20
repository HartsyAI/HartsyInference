namespace HartsyInference.Engine.Audio;

/// <summary>Per-model specifics for the generic transcription path: how to turn a variant hint into a HuggingFace
/// repo, how to load it, and the rate the input audio is decoded to first.</summary>
internal sealed class SttModelDescriptor
{
    /// <summary>Maps the request's variant hint to a HuggingFace repo id.</summary>
    internal required Func<string, string> ResolveRepo { get; init; }

    /// <summary>Loads the repo (downloading on first use) into a uniform runner.</summary>
    internal required Func<string, CancellationToken, Task<ISttRunner>> LoadAsync { get; init; }

    /// <summary>Rate the input is decoded to before transcription (Whisper/Moonshine 16 kHz; Kyutai 24 kHz).</summary>
    internal int InputSampleRate { get; init; } = 16_000;
}
