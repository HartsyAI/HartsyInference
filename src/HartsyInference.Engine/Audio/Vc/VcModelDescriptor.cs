namespace HartsyInference.Engine.Audio;

/// <summary>Per-model specifics for the generic voice-conversion path.</summary>
internal sealed class VcModelDescriptor
{
    /// <summary>True when the model auto-downloads its own weights; false for user-placed checkpoints (RVC voices).</summary>
    internal required bool ManagesOwnWeights { get; init; }

    /// <summary>Stable cache key for a resolved selector (an HF repo id, or the local checkpoint path).</summary>
    internal required Func<AudioModelSelector, string> CacheKey { get; init; }

    /// <summary>Loads the model into a runner. Loading is device-independent (no backend needed).</summary>
    internal required Func<AudioModelSelector, CancellationToken, Task<IVcRunner>> LoadAsync { get; init; }

    /// <summary>Rate the source/target audio is decoded to (RVC 16 kHz content input; OpenVoice 22.05 kHz).</summary>
    internal int InputSampleRate { get; init; } = 16_000;
}
