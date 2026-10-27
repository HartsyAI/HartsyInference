using HartsyInference.Audio.Dsp;

namespace HartsyInference.Audio.Pipelines;

/// <summary>How an Orpheus token run ended.</summary>
public enum GenerationStop
{
    /// <summary>The model emitted EndOfSpeech.</summary>
    NaturalEnd,

    /// <summary>The run used every allowed token without EndOfSpeech. The take is probably cut short.</summary>
    TokenLimit,

    /// <summary>The KV cache reached the model's context window before EndOfSpeech.</summary>
    ContextLimit,
}

/// <summary>Filled in by the token loop while it runs.</summary>
internal sealed class GenerationTrace
{
    public GenerationStop Stop { get; set; } = GenerationStop.TokenLimit;
}

/// <summary>Result of <see cref="OrpheusPipeline.SynthesizeWithTrace"/>: 24 kHz PCM plus how generation ended.</summary>
public sealed record OrpheusSynthesis(float[] Audio, GenerationStop Stop, int GeneratedTokens);

/// <summary>RNG state shared by every chunk of one request. Seed it once; each chunk continues the stream, so
/// consecutive chunks don't replay the same random draws.</summary>
public sealed class OrpheusSamplerState(int seed)
{
    internal uint Rng { get; set; } = DeterministicRng.Seed(seed);
}
