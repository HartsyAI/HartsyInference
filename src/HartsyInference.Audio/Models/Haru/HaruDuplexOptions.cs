using HartsyInference.LLM.Sampling;

namespace HartsyInference.Audio.Models.Haru;

/// <summary>How a conversation samples and starts. The defaults are the reference implementation's.</summary>
internal sealed record HaruDuplexOptions
{
    /// <summary>Audio guidance weight, valid from 1 to 2.</summary>
    public float GuidanceScale { get; init; } = 1.5f;

    /// <summary>Text sampling temperature; zero or less takes the most likely token.</summary>
    public float Temperature { get; init; } = 0.7f;

    public int TopK { get; init; } = 25;

    public float TopP { get; init; } = 1.0f;

    /// <summary>The repetition penalty of the text stream; null turns it off.</summary>
    public DryOptions? Dry { get; init; } = new();

    /// <summary>Added to the EOP logit: above zero the model starts to speak sooner.</summary>
    public float EopBias { get; init; }

    /// <summary>The model opens the conversation.</summary>
    public bool SpeakFirst { get; init; }

    /// <summary>Frames after which the conversation is done: 60 minutes.</summary>
    public int MaxFrames { get; init; } = 27_000;

    /// <summary>The longest voice reference that is kept, counted from its end.</summary>
    public double ReferenceSeconds { get; init; } = HaruAudio.MaxReferenceSeconds;

    /// <summary>Seed of the sampler and of the default noise source.</summary>
    public int Seed { get; init; } = 1;
}
