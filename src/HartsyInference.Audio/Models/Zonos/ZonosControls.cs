namespace HartsyInference.Audio.Models.Zonos;

/// <summary>The fine-grained conditioning controls Zonos accepts alongside the speaker embedding, with the
/// reference <c>make_cond_dict</c> defaults. <see cref="Emotion"/> is the 8-way vector (Happiness, Sadness,
/// Disgust, Fear, Surprise, Anger, Other, Neutral); it is renormalized to sum 1 before use, matching the
/// reference.</summary>
public sealed record ZonosControls
{
    public float[] Emotion { get; init; } = [0.3077f, 0.0256f, 0.0256f, 0.0256f, 0.0256f, 0.0256f, 0.2564f, 0.3077f];

    /// <summary>Maximum frequency in Hz (0–24000); 22050 for voice cloning.</summary>
    public float Fmax { get; init; } = 22_050f;

    /// <summary>Pitch standard deviation (0–400); 20–45 normal, 60–150 expressive.</summary>
    public float PitchStd { get; init; } = 20f;

    /// <summary>Speaking rate in phonemes per second (0–40); 15 default, 30 very fast, 10 slow.</summary>
    public float SpeakingRate { get; init; } = 15f;

    /// <summary>espeak language id (index into <see cref="ZonosLanguages"/>); default en-us.</summary>
    public int LanguageId { get; init; } = ZonosLanguages.DefaultId;

    /// <summary>CFG guidance scale (reference default 2.0).</summary>
    public float CfgScale { get; init; } = 2.0f;

    /// <summary>Sampling temperature (reference default 1.0; 0 → greedy argmax).</summary>
    public float Temperature { get; init; } = 1.0f;

    /// <summary>Min-p sampling threshold (reference default 0.1).</summary>
    public float MinP { get; init; } = 0.1f;

    /// <summary>Repetition penalty over the last 2 tokens per codebook (reference default 3.0; 1 disables).</summary>
    public float RepetitionPenalty { get; init; } = 3.0f;

    /// <summary>Max audio frames to generate (0 → config default ≈ 30 s).</summary>
    public int MaxNewTokens { get; init; } = 0;
}
