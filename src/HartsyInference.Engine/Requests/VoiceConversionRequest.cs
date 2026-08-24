using HartsyInference.Core.MemoryManagement;

namespace HartsyInference.Engine.Requests;

/// <summary>Native voice-conversion request: re-voice the source audio, optionally toward a target voice (for models that condition on one, e.g. OpenVoice tone-color transfer; RVC carries the target in its weights).</summary>
public sealed record VoiceConversionRequest
{
    /// <summary>The source audio to re-voice.</summary>
    public required AudioClip Source { get; init; }

    /// <summary>Target voice reference; null for source-only models (e.g. RVC).</summary>
    public AudioClip? Target { get; init; }

    /// <summary>Pitch shift in semitones (RVC); 0 = no shift.</summary>
    public double PitchShift { get; init; }

    /// <summary>F0 estimator for RVC (<c>"rmvpe"</c> or <c>"yin"</c>); null defaults to RMVPE.</summary>
    public string? F0Method { get; init; }

    /// <summary>Per-request VRAM lever overrides; null follows the backend's policy.</summary>
    public VramOverrides? Vram { get; init; }
}
