using HartsyInference.Core.Backends;

namespace HartsyInference.Audio.Models.Wake;

/// <summary>A voice-activity model scored one fixed-size chunk at a time, carrying its own recurrent state.</summary>
/// <remarks>What <see cref="SileroVadStream"/> needs from a model and nothing more, so the endpointing logic can be
/// driven by a scripted model in tests, or by another VAD later, without weights. One instance holds one stream's
/// state and is not thread-safe.</remarks>
public interface IVadModel
{
    /// <summary>Samples consumed per <see cref="Process"/> call.</summary>
    int WindowSamples { get; }

    /// <summary>Speech probability in <c>[0, 1]</c> for one chunk of exactly <see cref="WindowSamples"/> samples normalized to ±1.</summary>
    float Process(IBackend backend, ReadOnlySpan<float> chunk);

    /// <summary>Clears the recurrent state; call on any discontinuity in the audio.</summary>
    void Reset();
}
