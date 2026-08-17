namespace HartsyInference.Engine.Audio.Wake.Wyoming;

/// <summary>Scores the audio of one Wyoming wake connection.
///
/// <para>Supplied by the host rather than built here, because wake scoring is continuous and must not touch the
/// shared inference backend the ASR/TTS admission gate protects — a detector that ran on it would contend with
/// every image or video job. The host owns that decision (its own <c>WakeModelSet</c> on a dedicated CPU backend,
/// typically), so this endpoint only advertises and drives it.</para>
///
/// <para>One instance per connection: Home Assistant opens a fresh one per wake stream, and Wyoming's
/// <c>detect</c> event lets each pick its own word list.</para></summary>
public interface IWyomingWakeDetector : IDisposable
{
    /// <summary>Feeds one chunk and returns a hit when a word fired on it.</summary>
    /// <param name="samples">Mono 16 kHz audio scaled to int16 range (±32768), not normalized to ±1.</param>
    WyomingWakeHit? Push(ReadOnlySpan<float> samples);

    /// <summary>Clears model state at a stream boundary so audio is never spliced across a gap.</summary>
    void Reset();
}
