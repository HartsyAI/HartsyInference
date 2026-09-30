using HartsyInference.Audio.Models.Wake;
using HartsyInference.Core.Backends;

namespace HartsyInference.Voice.Tests.Fakes;

/// <summary>A VAD scripted by the audio itself: a chunk's speech probability is its mean absolute sample value, so a
/// test writes 0.9 to mean speech and 0 to mean silence and drives the real <see cref="SileroVadStream"/> exactly.
/// Allocation-free, like the model it stands in for.</summary>
internal sealed class LevelVadModel : IVadModel
{
    public int WindowSamples => 512;

    public int Resets { get; private set; }

    public float Process(IBackend backend, ReadOnlySpan<float> chunk)
    {
        float sum = 0f;
        foreach (float sample in chunk)
        {
            sum += MathF.Abs(sample);
        }
        return Math.Clamp(sum / chunk.Length, 0f, 1f);
    }

    public void Reset() => Resets++;
}
