using HartsyInference.Audio.Dsp;
using HartsyInference.Audio.Sampling;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>CPU-only checks that <see cref="RasSampling"/> follows CosyVoice 2's reference <c>ras_sampling</c>: a token
/// seen in the trailing window is masked and redrawn from the full distribution; otherwise the nucleus draw stands.</summary>
public sealed class RasSamplingTests
{
    [Fact]
    public void TokenSeenOnceInWindow_IsMaskedAndRedrawn()
    {
        // Token 0 dominates; it already appears once in the last 10, so with tau_r 0.1 it must never be returned.
        List<int> history = [5, 0, 7];
        uint rng = DeterministicRng.Seed(3);
        for (int i = 0; i < 100; i++)
        {
            float[] logits = [20f, 0f, 0f, 0f, 0f, 0f, 0f, 0f];
            int tok = RasSampling.Draw(logits, history, topK: 25, topP: 0.8f, window: 10, tauR: 0.1f, ref rng);
            Assert.NotEqual(0, tok);
        }
    }

    [Fact]
    public void TokenOutsideWindow_IsKept()
    {
        // The earlier 0 sits outside the 10-token window, so the dominant token is drawn as usual.
        List<int> history = [0, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3];
        uint rng = DeterministicRng.Seed(3);
        float[] logits = [20f, 0f, 0f, 0f, 0f, 0f, 0f, 0f];
        Assert.Equal(0, RasSampling.Draw(logits, history, topK: 25, topP: 0.8f, window: 10, tauR: 0.1f, ref rng));
    }

    [Fact]
    public void RasDisabled_ReturnsNucleusDraw()
    {
        List<int> history = [0];
        uint rng = DeterministicRng.Seed(3);
        float[] logits = [20f, 0f, 0f, 0f];
        Assert.Equal(0, RasSampling.Draw(logits, history, topK: 25, topP: 0.8f, window: 10, tauR: 0.1f, ref rng, useRas: false));
    }
}
