using HartsyInference.Audio.Dsp;
using HartsyInference.Audio.Sampling;
using Xunit;

namespace HartsyInference.Audio.Tests.Sampling;

/// <summary>Pins <see cref="NucleusSampler.Draw"/>'s HF ordering option: with <c>topPOverTopK</c> the top-p cut is taken
/// on the distribution renormalized over the top-k set (HF <c>TopKLogitsWarper</c> → <c>TopPLogitsWarper</c>), which
/// keeps fewer tokens than a cut on the full-vocab softmax. Both the bounded path and the fallback (k above 1024) are
/// covered; the logits are built so the two rules disagree on which tokens survive.</summary>
public sealed class NucleusSamplerTopPOverTopKTests
{
    [Theory]
    [InlineData(100, 2, 2f, 0.5f)]
    [InlineData(3000, 1500, 10f, 0.9f)]
    public void TopPOverTopK_KeepsOnlyTheHead_WhereFullVocabCutKeepsTheTail(int count, int topK, float head, float topP)
    {
        // Full-vocab mass of token 0 stays under topP, its mass within the top-k set is over it.
        float[] logits = new float[count];
        logits[0] = head;
        logits[1] = head > 5f ? 0f : 1f;
        int otherHf = 0, otherFull = 0;
        uint rngHf = DeterministicRng.Seed(7), rngFull = DeterministicRng.Seed(7);
        for (int i = 0; i < 400; i++)
        {
            if (NucleusSampler.Draw(logits, count, 1f, topK, topP, ref rngHf, topPOverTopK: true) != 0) otherHf++;
            if (NucleusSampler.Draw(logits, count, 1f, topK, topP, ref rngFull) != 0) otherFull++;
        }
        Assert.Equal(0, otherHf);
        Assert.True(otherFull > 0, "the full-vocab cut should keep tail tokens for these logits");
    }
}
