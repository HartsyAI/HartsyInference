using HartsyInference.Audio.Models.CosyVoice;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>CPU-only checks for the end-of-speech gate in <see cref="SpeechSampler"/>: with EOS dominant the
/// unrestricted draw returns it, and <c>ignoreEos</c> must never return it. Covers the top-k fast path and
/// the full-distribution path (<c>TopK = 0</c>).</summary>
public sealed unsafe class CosyVoiceSpeechSamplerTests
{
    private const int EosToken = 4;     // candidates [0, 4] inclusive; index 4 is end-of-speech
    private const int Vocab = 6;        // extra decoder rows above EOS are never sampled

    private static Tensor EosDominantLogits()
    {
        Tensor t = new(new TensorShape(1, 1, Vocab), DType.F32);
        float* p = (float*)t.DataPointer;
        for (int i = 0; i < Vocab; i++) p[i] = 0f;
        p[EosToken] = 20f;
        return t;
    }

    [Theory]
    [InlineData(25)]
    [InlineData(0)]
    public void WithoutIgnoreEos_EosDominant_DrawsEos(int topK)
    {
        CosyVoiceSamplingConfig cfg = new CosyVoiceSamplingConfig { TopK = topK, UseRas = false, RepetitionPenalty = 1f };
        SpeechSampler sampler = new SpeechSampler(cfg, EosToken, seed: 7);
        using Tensor logits = EosDominantLogits();
        List<int> history = new List<int>();

        for (int i = 0; i < 50; i++)
            Assert.Equal(EosToken, sampler.Sample(logits, history));
    }

    [Theory]
    [InlineData(25)]
    [InlineData(0)]
    public void IgnoreEos_EosDominant_NeverDrawsEos(int topK)
    {
        CosyVoiceSamplingConfig cfg = new CosyVoiceSamplingConfig { TopK = topK, UseRas = false, RepetitionPenalty = 1f };
        SpeechSampler sampler = new SpeechSampler(cfg, EosToken, seed: 7);
        using Tensor logits = EosDominantLogits();
        List<int> history = new List<int>();

        for (int i = 0; i < 200; i++)
        {
            int tok = sampler.Sample(logits, history, ignoreEos: true);
            Assert.InRange(tok, 0, EosToken - 1);
            history.Add(tok);
        }
    }
}
