using HartsyInference.Audio.Models.CosyVoice;
using HartsyInference.Audio.Sampling;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>CPU-only checks for the end-of-speech gate in <see cref="SpeechSampler"/>: with EOS dominant the
/// unrestricted draw returns it, and a closed <see cref="StopGate"/> must never return it until its floor is reached.
/// Covers the top-k fast path and the full-distribution path (<c>TopK = 0</c>).</summary>
public sealed unsafe class CosyVoiceSpeechSamplerTests
{
    private const int EosToken = 4;     // candidates [0, 4] inclusive; index 4 is end-of-speech
    private const int Vocab = 6;        // extra decoder rows above EOS are never sampled
    private static readonly StopGate Gate = new(MinTokens: 0, MinTokenTextRatio: 2f);
    private const int TargetText = 5;   // floor = 10 speech tokens

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
            Assert.Equal(EosToken, sampler.Sample(logits, history, StopGate.Disabled, 0));
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

        for (int i = 0; i < 10; i++)
        {
            int tok = sampler.Sample(logits, history, Gate, TargetText);
            Assert.InRange(tok, 0, EosToken - 1);
            history.Add(tok);
        }
        // Gate opens at 10 generated tokens: EOS is then drawable again.
        Assert.Equal(EosToken, sampler.Sample(logits, history, Gate, TargetText));
    }
}
