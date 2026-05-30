using SharpInference.Audio.Models.CosyVoice;
using SharpInference.Core.Tensors;
using Xunit;

namespace SharpInference.Audio.Tests;

/// <summary>Checkpoint-free tests for the CosyVoice 2 LM-side logic — the config preset and the
/// speech-token sampler (deterministic draw, top-k, candidate masking, RAS re-roll). The full LM
/// forward needs the ~1 GB <c>llm.pt</c> (covered by the env-gated generation test later).</summary>
public sealed unsafe class CosyVoiceLmTests
{
    [Fact]
    public void Config_V2_0_5B_HasExpectedValues()
    {
        CosyVoiceConfig c = CosyVoiceConfig.V2_0_5B;
        Assert.Equal(6_561, c.SpeechTokenSize);
        Assert.Equal(25, c.TokenRateHz);
        Assert.Equal(24_000, c.SampleRate);
        Assert.Equal(896, c.Llm.HiddenSize);
        Assert.Equal(24, c.Llm.NumHiddenLayers);
        Assert.Equal(2, c.Llm.NumKeyValueHeads);
        Assert.Equal(10, c.Flow.NumEulerSteps);
        Assert.Equal(0.7f, c.Flow.CfgRate);
    }

    private static Tensor MakeLogits(int vocab, ReadOnlySpan<float> values)
    {
        Tensor t = new(new TensorShape(1, 1, vocab), DType.F32);
        float* p = (float*)t.DataPointer;
        for (int i = 0; i < vocab; i++) p[i] = i < values.Length ? values[i] : 0f;
        return t;
    }

    [Fact]
    public void Sampler_TopK1_ReturnsArgmaxAmongCandidates()
    {
        // eosToken=4 → candidates [0,4]. Index 2 is the max; out-of-range index 6 is bigger but ignored.
        CosyVoiceSamplingConfig cfg = new() { TopK = 1, TopP = 1f, Temperature = 1f, RepetitionPenalty = 1f, UseRas = false };
        SpeechSampler s = new(cfg, eosToken: 4, seed: 1);
        using Tensor logits = MakeLogits(8, [0.1f, 0.2f, 5.0f, 0.3f, 0.4f, 0.0f, 99.0f, 0.0f]);
        Assert.Equal(2, s.Sample(logits, new List<int>()));
    }

    [Fact]
    public void Sampler_IsDeterministic_ForFixedSeed()
    {
        CosyVoiceSamplingConfig cfg = new() { TopK = 25, TopP = 0.8f, Temperature = 0.8f, RepetitionPenalty = 1f, UseRas = false };
        using Tensor logits = MakeLogits(10, [1f, 2f, 1.5f, 0.5f, 3f, 0.2f, 1.1f]);
        List<int> a = new(), b = new();
        SpeechSampler sa = new(cfg, eosToken: 6, seed: 42);
        SpeechSampler sb = new(cfg, eosToken: 6, seed: 42);
        for (int i = 0; i < 20; i++) { a.Add(sa.Sample(logits, a)); b.Add(sb.Sample(logits, b)); }
        Assert.Equal(a, b);
    }

    [Fact]
    public void Sampler_NeverDrawsAboveEosToken()
    {
        // Out-of-range candidates [5,7] have huge logits but must never be drawn.
        CosyVoiceSamplingConfig cfg = new() { TopK = 25, TopP = 1f, Temperature = 1f, RepetitionPenalty = 1f, UseRas = false };
        SpeechSampler s = new(cfg, eosToken: 4, seed: 7);
        using Tensor logits = MakeLogits(8, [1f, 1f, 1f, 1f, 1f, 50f, 50f, 50f]);
        List<int> hist = new();
        for (int i = 0; i < 200; i++)
        {
            int tok = s.Sample(logits, hist);
            Assert.InRange(tok, 0, 4);
            hist.Add(tok);
        }
    }

    [Fact]
    public void Sampler_Ras_BreaksDegenerateRepeat()
    {
        // One token dominates; RAS must eventually force a different token instead of an infinite run.
        CosyVoiceSamplingConfig cfg = new() { TopK = 5, TopP = 1f, Temperature = 1f, RepetitionPenalty = 1f, UseRas = true, RasWindow = 10, RasMaxRepeat = 3 };
        SpeechSampler s = new(cfg, eosToken: 4, seed: 3);
        using Tensor logits = MakeLogits(8, [0f, 0f, 20f, 0.5f, 0.5f]);
        List<int> hist = new();
        int maxRun = 0, run = 0, prev = -1;
        for (int i = 0; i < 50; i++)
        {
            int tok = s.Sample(logits, hist);
            run = tok == prev ? run + 1 : 1;
            maxRun = Math.Max(maxRun, run);
            prev = tok;
            hist.Add(tok);
        }
        Assert.True(maxRun <= cfg.RasMaxRepeat, $"RAS should cap the repeat run at {cfg.RasMaxRepeat}, saw {maxRun}.");
    }
}
