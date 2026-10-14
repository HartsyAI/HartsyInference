using HartsyInference.Audio.Dsp;
using HartsyInference.Audio.Sampling;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.CosyVoice;

/// <summary>Autoregressive speech-token sampler for CosyVoice 2's LM: the reference <c>ras_sampling</c>
/// (<see cref="RasSampling"/>) over the speech-token candidates, with the end-of-speech gate applied first.</summary>
/// <remarks>Candidates span <c>[0, eosToken]</c> inclusive — the 6561 FSQ codes plus the end-of-speech token; any extra
/// <c>llm_decoder</c> slots above <c>eosToken</c> are never sampled. Deterministic for a fixed seed.</remarks>
public sealed unsafe class SpeechSampler
{
    private readonly CosyVoiceSamplingConfig _cfg;
    private readonly int _numCandidates;     // eosToken + 1
    private readonly int _eosToken;
    private uint _rng;

    public SpeechSampler(CosyVoiceSamplingConfig cfg, int eosToken, int seed)
    {
        _cfg = cfg;
        _numCandidates = eosToken + 1;
        _eosToken = eosToken;
        _rng = DeterministicRng.Seed(seed);
    }

    /// <summary>Draws the next speech token from <paramref name="logits"/> (<c>[1, 1, vocab]</c>) given the already-generated
    /// <paramref name="history"/>. <paramref name="gate"/> keeps end-of-speech out of the draw until its floor for
    /// <paramref name="targetTextTokens"/> is reached.</summary>
    public int Sample(Tensor logits, List<int> history, StopGate gate, int targetTextTokens)
    {
        float* lp = (float*)logits.DataPointer;
        float[] work = new float[_numCandidates];
        for (int i = 0; i < _numCandidates; i++) work[i] = lp[i];
        gate.Apply(work, history.Count, targetTextTokens, _eosToken);
        return RasSampling.Draw(work, history, _cfg.TopK, _cfg.TopP, _cfg.RasWindow, _cfg.RasTauR, ref _rng, _cfg.UseRas);
    }
}
