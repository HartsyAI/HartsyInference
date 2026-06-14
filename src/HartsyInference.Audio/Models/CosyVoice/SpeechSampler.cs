using HartsyInference.Audio.Dsp;
using HartsyInference.Audio.Sampling;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.CosyVoice;

/// <summary>Autoregressive speech-token sampler for CosyVoice 2's LM. Applies repetition penalty →
/// temperature → top-k → top-p (nucleus) → multinomial draw, with Repetition-Aware Sampling (RAS):
/// if the draw would extend a degenerate run beyond <see cref="CosyVoiceSamplingConfig.RasMaxRepeat"/>
/// within the trailing <see cref="CosyVoiceSamplingConfig.RasWindow"/>, the chosen token is masked out
/// and the draw repeats from the same filtered distribution. Mirrors <c>cosyvoice/llm/llm.py</c>'s
/// <c>ras_sampling</c>. Deterministic for a fixed seed.
///
/// <para>Candidates span <c>[0, eosToken]</c> inclusive — the 6561 FSQ codes plus the end-of-speech
/// token; any extra <c>llm_decoder</c> slots above <c>eosToken</c> are never sampled.</para></summary>
public sealed unsafe class SpeechSampler
{
    private readonly CosyVoiceSamplingConfig _cfg;
    private readonly int _numCandidates;     // eosToken + 1
    private uint _rng;

    public SpeechSampler(CosyVoiceSamplingConfig cfg, int eosToken, int seed)
    {
        _cfg = cfg;
        _numCandidates = eosToken + 1;
        _rng = DeterministicRng.Seed(seed);
    }

    /// <summary>Draws the next speech token from <paramref name="logits"/> (<c>[1, 1, vocab]</c>),
    /// conditioning the repetition penalty + RAS check on the already-generated <paramref name="history"/>.</summary>
    public int Sample(Tensor logits, List<int> history)
    {
        float* lp = (float*)logits.DataPointer;
        float[] work = new float[_numCandidates];
        for (int i = 0; i < _numCandidates; i++) work[i] = lp[i];

        // Repetition penalty over the trailing window (HF convention: >0 divide, <0 multiply).
        if (_cfg.RepetitionPenalty != 1f && history.Count > 0)
        {
            int from = Math.Max(0, history.Count - _cfg.RasWindow);
            for (int i = from; i < history.Count; i++)
            {
                int tok = history[i];
                if ((uint)tok >= (uint)_numCandidates) continue;
                work[tok] = work[tok] > 0 ? work[tok] / _cfg.RepetitionPenalty : work[tok] * _cfg.RepetitionPenalty;
            }
        }

        int recentRepeat = CountTrailingRepeat(history);

        // Filtered draw via the shared sampler; RAS re-rolls (masking the repeated token) if we're at
        // the repeat ceiling.
        int chosen = NucleusSampler.Draw(work, _numCandidates, _cfg.Temperature, _cfg.TopK, _cfg.TopP, ref _rng);
        if (_cfg.UseRas && recentRepeat >= _cfg.RasMaxRepeat - 1 && history.Count > 0 && chosen == history[^1])
            chosen = NucleusSampler.Draw(work, _numCandidates, _cfg.Temperature, _cfg.TopK, _cfg.TopP, ref _rng, maskToken: history[^1]);
        return chosen;
    }

    private static int CountTrailingRepeat(List<int> history)
    {
        if (history.Count == 0) return 0;
        int last = history[^1];
        int count = 0;
        for (int i = history.Count - 1; i >= 0 && history[i] == last; i--) count++;
        return count;
    }
}
