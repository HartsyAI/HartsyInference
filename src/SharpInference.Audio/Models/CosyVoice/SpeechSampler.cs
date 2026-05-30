using SharpInference.Core.Tensors;

namespace SharpInference.Audio.Models.CosyVoice;

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
        _rng = unchecked((uint)seed * 2654435761u + 0x9E3779B9u) | 1u;
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

        float temp = _cfg.Temperature > 0 ? _cfg.Temperature : 1f;
        int recentRepeat = CountTrailingRepeat(history);

        // RAS: if we're already at the repeat ceiling, re-roll while masking the repeated token.
        int chosen = DrawFiltered(work, temp, maskToken: -1);
        if (_cfg.UseRas && recentRepeat >= _cfg.RasMaxRepeat - 1 && history.Count > 0 && chosen == history[^1])
            chosen = DrawFiltered(work, temp, maskToken: history[^1]);
        return chosen;
    }

    /// <summary>One filtered multinomial draw: temperature → softmax → top-k → top-p → sample,
    /// optionally forcing one token's probability to zero (RAS re-roll).</summary>
    private int DrawFiltered(float[] logits, float temp, int maskToken)
    {
        float[] probs = new float[_numCandidates];
        float max = float.NegativeInfinity;
        for (int i = 0; i < _numCandidates; i++)
        {
            float v = logits[i] / temp;
            probs[i] = v;
            if (v > max) max = v;
        }
        double sum = 0;
        for (int i = 0; i < _numCandidates; i++)
        {
            float e = MathF.Exp(probs[i] - max);
            probs[i] = e;
            sum += e;
        }
        float inv = (float)(1.0 / sum);
        for (int i = 0; i < _numCandidates; i++) probs[i] *= inv;
        if ((uint)maskToken < (uint)_numCandidates) probs[maskToken] = 0f;

        // top-k + top-p over an index list sorted by probability (descending).
        int[] order = ArgsortDescending(probs);
        int k = _cfg.TopK > 0 ? Math.Min(_cfg.TopK, _numCandidates) : _numCandidates;
        float cumulative = 0f;
        int keep = 0;
        for (int rank = 0; rank < k; rank++)
        {
            cumulative += probs[order[rank]];
            keep = rank + 1;
            if (_cfg.TopP > 0 && _cfg.TopP < 1f && cumulative >= _cfg.TopP) break;
        }

        // Renormalize the kept set and draw.
        float keptSum = 0f;
        for (int rank = 0; rank < keep; rank++) keptSum += probs[order[rank]];
        if (keptSum <= 0f) return order[0];
        float r = NextFloat() * keptSum;
        float acc = 0f;
        for (int rank = 0; rank < keep; rank++)
        {
            acc += probs[order[rank]];
            if (r <= acc) return order[rank];
        }
        return order[keep - 1];
    }

    private static int CountTrailingRepeat(List<int> history)
    {
        if (history.Count == 0) return 0;
        int last = history[^1];
        int count = 0;
        for (int i = history.Count - 1; i >= 0 && history[i] == last; i--) count++;
        return count;
    }

    private static int[] ArgsortDescending(float[] values)
    {
        int[] idx = new int[values.Length];
        for (int i = 0; i < idx.Length; i++) idx[i] = i;
        Array.Sort(idx, (a, b) => values[b].CompareTo(values[a]));
        return idx;
    }

    private float NextFloat()
    {
        _rng ^= _rng << 13; _rng ^= _rng >> 17; _rng ^= _rng << 5;
        return (_rng & 0xFFFFFF) / 16777216f;
    }
}
