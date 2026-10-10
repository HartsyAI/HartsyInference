using System;
using System.Collections.Generic;

namespace HartsyInference.LLM.Sampling;

/// <summary>Penalizes the logits of already-generated tokens following the Hugging Face convention (divide positive logits, multiply
/// negative ones), ONCE per distinct token however often it occurs. A <paramref name="penalty"/> of 1.0 leaves logits unchanged at
/// apply time.</summary>
/// <remarks>
/// Hugging Face (<c>RepetitionPenaltyLogitsProcessor</c> gathers and scatters by token id) and llama.cpp (<c>penalty_repeat</c> over the
/// distinct tokens of its window) both apply the penalty once per token. This step used to apply it once per OCCURRENCE, so a token
/// generated n times had its logit divided by <c>penalty^n</c>: after a few thousand tokens of reasoning, spaces, punctuation and the
/// identifiers a code review keeps naming were suppressed far past their next-best alternatives, and the output degraded into
/// misspelled names and stray languages. The graph-decode device path keeps its history distinct for the same reason.
/// </remarks>
public sealed class RepetitionPenaltyStep(float penalty) : ISamplerStep
{
    private readonly float _penalty = penalty;

    // One stamp per vocabulary entry: a token is penalized only if its stamp is not this call's. A chain is built per generation and
    // driven from one thread (see SamplerChain.FromOptions), so the array is sized once and reused without locking.
    private int[] _stamps = [];
    private int _epoch;

    /// <inheritdoc/>
    public void Apply(Span<float> logits, IReadOnlyList<int> history)
    {
        if (_penalty == 1.0f)
        {
            return;
        }
        if (_stamps.Length < logits.Length)
        {
            _stamps = new int[logits.Length];
            _epoch = 0;
        }
        if (++_epoch == int.MaxValue)
        {
            Array.Clear(_stamps);
            _epoch = 1;
        }
        for (int i = 0; i < history.Count; i++)
        {
            int token = history[i];
            if ((uint)token >= (uint)logits.Length || _stamps[token] == _epoch)
            {
                continue;
            }
            _stamps[token] = _epoch;
            float logit = logits[token];
            logits[token] = logit > 0.0f ? logit / _penalty : logit * _penalty;
        }
    }
}
