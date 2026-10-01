using System;
using System.Collections.Generic;

namespace HartsyInference.LLM.Sampling;

/// <summary>Nucleus filter: keeps the smallest set of highest-probability tokens whose cumulative softmax mass reaches p, masking the rest to negative infinity. Values of <paramref name="p"/> at or above 1.0 disable the filter at apply time.</summary>
/// <remarks>One <see cref="TopPStep"/> instance lives for exactly one generation (built fresh per
/// <see cref="SamplerChain.FromOptions"/> call, which <c>TextGenerationPipeline.Generate</c> calls once per
/// request) and its <see cref="Apply"/> runs once per decode step of that SAME generation — so the scratch
/// buffers below are sized once, on the first token, and reused for every later token instead of being
/// reallocated on every single one.
///
/// <para>Sorting the full vocabulary every token — even via <see cref="SamplerMath.SortDescendingByValue"/>'s
/// fast primitive sort — measured ~11 ms of a ~12.4 ms step at Qwen3's ~152K-token vocabulary (see
/// <c>benchmarks/results/2026-10-01_llm_short_reply_decode.md</c>'s "Tier 2" section): the sort's O(n log n) is
/// unavoidable in <c>n</c> as long as <c>n</c> stays the whole vocabulary. But the cumulative walk almost never
/// needs most of it — a token whose probability cannot exceed every token this step is about to keep anyway
/// never changes the kept SET, only how fast the (already <c>-Infinity</c> either way) tail would have been
/// reached. <see cref="Apply"/> exploits that: collect the tokens above a tiny probability floor
/// (<see cref="CandidateEps"/>) into a compact buffer first (one O(n) pass, no sort), and sort ONLY those when
/// their total probability already reaches <c>p</c> on its own — which it provably does whenever the
/// distribution is at all peaked, the common case for a trained LLM's next-token distribution. The identity
/// argument: every candidate's probability is, by construction, strictly greater than every non-candidate's, so
/// a full descending sort would always place every candidate before every non-candidate; if the candidates'
/// OWN total already reaches <c>p</c>, the full sort's cumulative walk reaches <c>p</c> before ever leaving the
/// candidate block, so sorting just that block reproduces EXACTLY the same kept set (and, read in the same
/// order, the exact same floating-point cumulative sums) the full sort would. When the candidates' total falls
/// short — a near-uniform distribution, e.g. a very high temperature — that guarantee doesn't hold, and
/// <see cref="Apply"/> falls back to sorting the whole vocabulary exactly as it did before this candidate
/// pre-filter existed.</para>
///
/// <para><see cref="Apply"/> also falls back when the candidates' total (summed in vocabulary-index order)
/// clears <c>p</c> but the SAME values, summed again in sorted order during the cumulative walk, do not —
/// float addition is not associative, so the two sums can round differently when <c>p</c> sits within a few
/// ULPs of the true candidate total. That is the only way the walk can exhaust every candidate without
/// reaching <c>p</c> once the index-order sum already cleared it, and falling back there is always correct
/// (never a different kept set than the old code), just occasionally paying the fallback's cost in a margin
/// case real model logits essentially never land on exactly.</para></remarks>
public sealed class TopPStep(float p) : ISamplerStep
{
    // A token at or below this can never be needed to reach ANY realistic p once something more probable
    // exists — see the type doc's identity argument. Chosen small enough that only a genuinely near-uniform
    // distribution (not a trained model's ordinary output) ever fails the candidates-reach-p check below; at
    // Qwen3's ~152K-token vocabulary even a PERFECTLY uniform distribution has every token's probability
    // (~6.6e-6) above this floor, so it alone is never the reason the fast path is unavailable.
    private const float CandidateEps = 1e-6f;

    private readonly float _p = p;
    private float[]? _probs;
    private int[]? _order;
    private float[]? _candidateProbs;
    private int[]? _candidateIndices;

    /// <inheritdoc/>
    public void Apply(Span<float> logits, IReadOnlyList<int> history)
    {
        if (_p >= 1.0f)
        {
            return;
        }
        int count = logits.Length;
        if (_probs is null || _probs.Length != count)
        {
            _probs = new float[count];
            _order = new int[count];
            _candidateProbs = new float[count];
            _candidateIndices = new int[count];
        }
        float[] probs = _probs;
        SamplerMath.Softmax(logits, probs);

        float[] candidateProbs = _candidateProbs!;
        int[] candidateIndices = _candidateIndices!;
        int candidateCount = 0;
        float candidateSum = 0.0f;
        for (int i = 0; i < count; i++)
        {
            float pr = probs[i];
            if (pr > CandidateEps)
            {
                candidateProbs[candidateCount] = pr;
                candidateIndices[candidateCount] = i;
                candidateCount++;
                candidateSum += pr;
            }
        }

        if (candidateCount > 0 && candidateSum >= _p)
        {
            SamplerMath.SortDescendingByValue(candidateProbs, candidateIndices, candidateCount);
            float cumulative = 0.0f;
            int keep = 0;
            bool reachedP = false;
            for (int rank = 0; rank < candidateCount; rank++)
            {
                cumulative += candidateProbs[rank];
                keep = rank + 1;
                if (cumulative >= _p)
                {
                    reachedP = true;
                    break;
                }
            }
            // candidateSum above summed the SAME values in vocabulary-index order; this walk sums them in
            // sorted order. Float addition isn't associative, so when p sits within a few ULPs of the true
            // candidate total the two sums can round differently — candidateSum clears p, but this walk
            // exhausts every candidate without ever doing so. That is the only way reachedP can be false here
            // (candidateSum >= _p already guarantees the EXACT candidate total, computed the other way, meets
            // or beats p). Falling through to the full-vocabulary fallback below is always correct, so this
            // never produces a different kept set than the old code — it only pays the fallback's cost in a
            // margin case that real model logits essentially never hit (see TopPSortRefactorIdentityTests'
            // NearCandidateBoundary cases, which construct p values exactly on these boundaries to check it).
            if (reachedP)
            {
                // Every non-candidate is already known-excluded (probability <= CandidateEps, strictly less
                // than every surviving candidate's) — read off probs directly rather than re-deriving the set.
                // (NaN logits: probs[i] would be NaN too, "<= CandidateEps" is false for NaN, so that entry is
                // left unmasked — same as the pre-fix code's undefined ordering of a NaN through its sort;
                // NaN logits mean sampling is already broken upstream, this isn't trying to make that defined.)
                for (int i = 0; i < count; i++)
                {
                    if (probs[i] <= CandidateEps)
                    {
                        logits[i] = float.NegativeInfinity;
                    }
                }
                for (int rank = keep; rank < candidateCount; rank++)
                {
                    logits[candidateIndices[rank]] = float.NegativeInfinity;
                }
                return;
            }
        }

        // Fallback: the candidates alone can't reach p (a near-uniform distribution) — sort the whole
        // vocabulary, exactly as before the candidate pre-filter above existed.
        int[] order = _order!;
        for (int i = 0; i < count; i++)
        {
            order[i] = i;
        }
        SamplerMath.SortDescendingByValue(probs, order, count);
        float fullCumulative = 0.0f;
        int fullKeep = 0;
        for (int rank = 0; rank < count; rank++)
        {
            fullCumulative += probs[rank];
            fullKeep = rank + 1;
            if (fullCumulative >= _p)
            {
                break;
            }
        }
        for (int rank = fullKeep; rank < count; rank++)
        {
            logits[order[rank]] = float.NegativeInfinity;
        }
    }
}
