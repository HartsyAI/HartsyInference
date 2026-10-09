using System.Runtime.InteropServices;
using HartsyInference.LLM.Sampling;

namespace HartsyInference.LLM.Generation.Speculative;

/// <summary>Speculative generation over a scorer that holds no state between calls: each round drafts, scores the draft in one pass, verifies it exactly and keeps the outcome.</summary>
public static class SpeculativeLoop
{
    /// <summary>Appends up to <paramref name="maxNewTokens"/> tokens to <paramref name="tokens"/> and returns how many were appended.</summary>
    /// <param name="sampler">Supplies the target distribution at each position (its repetition penalty and truncation); its own random state is not used.</param>
    /// <param name="uniform">The draws used for acceptance and correction.</param>
    /// <param name="stopTokens">Tokens that end the run. The first one emitted is kept, and nothing after it is emitted, even inside an accepted draft.</param>
    public static int Generate(ISpeculativeScorer scorer, IDraftProposer proposer, SamplerChain sampler, Func<double> uniform,
        List<int> tokens, int maxNewTokens, int maxDraft, int vocab, IReadOnlySet<int>? stopTokens = null)
    {
        ArgumentNullException.ThrowIfNull(scorer);
        ArgumentNullException.ThrowIfNull(proposer);
        ArgumentNullException.ThrowIfNull(sampler);
        ArgumentNullException.ThrowIfNull(uniform);
        ArgumentNullException.ThrowIfNull(tokens);
        if (vocab <= 0) throw new ArgumentOutOfRangeException(nameof(vocab));

        float[] scratch = new float[vocab];
        int produced = 0;
        while (produced < maxNewTokens)
        {
            // the round always ends with one more token (the correction or the bonus), so the draft leaves room for it
            int draftLimit = Math.Max(0, Math.Min(maxDraft, maxNewTokens - produced - 1));
            DraftBlock block = proposer.Propose(CollectionsMarshal.AsSpan(tokens), draftLimit);
            int k = block.Tokens.Length;
            if (k > draftLimit) throw new InvalidOperationException($"The proposer returned {k} tokens for a limit of {draftLimit}.");

            float[][] logits = Rows(k + 1, vocab);
            scorer.Score(CollectionsMarshal.AsSpan(tokens), block.Tokens, logits);

            float[][] target = Rows(k + 1, vocab);
            List<int> history = new(tokens);
            for (int j = 0; j <= k; j++)
            {
                logits[j].AsSpan().CopyTo(scratch);
                sampler.Distribution(scratch, history, target[j]);
                if (j < k) history.Add(block.Tokens[j]);
            }

            SpeculativeOutcome outcome = RejectionSampler.Verify(block.Tokens, target, block.Probs, uniform);
            // the accepted draft, then the correction or the bonus token
            for (int j = 0; j <= outcome.Accepted; j++)
            {
                int emitted = j < outcome.Accepted ? block.Tokens[j] : outcome.NextToken;
                tokens.Add(emitted);
                produced++;
                if (stopTokens is not null && stopTokens.Contains(emitted)) return produced;
            }
        }
        return produced;
    }

    private static float[][] Rows(int count, int width)
    {
        float[][] rows = new float[count][];
        for (int i = 0; i < count; i++) rows[i] = new float[width];
        return rows;
    }
}
