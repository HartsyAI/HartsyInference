namespace HartsyInference.LLM.Generation.Speculation;

/// <summary>Prompt-lookup drafting: finds the OLDEST earlier occurrence of the context's last <c>ngramSize</c> tokens
/// and drafts the tokens that followed it.</summary>
/// <remarks>This is the inline drafting that <c>TextGenerationPipeline</c> used before the speculation framework,
/// moved here unchanged: same n-gram size, same lookback window, same search order, so it proposes exactly the same tokens.
/// It searches OLDEST-match-first rather than nearest-match-first: the nearer a match is to the current position,
/// the less context trails it, so nearest-first degenerates to single-token drafts on short-period repeats
/// (a stuck "the the the..." loop), the case this technique helps most.
/// Returns an empty draft whenever no match exists. The lookback is capped so a long generation cannot turn the scan
/// into O(n²); missing a distant match only forgoes a speedup, it never affects correctness.</remarks>
public sealed class PromptLookupDraftProvider : ISpeculativeDraftProvider
{
    /// <summary>The n-gram size the pipeline has always used.</summary>
    public const int DefaultNgramSize = 3;

    /// <summary>The search window the pipeline has always used, in tokens.</summary>
    public const int DefaultMaxLookback = 4096;

    private readonly int _ngramSize;
    private readonly int _maxLookback;

    public PromptLookupDraftProvider(int ngramSize = DefaultNgramSize, int maxLookback = DefaultMaxLookback)
    {
        if (ngramSize < 1) throw new ArgumentOutOfRangeException(nameof(ngramSize), ngramSize, "The n-gram size must be at least 1.");
        if (maxLookback < 1) throw new ArgumentOutOfRangeException(nameof(maxLookback), maxLookback, "The lookback must be at least 1 token.");
        _ngramSize = ngramSize;
        _maxLookback = maxLookback;
    }

    /// <inheritdoc />
    public string Name => "prompt-lookup";

    /// <inheritdoc />
    public int[] Propose(int[] promptIds, IReadOnlyList<int> generated, int maxDraftLen)
    {
        ArgumentNullException.ThrowIfNull(promptIds);
        ArgumentNullException.ThrowIfNull(generated);
        int totalLen = promptIds.Length + generated.Count;
        if (maxDraftLen <= 0 || totalLen < _ngramSize) return [];

        int[] context = SpeculationWindow.Tail(promptIds, generated, _maxLookback);
        int windowLen = context.Length;
        int needleStart = windowLen - _ngramSize;
        for (int start = 0; start < needleStart; start++)
        {
            bool match = true;
            for (int k = 0; k < _ngramSize; k++)
            {
                if (context[start + k] != context[needleStart + k]) { match = false; break; }
            }
            if (!match) continue;

            int matchEnd = start + _ngramSize;
            int draftLen = Math.Min(maxDraftLen, windowLen - matchEnd);
            if (draftLen <= 0) continue;
            int[] draft = new int[draftLen];
            Array.Copy(context, matchEnd, draft, 0, draftLen);
            return draft;
        }
        return [];
    }
}
