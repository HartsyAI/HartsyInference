using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.LLM.Sampling;

/// <summary>Hard structural constraint that masks every candidate token whose decoded text would make the
/// accumulated output an invalid prefix of some JSON document — the model can only ever emit syntactically
/// valid JSON. One instance is scoped to a single generation (constructed fresh per request by
/// <see cref="SamplerChain.FromOptions"/>, mirroring every other sampler step); its internal
/// <see cref="JsonGrammarState"/> advances incrementally as tokens are committed, so re-validating already-
/// accepted history is O(1) amortized rather than re-parsing from scratch every step.
///
/// <para><b>Cost</b>: unlike the elementwise steps (temperature, min-p, ...), this one is O(vocab size) per
/// generated token — for each candidate id it clones the current grammar state and trial-feeds that token's
/// decoded text. This is the accepted cost of constrained decoding in general (llama.cpp's GBNF grammar
/// sampler has the same complexity); it only runs when <see cref="SamplingOptions.JsonMode"/> is set, so it
/// never affects unconstrained requests.</para>
///
/// <para><b>Vocab-text cache</b>: decoding every token id to text is done ONCE per tokenizer instance (i.e.
/// once per loaded model, not once per request) via <see cref="JsonVocabText"/>, shared with
/// <see cref="SentinelJsonGrammarStep"/> — the table is immutable per-model data, safe to share across
/// concurrently-served requests.</para></summary>
public sealed class JsonGrammarStep : ISamplerStep
{
    private readonly string[] _vocabText;
    private readonly HashSet<int> _stopIds;
    private readonly JsonGrammarState _state = new();
    private int _repliedHistoryCount;

    public JsonGrammarStep(ILlmTokenizer tokenizer, int vocabSize)
    {
        _vocabText = JsonVocabText.GetOrBuild(tokenizer, vocabSize);
        _stopIds = [.. tokenizer.StopIds];
    }

    /// <inheritdoc/>
    public void Apply(Span<float> logits, IReadOnlyList<int> history)
    {
        // Replay tokens committed since the last call (SamplerChain.Next is called once per generated
        // token with a monotonically-growing history) — advances the REAL state incrementally.
        for (int i = _repliedHistoryCount; i < history.Count; i++)
        {
            if (!_state.TryFeed(TokenText(history[i])))
            {
                // A previously-accepted token turned out to break the grammar — should not happen if every
                // prior step was masked correctly. Defensive: stop constraining rather than corrupt state
                // further or throw mid-generation.
                _repliedHistoryCount = history.Count;
                return;
            }
        }
        _repliedHistoryCount = history.Count;

        bool complete = _state.IsComplete;
        int n = Math.Min(logits.Length, _vocabText.Length);
        for (int id = 0; id < n; id++)
        {
            if (complete && _stopIds.Contains(id)) continue; // always allow stopping once the JSON is complete
            if (!_state.Clone().TryFeed(TokenText(id))) logits[id] = float.NegativeInfinity;
        }
    }

    private string TokenText(int id) => (uint)id < (uint)_vocabText.Length ? _vocabText[id] : string.Empty;
}
