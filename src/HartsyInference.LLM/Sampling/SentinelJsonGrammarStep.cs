using HartsyInference.Tokenizers;

namespace HartsyInference.LLM.Sampling;

/// <summary>Grammar-masks output to valid JSON only for the span that follows a literal text sentinel (e.g. a
/// prompted <c>&lt;tool_call&gt;</c> tag) — everywhere else, generation is completely unconstrained plain
/// text. Unlike <see cref="JsonGrammarStep"/> (which masks the entire response from token 0), this lets a
/// model answer normally in natural language and only pays the JSON-grammar cost/constraint for the one
/// structured blob it's asked to emit — matching how every major chat API (Claude tool_use, OpenAI
/// tool_calls, self-hosted grammar-constrained servers) scopes structured output to just the tool-call
/// arguments rather than the whole reply.
///
/// <para><b>Activation</b>: while inactive, a small rolling tail of recently decoded text (bounded to twice
/// the sentinel's length) is checked for the sentinel; once it matches, a fresh <see cref="JsonGrammarState"/>
/// starts constraining every subsequent token.</para>
///
/// <para><b>Deactivation</b>: happens automatically the instant the constrained JSON value completes
/// (<see cref="JsonGrammarState.IsComplete"/>) — not by matching a closing tag, since the model literally
/// cannot type a closing tag like <c>&lt;/tool_call&gt;</c> while still masked to valid JSON (<c>&lt;</c> is
/// not a valid JSON continuation). The caller's own prompt convention is expected to close the tag itself
/// once masking lifts.</para></summary>
public sealed class SentinelJsonGrammarStep : ISamplerStep
{
    private readonly string[] _vocabText;
    private readonly string _openSentinel;
    private readonly int _tailCap;
    private JsonGrammarState? _state; // null = inactive, no masking applied
    private string _tail = string.Empty;
    private int _repliedHistoryCount;

    public SentinelJsonGrammarStep(ILlmTokenizer tokenizer, int vocabSize, string openSentinel)
    {
        if (string.IsNullOrEmpty(openSentinel))
        {
            throw new ArgumentException("Sentinel must be non-empty.", nameof(openSentinel));
        }
        _vocabText = JsonVocabText.GetOrBuild(tokenizer, vocabSize);
        _openSentinel = openSentinel;
        _tailCap = openSentinel.Length * 2;
    }

    /// <inheritdoc/>
    public void Apply(Span<float> logits, IReadOnlyList<int> history)
    {
        for (int i = _repliedHistoryCount; i < history.Count; i++)
        {
            string text = TokenText(history[i]);
            if (_state is null)
            {
                _tail += text;
                if (_tail.Length > _tailCap) _tail = _tail[^_tailCap..];
                if (_tail.EndsWith(_openSentinel, StringComparison.Ordinal))
                {
                    _state = new JsonGrammarState();
                    _tail = string.Empty;
                }
                continue;
            }
            if (!_state.TryFeed(text))
            {
                // A previously-accepted token turned out to break the grammar — should not happen if every
                // prior step was masked correctly. Defensive: go inactive rather than corrupt state further.
                _state = null;
                continue;
            }
            if (_state.IsComplete)
            {
                _state = null; // lift the mask immediately so the model is free to close its own tag next
            }
        }
        _repliedHistoryCount = history.Count;

        if (_state is null) return; // plain text — no constraint
        JsonGrammarState state = _state;
        int n = Math.Min(logits.Length, _vocabText.Length);
        for (int id = 0; id < n; id++)
        {
            if (!state.Clone().TryFeed(TokenText(id))) logits[id] = float.NegativeInfinity;
        }
    }

    private string TokenText(int id) => (uint)id < (uint)_vocabText.Length ? _vocabText[id] : string.Empty;
}
