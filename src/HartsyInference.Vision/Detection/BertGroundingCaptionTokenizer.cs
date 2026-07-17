using HartsyInference.Tokenizers;

namespace HartsyInference.Vision.Detection;

/// <summary>Default caption tokenizer for Grounding DINO — wraps <see cref="BertWordPieceTokenizer"/> (the same
/// WordPiece vocab the BERT text tower expects). The caption is split into phrases on '.', each phrase is
/// WordPiece-encoded, and the pieces are wrapped as <c>[CLS] phrase₀ [SEP] phrase₁ [SEP] … [SEP]</c> — the '.'
/// separators become <c>[SEP]</c> so each phrase occupies a contiguous, recoverable token span.</summary>
public sealed class BertGroundingCaptionTokenizer : IGroundingCaptionTokenizer
{
    private readonly BertWordPieceTokenizer _tokenizer;
    private readonly int _clsId;
    private readonly int _sepId;
    private readonly int _maxTokens;

    public BertGroundingCaptionTokenizer(BertWordPieceTokenizer tokenizer, int maxTokens = 256)
    {
        ArgumentNullException.ThrowIfNull(tokenizer);
        _tokenizer = tokenizer;
        _maxTokens = maxTokens;

        // Derive the [CLS]/[SEP] ids from an empty encode (yields exactly [CLS][SEP] for BERT); fall back to the
        // canonical bert-base ids if the tokenizer returns something unexpected.
        IReadOnlyList<int> specials = tokenizer.EncodeWithSpecial("");
        _clsId = specials.Count > 0 ? specials[0] : 101;
        _sepId = specials.Count > 1 ? specials[^1] : 102;
    }

    public GroundingCaptionTokens Tokenize(string caption)
    {
        List<int> ids = new() { _clsId };
        List<GroundingPhraseSpan> phrases = new();

        foreach (string rawPhrase in (caption ?? "").Split('.'))
        {
            string phrase = rawPhrase.Trim();
            if (phrase.Length == 0)
                continue;
            IReadOnlyList<int> pieces = _tokenizer.EncodeRaw(phrase);
            if (pieces.Count == 0)
                continue;
            if (ids.Count + pieces.Count + 1 > _maxTokens)
                break;
            int start = ids.Count;
            ids.AddRange(pieces);
            int end = ids.Count;
            phrases.Add(new GroundingPhraseSpan(phrase, start, end));
            ids.Add(_sepId);
        }

        return new GroundingCaptionTokens
        {
            TokenIds = ids.ToArray(),
            Phrases = phrases,
        };
    }
}
