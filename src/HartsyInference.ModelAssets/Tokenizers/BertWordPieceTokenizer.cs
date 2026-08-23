using System.IO;
using Microsoft.ML.Tokenizers;

namespace HartsyInference.ModelAssets.Tokenizers;

/// <summary>BERT WordPiece tokenizer (bert-base-multilingual-cased) — the text front-end Bark consumes. Bark encodes text with NO special tokens, then shifts every id by <c>TEXT_ENCODING_OFFSET</c> before the semantic stage (the shift is applied by the caller / AudioTextFrontend, not here). Cased by default to match Bark's multilingual-cased vocab. Loads the standard BERT <c>vocab.txt</c> (one token per line); the caller supplies it (the vocab is not embedded).</summary>
public sealed class BertWordPieceTokenizer : IDisposable
{
    private readonly BertTokenizer _tokenizer;

    public BertWordPieceTokenizer(Stream vocab, bool lowerCase = false)
    {
        _tokenizer = BertTokenizer.Create(vocab, new BertOptions { LowerCaseBeforeTokenization = lowerCase });
    }

    public BertWordPieceTokenizer(string vocabPath, bool lowerCase = false)
    {
        if (!File.Exists(vocabPath))
        {
            throw new FileNotFoundException($"BERT vocab.txt not found: {vocabPath}", vocabPath);
        }
        using FileStream fs = File.OpenRead(vocabPath);
        _tokenizer = BertTokenizer.Create(fs, new BertOptions { LowerCaseBeforeTokenization = lowerCase });
    }

    /// <summary>Encodes text to raw WordPiece ids — no <c>[CLS]</c>/<c>[SEP]</c> (Bark's convention).</summary>
    public IReadOnlyList<int> EncodeRaw(string text)
        => _tokenizer.EncodeToIds(text ?? "", addSpecialTokens: false);

    /// <summary>Encodes text to WordPiece ids WITH <c>[CLS]</c>/<c>[SEP]</c> (standard BERT input).</summary>
    public IReadOnlyList<int> EncodeWithSpecial(string text)
        => _tokenizer.EncodeToIds(text ?? "", addSpecialTokens: true);

    /// <summary>Returns the WordPiece token strings (continuation pieces keep their <c>##</c> prefix), no specials — used to group sub-tokens into words (MeloTTS g2p).</summary>
    public IReadOnlyList<string> Tokens(string text)
    {
        IReadOnlyList<EncodedToken> toks = _tokenizer.EncodeToTokens(text ?? "", out _);
        List<string> outList = new(toks.Count);
        foreach (EncodedToken tk in toks)
        {
            string v = tk.Value;
            if (v is "[CLS]" or "[SEP]" or "[PAD]" or "[UNK]" or "[MASK]") continue; // drop specials; keep ## pieces
            outList.Add(v);
        }
        return outList;
    }

    public void Dispose() { }
}
