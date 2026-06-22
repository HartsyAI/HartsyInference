using System.IO;
using Microsoft.ML.Tokenizers;

namespace HartsyInference.Tokenizers;

/// <summary>Kyutai STT text-vocab decoder — turns the SentencePiece text token ids
/// <c>KyutaiSttPipeline.Transcribe</c> emits into text. The Audio package carries no text-vocab dependency,
/// so the caller supplies the model's SentencePiece <c>text.model</c> (8k en_fr / en). Decode is what STT
/// needs; Encode is provided for symmetry.</summary>
public sealed class KyutaiSttTokenizer : IDisposable
{
    private readonly SentencePieceTokenizer _sp;

    public KyutaiSttTokenizer(Stream spm)
    {
        _sp = SentencePieceTokenizer.Create(spm, addBeginningOfSentence: false, addEndOfSentence: false)
            ?? throw new InvalidOperationException("Failed to create Kyutai SentencePiece tokenizer.");
    }

    public KyutaiSttTokenizer(string spmPath)
    {
        if (!File.Exists(spmPath))
        {
            throw new FileNotFoundException($"Kyutai SentencePiece model not found: {spmPath}", spmPath);
        }
        using FileStream fs = File.OpenRead(spmPath);
        _sp = SentencePieceTokenizer.Create(fs, addBeginningOfSentence: false, addEndOfSentence: false)
            ?? throw new InvalidOperationException("Failed to create Kyutai SentencePiece tokenizer.");
    }

    /// <summary>Decodes STT text token ids to a string (SentencePiece ▁ word-boundary → spaces).</summary>
    public string Decode(IEnumerable<int> tokenIds) => _sp.Decode(tokenIds);

    /// <summary>Encodes text to ids (symmetry / testing).</summary>
    public IReadOnlyList<int> Encode(string text) => _sp.EncodeToIds(text ?? "");

    public void Dispose() { }
}
