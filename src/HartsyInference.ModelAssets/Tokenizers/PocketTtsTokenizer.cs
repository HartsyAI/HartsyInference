using System.IO;
using Microsoft.ML.Tokenizers;

namespace HartsyInference.ModelAssets.Tokenizers;

/// <summary>Pocket-TTS text tokenizer — the SentencePiece <c>tokenizer.model</c> shipped with the checkpoint
/// (vocab 4000). Mirrors upstream <c>sp.encode(text, out_type=int)</c> (no BOS/EOS); the ids index the FlowLM
/// <c>conditioner.embed</c> LUT.</summary>
public sealed class PocketTtsTokenizer
{
    private readonly SentencePieceTokenizer _sp;

    public PocketTtsTokenizer(string spmPath)
    {
        if (!File.Exists(spmPath))
        {
            throw new FileNotFoundException($"Pocket-TTS SentencePiece model not found: {spmPath}", spmPath);
        }
        using FileStream fs = File.OpenRead(spmPath);
        _sp = SentencePieceTokenizer.Create(fs, addBeginningOfSentence: false, addEndOfSentence: false)
            ?? throw new InvalidOperationException("Failed to create Pocket-TTS SentencePiece tokenizer.");
    }

    /// <summary>Vocabulary size (should be 4000 for the released checkpoint).</summary>
    public int VocabSize => (int)_sp.Vocabulary.Count;

    /// <summary>Encodes text to SentencePiece ids.</summary>
    public IReadOnlyList<int> Encode(string text) => _sp.EncodeToIds(text ?? "");
}
