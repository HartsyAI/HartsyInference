using Microsoft.ML.Tokenizers;

namespace HartsyInference.ModelAssets.Tokenizers;

/// <summary>IndexTTS SentencePiece tokenizer (12k vocab, Unigram model type despite the upstream filename <c>bpe.model</c> — verified against the real checkpoint's <c>trainer_spec.model_type</c>). No BOS/EOS: IndexTTS inserts explicit <c>start_text_token</c>/<c>stop_text_token</c> sentinel ids itself (0/1 for IndexTTS-1.5), outside the subword vocabulary proper.</summary>
public sealed class IndexTtsTokenizer : IDisposable
{
    private readonly Tokenizer _tokenizer;
    private int _disposed;

    public IndexTtsTokenizer(string modelPath)
    {
        using Stream stream = File.OpenRead(modelPath);
        _tokenizer = SentencePieceTokenizer.Create(stream, addBeginningOfSentence: false, addEndOfSentence: false)
            ?? throw new InvalidOperationException("Failed to create IndexTTS SentencePiece tokenizer.");
    }

    public IndexTtsTokenizer(Stream modelStream)
    {
        _tokenizer = SentencePieceTokenizer.Create(modelStream, addBeginningOfSentence: false, addEndOfSentence: false)
            ?? throw new InvalidOperationException("Failed to create IndexTTS SentencePiece tokenizer.");
    }

    /// <summary>Encodes text to subword ids. <see cref="IndexTtsTextNormalizer.InjectCjkBoundaries"/> should run on
    /// the input first so CJK runs don't merge with adjacent Latin text/punctuation in the Unigram segmentation.</summary>
    public int[] Encode(string text)
    {
        ThrowIfDisposed();
        IReadOnlyList<int> ids = _tokenizer.EncodeToIds(text);
        int[] result = new int[ids.Count];
        for (int i = 0; i < ids.Count; i++) result[i] = ids[i];
        return result;
    }

    /// <summary>Encodes text to <c>(piece, id)</c> pairs — the reference's <c>tokenizer.tokenize</c> output (pieces keep
    /// their <c>▁</c> word-start marker), needed by the token-level segment splitter, which keys on piece text
    /// (<c>"."</c>, <c>"▁."</c>, <c>","</c>…) rather than ids.</summary>
    public IReadOnlyList<(string Piece, int Id)> EncodeToPieces(string text)
    {
        ThrowIfDisposed();
        IReadOnlyList<EncodedToken> tokens = _tokenizer.EncodeToTokens(text, out _);
        List<(string, int)> result = new(tokens.Count);
        foreach (EncodedToken t in tokens) result.Add((t.Value, t.Id));
        return result;
    }

    public string Decode(ReadOnlySpan<int> tokenIds)
    {
        ThrowIfDisposed();
        // Tokenizer.Decode takes IEnumerable<int>; a span can't implement that, so materialize it first.
        return _tokenizer.Decode(tokenIds.ToArray()) ?? string.Empty;
    }

    private void ThrowIfDisposed()
    {
        if (Volatile.Read(ref _disposed) != 0) throw new ObjectDisposedException(nameof(IndexTtsTokenizer));
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            (_tokenizer as IDisposable)?.Dispose();
        }
    }
}
