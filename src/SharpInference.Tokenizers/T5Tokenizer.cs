using Microsoft.ML.Tokenizers;

namespace SharpInference.Tokenizers;

/// <summary>T5 SentencePiece tokenizer for SD3 and Flux text encoding. Wraps Microsoft.ML.Tokenizers SentencePieceTokenizer with T5-specific conventions (EOS token, space prefix, no BOS).</summary>
public sealed class T5Tokenizer : IDisposable
{
    /// <summary>Padding token ID (T5 pad_token_id = 0).</summary>
    public const int PadTokenId = 0;

    /// <summary>End-of-sequence token ID (T5 eos_token_id = 1, represents &lt;/s&gt;).</summary>
    public const int EosTokenId = 1;

    /// <summary>Unknown token ID (T5 unk_token_id = 2).</summary>
    public const int UnkTokenId = 2;

    /// <summary>Default max sequence length for SD3/Flux T5 encoding.</summary>
    public const int DefaultMaxLength = 77;

    private readonly Tokenizer _tokenizer;
    private readonly int _maxLength;
    private int _disposed;

    /// <summary>Creates a T5 tokenizer from a SentencePiece .model file.</summary>
    /// <param name="modelPath">Path to the SentencePiece .model protobuf file.</param>
    /// <param name="maxLength">Maximum sequence length. Default: 77.</param>
    public T5Tokenizer(string modelPath, int maxLength = DefaultMaxLength)
    {
        using Stream stream = File.OpenRead(modelPath);
        _tokenizer = SentencePieceTokenizer.Create(stream, addBeginningOfSentence: false, addEndOfSentence: false) ?? throw new InvalidOperationException("Failed to create SentencePiece tokenizer.");
        _maxLength = maxLength;
    }

    /// <summary>Creates a T5 tokenizer from a stream.</summary>
    /// <param name="modelStream">Stream containing the SentencePiece .model protobuf.</param>
    /// <param name="maxLength">Maximum sequence length. Default: 77.</param>
    public T5Tokenizer(Stream modelStream, int maxLength = DefaultMaxLength)
    {
        _tokenizer = SentencePieceTokenizer.Create(modelStream, addBeginningOfSentence: false, addEndOfSentence: false) ?? throw new InvalidOperationException("Failed to create SentencePiece tokenizer.");
        _maxLength = maxLength;
    }

    /// <summary>Tokenizes text into token IDs with T5 conventions. Appends EOS, pads to maxLength.</summary>
    /// <param name="text">Input text to tokenize.</param>
    /// <returns>Array of token IDs with length maxLength.</returns>
    public int[] Encode(string text)
    {
        ThrowIfDisposed();

        // T5 adds a space prefix to all input (SentencePiece convention)
        IReadOnlyList<int> tokenIds = _tokenizer.EncodeToIds(text);

        // Build final sequence: [...tokens..., EOS, PAD, PAD, ...]
        int[] result = new int[_maxLength];

        int maxTokens = _maxLength - 1; // Reserve one slot for EOS
        int tokenCount = Math.Min(tokenIds.Count, maxTokens);

        for (int i = 0; i < tokenCount; i++)
        {
            result[i] = tokenIds[i];
        }

        result[tokenCount] = EosTokenId;

        // Fill remaining with pad tokens
        for (int i = tokenCount + 1; i < _maxLength; i++)
        {
            result[i] = PadTokenId;
        }

        return result;
    }

    /// <summary>Tokenizes text and returns just the token IDs without padding or EOS.</summary>
    public IReadOnlyList<int> EncodeRaw(string text)
    {
        ThrowIfDisposed();
        return _tokenizer.EncodeToIds(text);
    }

    /// <summary>Decodes token IDs back to text.</summary>
    public string Decode(ReadOnlySpan<int> tokenIds)
    {
        ThrowIfDisposed();

        // Filter out PAD and EOS tokens
        List<int> filtered = new List<int>(tokenIds.Length);
        for (int i = 0; i < tokenIds.Length; i++)
        {
            int id = tokenIds[i];
            if (id != PadTokenId && id != EosTokenId)
            {
                filtered.Add(id);
            }
        }

        return _tokenizer.Decode(filtered) ?? string.Empty;
    }

    /// <summary>Creates an attention mask for the encoded tokens (1 for real tokens, 0 for padding).</summary>
    public static int[] CreateAttentionMask(ReadOnlySpan<int> tokenIds)
    {
        int[] mask = new int[tokenIds.Length];
        for (int i = 0; i < tokenIds.Length; i++)
        {
            mask[i] = tokenIds[i] != PadTokenId ? 1 : 0;
        }
        return mask;
    }

    private void ThrowIfDisposed()
    {
        if (Volatile.Read(ref _disposed) != 0)
            throw new ObjectDisposedException(nameof(T5Tokenizer));
    }

    /// <summary>Disposes the underlying tokenizer resources.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            if (_tokenizer is IDisposable disposable)
            {
                disposable.Dispose();
            }
        }
    }
}
