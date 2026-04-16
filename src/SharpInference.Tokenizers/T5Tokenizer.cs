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
        MemoryStream patched = PatchT5ProtobufStream(stream);
        _tokenizer = SentencePieceTokenizer.Create(patched, addBeginningOfSentence: false, addEndOfSentence: false) ?? throw new InvalidOperationException("Failed to create SentencePiece tokenizer.");
        _maxLength = maxLength;
    }

    /// <summary>Creates a T5 tokenizer from a stream.</summary>
    /// <param name="modelStream">Stream containing the SentencePiece .model protobuf.</param>
    /// <param name="maxLength">Maximum sequence length. Default: 77.</param>
    public T5Tokenizer(Stream modelStream, int maxLength = DefaultMaxLength)
    {
        MemoryStream patched = PatchT5ProtobufStream(modelStream);
        _tokenizer = SentencePieceTokenizer.Create(patched, addBeginningOfSentence: false, addEndOfSentence: false) ?? throw new InvalidOperationException("Failed to create SentencePiece tokenizer.");
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

    /// <summary>Patches T5 SentencePiece protobuf to work around bos_id=-1 causing IndexOutOfRangeException in Microsoft.ML.Tokenizers. T5 models set bos_id=-1 in TrainerSpec meaning "no BOS token", but the library interprets -1 as a vocabulary index.</summary>
    private static MemoryStream PatchT5ProtobufStream(Stream source)
    {
        MemoryStream ms = new MemoryStream();
        source.CopyTo(ms);
        byte[] data = ms.GetBuffer();
        long length = ms.Length;

        // Find protobuf field tag for TrainerSpec.bos_id (field 41, wire type 0 = varint)
        // Tag encoding: (41 << 3) | 0 = 328 → varint bytes 0xC8, 0x02
        // Value -1 as varint: 0xFF 0xFF 0xFF 0xFF 0xFF 0xFF 0xFF 0xFF 0xFF 0x01 (10 bytes)
        // Patch: rename field 41 to unused field 99 → (99 << 3) | 0 = 792 → 0x98, 0x06
        for (long i = 0; i < length - 11; i++)
        {
            if (data[i] == 0xC8 && data[i + 1] == 0x02 &&
                data[i + 2] == 0xFF && data[i + 3] == 0xFF &&
                data[i + 4] == 0xFF && data[i + 5] == 0xFF &&
                data[i + 6] == 0xFF && data[i + 7] == 0xFF &&
                data[i + 8] == 0xFF && data[i + 9] == 0xFF &&
                data[i + 10] == 0xFF && data[i + 11] == 0x01)
            {
                data[i] = 0x98;
                data[i + 1] = 0x06;
                break;
            }
        }

        ms.Position = 0;
        return ms;
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
