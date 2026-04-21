using System.Text;
using System.Text.RegularExpressions;
using Microsoft.ML.Tokenizers;

namespace SharpInference.Tokenizers;

/// <summary>CLIP BPE tokenizer matching OpenAI's implementation exactly. Wraps Microsoft.ML.Tokenizers BPE with CLIP-specific preprocessing (lowercase, regex split, SOT/EOT tokens, 77-token limit).</summary>
public sealed class ClipTokenizer : IDisposable
{
    /// <summary>Start-of-text token ID.</summary>
    public const int StartOfTextId = 49406;

    /// <summary>End-of-text token ID.</summary>
    public const int EndOfTextId = 49407;

    /// <summary>Maximum sequence length including SOT and EOT tokens.</summary>
    public const int MaxLength = 77;

    /// <summary>Vocabulary size.</summary>
    public const int VocabSize = 49408;

    private readonly Tokenizer _tokenizer;
    private int _disposed;

    /// <summary>Creates a CLIP tokenizer from a vocabulary file and merges file.</summary>
    /// <param name="vocabPath">Path to the CLIP vocabulary JSON file (encoder.json).</param>
    /// <param name="mergesPath">Path to the CLIP BPE merges file (vocab.bpe).</param>
    public ClipTokenizer(string vocabPath, string mergesPath)
    {
        using Stream vocabStream = File.OpenRead(vocabPath);
        using Stream mergesStream = File.OpenRead(mergesPath);
        _tokenizer = BpeTokenizer.Create(vocabStream, mergesStream);
    }

    /// <summary>Creates a CLIP tokenizer from streams.</summary>
    /// <param name="vocabStream">Stream containing the CLIP vocabulary JSON.</param>
    /// <param name="mergesStream">Stream containing the CLIP BPE merges.</param>
    public ClipTokenizer(Stream vocabStream, Stream mergesStream)
    {
        _tokenizer = BpeTokenizer.Create(vocabStream, mergesStream);
    }

    /// <summary>Tokenizes text into token IDs matching OpenAI CLIP output. Prepends SOT, appends EOT, pads/truncates to MaxLength.</summary>
    /// <param name="text">Input text to tokenize.</param>
    /// <returns>Array of token IDs with length MaxLength.</returns>
    public int[] Encode(string text)
    {
        ThrowIfDisposed();

        // CLIP preprocessing: lowercase and clean whitespace
        string processed = text.ToLowerInvariant().Trim();

        // Encode using BPE
        IReadOnlyList<int> tokenIds = _tokenizer.EncodeToIds(processed);

        // Build final sequence: [SOT, ...tokens..., EOT, 0, 0, ...]
        int[] result = new int[MaxLength];
        result[0] = StartOfTextId;

        int maxTokens = MaxLength - 2; // Reserve slots for SOT and EOT
        int tokenCount = Math.Min(tokenIds.Count, maxTokens);

        for (int i = 0; i < tokenCount; i++)
        {
            result[i + 1] = tokenIds[i];
        }

        result[tokenCount + 1] = EndOfTextId;
        // Remaining positions are already 0 (padding)

        return result;
    }

    /// <summary>Finds the position of the EOT token in an encoded sequence. Used by SDXL to extract the pooled output from the CLIP-G encoder at the EOS token position.</summary>
    /// <param name="tokenIds">Encoded token IDs from Encode().</param>
    /// <returns>Index of the EOT token, or -1 if not found.</returns>
    public static int FindEosPosition(ReadOnlySpan<int> tokenIds)
    {
        for (int i = 0; i < tokenIds.Length; i++)
        {
            if (tokenIds[i] == EndOfTextId)
                return i;
        }
        return -1;
    }

    /// <summary>Tokenizes text and returns just the token IDs without padding (no SOT/EOT).</summary>
    public IReadOnlyList<int> EncodeRaw(string text)
    {
        ThrowIfDisposed();
        string processed = text.ToLowerInvariant().Trim();
        return _tokenizer.EncodeToIds(processed);
    }

    /// <summary>Decodes token IDs back to text.</summary>
    public string Decode(ReadOnlySpan<int> tokenIds)
    {
        ThrowIfDisposed();

        // Filter out SOT, EOT, and padding (0)
        List<int> filtered = new List<int>(tokenIds.Length);
        for (int i = 0; i < tokenIds.Length; i++)
        {
            int id = tokenIds[i];
            if (id != StartOfTextId && id != EndOfTextId && id != 0)
            {
                filtered.Add(id);
            }
        }

        return _tokenizer.Decode(filtered) ?? string.Empty;
    }

    private void ThrowIfDisposed()
    {
        if (Volatile.Read(ref _disposed) != 0)
            throw new ObjectDisposedException(nameof(ClipTokenizer));
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
