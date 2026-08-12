using Microsoft.ML.Tokenizers;

namespace HartsyInference.ModelAssets.Tokenizers;

/// <summary>Gemma SentencePiece tokenizer (256k vocab) — the conditioning-encoder tokenizer for Gemma-family
/// text encoders: LTX-2's Gemma 3 12B and Lumina-2's Gemma 2. Wraps Microsoft.ML.Tokenizers'
/// <see cref="SentencePieceTokenizer"/> over Gemma's <c>tokenizer.model</c> protobuf (bundled with LTX-2 at
/// <c>tokenizer/tokenizer.model</c>, or in the Gemma repos).
///
/// <para>Gemma prepends <c>&lt;bos&gt;</c> (id 2) and does NOT append <c>&lt;eos&gt;</c> for encoder
/// conditioning. Special ids: pad=0, eos=1, bos=2, unk=3. Unlike T5 (bos_id=-1, which needs a protobuf
/// patch), Gemma's bos_id is a real token, so no patch is needed.</para>
///
/// <para>The exact max length / whether the diffusion model right-pads is model-specific (validation-gated
/// against the LTX-2 reference); callers pass <paramref name="maxLength"/> and assemble the final sequence.</para></summary>
public sealed class GemmaTokenizer : IDisposable, ILtx2PromptTokenizer
{
    /// <summary>Padding token id (&lt;pad&gt; = 0).</summary>
    public const int PadTokenId = 0;

    /// <summary>End-of-sequence token id (&lt;eos&gt; = 1).</summary>
    public const int EosTokenId = 1;

    /// <summary>Beginning-of-sequence token id (&lt;bos&gt; = 2). Prepended automatically by the SentencePiece config.</summary>
    public const int BosTokenId = 2;

    /// <summary>Unknown token id (&lt;unk&gt; = 3).</summary>
    public const int UnkTokenId = 3;

    /// <summary>Default max sequence length for conditioning encodes.</summary>
    public const int DefaultMaxLength = 256;

    private readonly Tokenizer _tokenizer;
    private readonly int _maxLength;
    private int _disposed;

    /// <summary>Creates a Gemma tokenizer from a SentencePiece <c>.model</c> file (e.g. LTX-2's
    /// <c>tokenizer/tokenizer.model</c>).</summary>
    public GemmaTokenizer(string modelPath, int maxLength = DefaultMaxLength)
    {
        using Stream stream = File.OpenRead(modelPath);
        _tokenizer = SentencePieceTokenizer.Create(stream, addBeginningOfSentence: true, addEndOfSentence: false)
            ?? throw new InvalidOperationException("Failed to create Gemma SentencePiece tokenizer.");
        _maxLength = maxLength;
    }

    /// <summary>Creates a Gemma tokenizer from a SentencePiece <c>.model</c> protobuf stream.</summary>
    public GemmaTokenizer(Stream modelStream, int maxLength = DefaultMaxLength)
    {
        _tokenizer = SentencePieceTokenizer.Create(modelStream, addBeginningOfSentence: true, addEndOfSentence: false)
            ?? throw new InvalidOperationException("Failed to create Gemma SentencePiece tokenizer.");
        _maxLength = maxLength;
    }

    /// <summary>Encodes a prompt to Gemma token ids (BOS prepended via the SentencePiece config), truncated
    /// to <see cref="DefaultMaxLength"/>/the ctor max. No padding or EOS — the pipeline assembles the
    /// conditioning sequence (Gemma-family text encoders condition on BOS + content).</summary>
    /// <summary><see cref="Encode"/> already returns raw truncated ids with no padding, which is exactly the
    /// contract; the pipeline pads.</summary>
    public int[] EncodeForConditioning(string text) => Encode(text);

    /// <summary>0 — LTX-2.3 conditions at whatever register multiple the prompt lands on, and that path is verified.</summary>
    public int MinimumConditioningLength => 0;

    public int[] Encode(string text)
    {
        ThrowIfDisposed();
        IReadOnlyList<int> ids = _tokenizer.EncodeToIds(text);
        int count = Math.Min(ids.Count, _maxLength);
        int[] result = new int[count];
        for (int i = 0; i < count; i++)
        {
            result[i] = ids[i];
        }
        return result;
    }

    /// <summary>Tokenizes and returns the raw ids (BOS included, no truncation/padding).</summary>
    public IReadOnlyList<int> EncodeRaw(string text)
    {
        ThrowIfDisposed();
        return _tokenizer.EncodeToIds(text);
    }

    /// <summary>Decodes token ids back to text, dropping pad tokens.</summary>
    public string Decode(ReadOnlySpan<int> tokenIds)
    {
        ThrowIfDisposed();
        List<int> filtered = new(tokenIds.Length);
        for (int i = 0; i < tokenIds.Length; i++)
        {
            if (tokenIds[i] != PadTokenId)
            {
                filtered.Add(tokenIds[i]);
            }
        }
        return _tokenizer.Decode(filtered) ?? string.Empty;
    }

    private void ThrowIfDisposed()
    {
        if (Volatile.Read(ref _disposed) != 0)
            throw new ObjectDisposedException(nameof(GemmaTokenizer));
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            (_tokenizer as IDisposable)?.Dispose();
        }
    }
}
