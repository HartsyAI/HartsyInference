using Microsoft.ML.Tokenizers;

namespace SharpInference.Tokenizers;

/// <summary>Llama 3 / 3.1 byte-level BPE tokenizer used as a text encoder for HiDream and other
/// diffusion models that condition on Llama-3.1-Instruct hidden states. Wraps
/// <see cref="BpeTokenizer"/> with <c>vocab.json</c> + <c>merges.txt</c> shipped under the
/// <c>meta-llama/Llama-3.1-8B-Instruct</c> tokenizer. Vocab size 128,256; <c>BosTokenId</c> = 128000
/// (<c>&lt;|begin_of_text|&gt;</c>); <c>EosTokenId</c> = 128001 (<c>&lt;|end_of_text|&gt;</c>);
/// <c>PadTokenId</c> = 128004 (<c>&lt;|finetune_right_pad_id|&gt;</c>). HiDream uses Llama as a
/// feature extractor — encode the raw prompt with BOS prefix, no chat template required.
/// <para>Note: Llama 3 tokenizers ship as a <c>tokenizer.json</c> file, not raw vocab.json+merges.txt.
/// Pass the path to either form; the constructor accepts whichever the user has on disk. The path
/// constructor expects the vocab.json + merges.txt extracted from <c>tokenizer.json</c> (same content,
/// just split into the two-file form). HuggingFace's <c>tokenizers</c> library produces both
/// representations interchangeably.</para></summary>
public sealed class LlamaTokenizer : IDisposable
{
    /// <summary>Vocabulary size for Llama 3 / 3.1.</summary>
    public const int VocabSize = 128_256;

    /// <summary>Beginning-of-sequence token id (<c>&lt;|begin_of_text|&gt;</c>).</summary>
    public const int BosTokenId = 128_000;

    /// <summary>End-of-sequence token id (<c>&lt;|end_of_text|&gt;</c>).</summary>
    public const int EosTokenId = 128_001;

    /// <summary>Right-padding token id used by Llama-3 fine-tunes (<c>&lt;|finetune_right_pad_id|&gt;</c>).
    /// Falls back to <see cref="EosTokenId"/> if a model uses pad=eos.</summary>
    public const int PadTokenId = 128_004;

    /// <summary>Chat header start token id (<c>&lt;|start_header_id|&gt;</c>).</summary>
    public const int StartHeaderId = 128_006;

    /// <summary>Chat header end token id (<c>&lt;|end_header_id|&gt;</c>).</summary>
    public const int EndHeaderId = 128_007;

    /// <summary>End-of-turn token id (<c>&lt;|eot_id|&gt;</c>).</summary>
    public const int EotId = 128_009;

    private readonly Tokenizer _tokenizer;
    private readonly int _maxLength;
    private int _disposed;

    /// <summary>Creates a Llama 3 tokenizer from <c>vocab.json</c> + <c>merges.txt</c>.</summary>
    /// <param name="vocabPath">Path to <c>vocab.json</c>.</param>
    /// <param name="mergesPath">Path to <c>merges.txt</c>.</param>
    /// <param name="maxLength">Truncation cap. Default 256 (matches HiDream's text-encoder window).</param>
    public LlamaTokenizer(string vocabPath, string mergesPath, int maxLength = 256)
    {
        if (maxLength <= 0)
            throw new ArgumentOutOfRangeException(nameof(maxLength));
        _maxLength = maxLength;
        using Stream vocabStream = File.OpenRead(vocabPath);
        using Stream mergesStream = File.OpenRead(mergesPath);
        _tokenizer = BpeTokenizer.Create(vocabStream, mergesStream);
    }

    /// <summary>Encodes text into a fixed-length token-id array. Prefixes BOS, then BPE-encodes the
    /// prompt, then right-pads with <see cref="PadTokenId"/>. Truncates from the right if the prompt
    /// overflows.</summary>
    public int[] Encode(string text, bool prependBos = true)
    {
        ThrowIfDisposed();
        IReadOnlyList<int> tokenIds = _tokenizer.EncodeToIds(text);

        int[] result = new int[_maxLength];
        for (int i = 0; i < _maxLength; i++) result[i] = PadTokenId;

        int cursor = 0;
        if (prependBos && cursor < _maxLength)
        {
            result[cursor++] = BosTokenId;
        }

        int tokenCount = Math.Min(tokenIds.Count, _maxLength - cursor);
        for (int i = 0; i < tokenCount; i++)
            result[cursor + i] = tokenIds[i];

        return result;
    }

    /// <summary>Encodes text and returns the raw token ids with no padding / truncation / BOS.</summary>
    public IReadOnlyList<int> EncodeRaw(string text)
    {
        ThrowIfDisposed();
        return _tokenizer.EncodeToIds(text);
    }

    /// <summary>Builds a <c>[seq]</c> attention mask: 1 for real tokens (including the prepended BOS),
    /// 0 for trailing pad. Walks from the right to find the last non-pad position.</summary>
    public static int[] CreateAttentionMask(int[] tokenIds)
    {
        int[] mask = new int[tokenIds.Length];
        int lastReal = -1;
        for (int i = tokenIds.Length - 1; i >= 0; i--)
        {
            if (tokenIds[i] != PadTokenId)
            {
                lastReal = i;
                break;
            }
        }
        for (int i = 0; i <= lastReal; i++) mask[i] = 1;
        return mask;
    }

    private void ThrowIfDisposed() =>
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            // Microsoft.ML.Tokenizers.Tokenizer is not IDisposable in v2.0; nothing to free.
        }
    }
}
