using Microsoft.ML.Tokenizers;

namespace HartsyInference.Tokenizers;

/// <summary>Qwen3 byte-level BPE tokenizer used for Flux.2 Klein text conditioning. Wraps <see cref="BpeTokenizer"/> with vocab.json + merges.txt from <c>Qwen/Qwen3-4B</c>. Vocab size 151,936; <c>BosTokenId</c> = 151643 (<c>&lt;|endoftext|&gt;</c>); <c>EosTokenId</c> = 151645 (<c>&lt;|im_end|&gt;</c>). Encode raw prompts as-is — Klein's text encoder runs Qwen3 as a feature extractor.</summary>
public sealed class Qwen3Tokenizer : IDisposable
{
    /// <summary>Vocabulary size (matches Qwen3-4B's <c>config.json</c>).</summary>
    public const int VocabSize = 151936;

    /// <summary>Beginning-of-sequence token id (<c>&lt;|endoftext|&gt;</c>).</summary>
    public const int BosTokenId = 151643;

    /// <summary>End-of-sequence token id (<c>&lt;|im_end|&gt;</c>).</summary>
    public const int EosTokenId = 151645;

    private readonly Tokenizer _tokenizer;
    private readonly int _maxLength;
    private int _disposed;

    /// <summary>Creates a Qwen3 tokenizer using the canonical <c>Qwen/Qwen3-4B</c>
    /// vocab/merges embedded in this assembly. This is the right constructor for
    /// Flux.2 Klein and Z-Image text conditioning. Use the path overload only if
    /// you need to override with a non-standard Qwen3 vocabulary.</summary>
    /// <param name="maxLength">Truncation cap. Default 512 (matches typical diffusion
    /// text-encoder windows; Qwen3 itself supports up to 40,960).</param>
    public Qwen3Tokenizer(int maxLength = 512)
    {
        if (maxLength <= 0)
            throw new ArgumentOutOfRangeException(nameof(maxLength));
        _maxLength = maxLength;
        using Stream vocabStream = EmbeddedTokenizerResources.OpenQwen3Vocab();
        using Stream mergesStream = EmbeddedTokenizerResources.OpenQwen3Merges();
        _tokenizer = BpeTokenizer.Create(vocabStream, mergesStream);
    }

    /// <summary>Creates a Qwen3 tokenizer from <c>vocab.json</c> and <c>merges.txt</c> files (download from <c>Qwen/Qwen3-4B</c> on HuggingFace).</summary>
    /// <param name="vocabPath">Path to <c>vocab.json</c>.</param>
    /// <param name="mergesPath">Path to <c>merges.txt</c>.</param>
    /// <param name="maxLength">Truncation cap. Default 512 (matches typical diffusion text-encoder windows; Qwen3 itself supports up to 40,960).</param>
    public Qwen3Tokenizer(string vocabPath, string mergesPath, int maxLength = 512)
    {
        if (maxLength <= 0)
            throw new ArgumentOutOfRangeException(nameof(maxLength));
        _maxLength = maxLength;
        using Stream vocabStream = File.OpenRead(vocabPath);
        using Stream mergesStream = File.OpenRead(mergesPath);
        _tokenizer = BpeTokenizer.Create(vocabStream, mergesStream);
    }

    /// <summary>Encodes text into a fixed-length <see cref="int"/> array of token ids, padded with <c>BosTokenId</c> on the right (Qwen3 has no dedicated pad token; using BOS/<c>&lt;|endoftext|&gt;</c> matches HF Tokenizers' default for Qwen3 instruct models). The first non-pad slots hold the BPE-encoded text; an EOS token is appended unless the prompt fills the entire window.</summary>
    /// <param name="text">Input prompt.</param>
    /// <param name="appendEos">Whether to append <c>EosTokenId</c> after the last real token (default true).</param>
    /// <returns>Array of length <c>maxLength</c>.</returns>
    public int[] Encode(string text, bool appendEos = true)
    {
        ThrowIfDisposed();
        IReadOnlyList<int> tokenIds = _tokenizer.EncodeToIds(text);

        int[] result = new int[_maxLength];
        // Pad with BOS (Qwen3's <|endoftext|>) — same convention as HF Qwen3 tokenizer when no
        // explicit pad_token is set.
        for (int i = 0; i < _maxLength; i++)
            result[i] = BosTokenId;

        int reserveForEos = appendEos ? 1 : 0;
        int tokenCount = Math.Min(tokenIds.Count, _maxLength - reserveForEos);
        for (int i = 0; i < tokenCount; i++)
            result[i] = tokenIds[i];

        if (appendEos && tokenCount < _maxLength)
            result[tokenCount] = EosTokenId;

        return result;
    }

    /// <summary>Encodes text and returns the raw token ids without padding/truncation/EOS.</summary>
    public IReadOnlyList<int> EncodeRaw(string text)
    {
        ThrowIfDisposed();
        // ML.Tokenizers' BpeTokenizer.Create(vocab, merges) does NOT apply the GPT-2 byte-level pre-tokenizer,
        // so it drops leading spaces (" there" -> "there") instead of mapping them to the Ġ-prefixed vocab keys
        // ("Ġthere"). Pre-encode the text into GPT-2 byte space so the byte-level vocab/merges match exactly.
        return _tokenizer.EncodeToIds(ByteLevelCodec.Encode(text));
    }

    /// <summary>Token id for <c>&lt;|im_start|&gt;</c>.</summary>
    public const int ImStartId = 151644;

    /// <summary>Token id for <c>&lt;|im_end|&gt;</c>.</summary>
    public const int ImEndId = 151645;

    /// <summary>Encodes a single user prompt using the Qwen3 chat template (matches <c>apply_chat_template([{role:"user",content:prompt}], add_generation_prompt=True)</c>). The diffusion text encoder receives chat-formatted hidden states; raw prompt encoding produces wrong conditioning.
    /// <para>Two model families differ in the assistant generation prefix:</para>
    /// <list type="bullet">
    ///   <item><b>Flux.2 Klein</b> (Qwen3 text, <paramref name="includeThinkBlock"/> = true, default): <c>…&lt;|im_start|&gt;assistant\n&lt;think&gt;\n\n&lt;/think&gt;\n\n</c> (the <c>enable_thinking=False</c> rendering of the Qwen3 text template).</item>
    ///   <item><b>Ideogram 4</b> (Qwen3-VL-8B-Instruct, <paramref name="includeThinkBlock"/> = false): <c>…&lt;|im_start|&gt;assistant\n</c> with NO think block — the VL-Instruct template emits no <c>&lt;think&gt;</c> tags and no default system message (verified against <c>Qwen/Qwen3-VL-8B-Instruct</c> tokenizer_config + upstream <c>pipeline_ideogram4.py</c>).</item>
    /// </list>
    /// Format: <c>&lt;|im_start|&gt;user\n{prompt}&lt;|im_end|&gt;\n&lt;|im_start|&gt;assistant\n[think]</c>. Right-padded with <see cref="BosTokenId"/> to maxLength.</summary>
    public int[] EncodeChat(string prompt, bool includeThinkBlock = true)
    {
        ThrowIfDisposed();

        List<int> ids = new(_maxLength);
        ids.Add(ImStartId);
        AppendBpe(ids, "user\n");
        AppendBpe(ids, prompt);
        ids.Add(ImEndId);
        AppendBpe(ids, "\n");
        ids.Add(ImStartId);
        AppendBpe(ids, includeThinkBlock ? "assistant\n<think>\n\n</think>\n\n" : "assistant\n");

        int realLen = ids.Count;
        if (realLen > _maxLength)
        {
            // Truncate from the right (drop trailing tokens including assistant prefix). Diffusers
            // does the same: `truncation=True, max_length=N`.
            ids.RemoveRange(_maxLength, realLen - _maxLength);
            realLen = _maxLength;
        }

        int[] result = new int[_maxLength];
        for (int i = 0; i < realLen; i++) result[i] = ids[i];
        for (int i = realLen; i < _maxLength; i++) result[i] = BosTokenId; // pad with <|endoftext|>
        return result;
    }

    private void AppendBpe(List<int> dst, string text)
    {
        // TODO(byte-level): this path does NOT byte-level encode (unlike EncodeRaw, which routes through
        // ByteLevelCodec.Encode). ML.Tokenizers' BpeTokenizer.Create(vocab, merges) drops leading spaces
        // (" there" -> "there" instead of "Ġthere"), so EncodeChat produces wrong tokens for the image
        // models (Flux.2 Klein, Ideogram4). Verified against the Qwen3-TTS C reference that the byte-level
        // fix is required for exact parity. Apply ByteLevelCodec.Encode here too and re-validate Klein/
        // Ideogram4 conditioning (their output WILL change), then remove this note.
        IReadOnlyList<int> ids = _tokenizer.EncodeToIds(text);
        for (int i = 0; i < ids.Count; i++) dst.Add(ids[i]);
    }

    /// <summary>Builds a [seq] attention mask: 1 for real tokens (including the appended EOS), 0 for the BOS padding.</summary>
    public static int[] CreateAttentionMask(int[] tokenIds)
    {
        int[] mask = new int[tokenIds.Length];
        // Find the last non-pad position. Pad token is BOS (<|endoftext|>=151643) which is also a
        // valid sentence opener — so we walk from the end and treat the trailing run of BOS as pad.
        // The appended EOS=151645 is always treated as a real token.
        int lastReal = -1;
        for (int i = tokenIds.Length - 1; i >= 0; i--)
        {
            if (tokenIds[i] != BosTokenId)
            {
                lastReal = i;
                break;
            }
        }
        for (int i = 0; i <= lastReal; i++)
            mask[i] = 1;
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
