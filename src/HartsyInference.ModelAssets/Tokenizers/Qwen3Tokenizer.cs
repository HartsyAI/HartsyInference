using Microsoft.ML.Tokenizers;
using System.Text;

namespace HartsyInference.ModelAssets.Tokenizers;

/// <summary>Qwen3 byte-level BPE tokenizer used by image-model text conditioning. The embedded production path loads the canonical tokenizer.json so Qwen's split regex, byte mapping, normalization, and added tokens match Hugging Face. The explicit vocab/merges constructor remains as a compatibility fallback. Vocab size is 151,936; <c>BosTokenId</c> = 151643 (<c>&lt;|endoftext|&gt;</c>); <c>EosTokenId</c> = 151645 (<c>&lt;|im_end|&gt;</c>).</summary>
public sealed class Qwen3Tokenizer : IDisposable
{
    /// <summary>Vocabulary size (matches Qwen3-4B's <c>config.json</c>).</summary>
    public const int VocabSize = 151936;

    /// <summary>Beginning-of-sequence token id (<c>&lt;|endoftext|&gt;</c>).</summary>
    public const int BosTokenId = 151643;

    /// <summary>End-of-sequence token id (<c>&lt;|im_end|&gt;</c>).</summary>
    public const int EosTokenId = 151645;

    private static readonly Lazy<GgufTokenizer> EmbeddedExactTokenizer = new(() =>
    {
        using Stream jsonStream = EmbeddedTokenizerResources.OpenQwen3TokenizerJson();
        return HfTokenizerJson.LoadByteLevelBpe(jsonStream);
    }, LazyThreadSafetyMode.ExecutionAndPublication);

    private readonly Tokenizer? _fallbackTokenizer;
    private readonly GgufTokenizer? _exactTokenizer;
    private readonly int _maxLength;
    private int _disposed;

    // Chat-template special ids resolved from the loaded artifact's added-token table. The canonical
    // constants describe the embedded Qwen3-4B tokenizer only; a caller-supplied tokenizer.json may lay
    // its added tokens out differently, and hardcoding would silently emit foreign ids into its stream.
    private readonly int _bosTokenId = BosTokenId;
    private readonly int _eosTokenId = EosTokenId;
    private readonly int _imStartId = ImStartId;
    private readonly int _imEndId = ImEndId;
    private readonly int _thinkStartId = ThinkStartId;
    private readonly int _thinkEndId = ThinkEndId;

    /// <summary>Resolves one chat special id from the exact tokenizer's added-token table, keeping the canonical Qwen3 id when the literal is present at its canonical position or the artifact omits it (an omission is logged once — the canonical id is then a best-effort guess, not a contract).</summary>
    private static int ResolveSpecialId(GgufTokenizer exact, string literal, int canonicalId)
    {
        int? resolved = exact.SpecialId(literal);
        if (resolved is null)
        {
            HartsyInference.Core.Logging.Logs.Warning(
                $"[Qwen3Tokenizer] loaded tokenizer.json has no added token '{literal}'; " +
                $"falling back to the canonical Qwen3 id {canonicalId}, which may not match this vocabulary.");
            return canonicalId;
        }
        return resolved.Value;
    }

    /// <summary>Creates a Qwen3 tokenizer using the canonical <c>Qwen/Qwen3-4B</c> tokenizer.json embedded in this assembly. This is the right constructor for Flux.2 Klein and Z-Image text conditioning. Use the path overload only if you need to override with a non-standard Qwen3 vocabulary.</summary>
    /// <param name="maxLength">Truncation cap. Default 512 (matches typical diffusion text-encoder windows; Qwen3 itself supports up to 40,960).</param>
    public Qwen3Tokenizer(int maxLength = 512)
    {
        if (maxLength <= 0)
            throw new ArgumentOutOfRangeException(nameof(maxLength));
        _maxLength = maxLength;
        if (EmbeddedTokenizerResources.HasQwen3TokenizerJson)
        {
            // tokenizer.json carries Qwen's family-specific split regex and added-token table. The two-file
            // BPE wrapper cannot reproduce either exactly, even after applying the byte-level mapping.
            _exactTokenizer = EmbeddedExactTokenizer.Value;
        }
        else
        {
            using Stream vocabStream = EmbeddedTokenizerResources.OpenQwen3Vocab();
            using Stream mergesStream = EmbeddedTokenizerResources.OpenQwen3Merges();
            _fallbackTokenizer = BpeTokenizer.Create(vocabStream, mergesStream);
        }
    }

    /// <summary>Creates a Qwen3 tokenizer from <c>vocab.json</c> and <c>merges.txt</c> files (download from <c>Qwen/Qwen3-4B</c> on Hugging Face). When the files share a directory containing <c>tokenizer.json</c>, that complete artifact is used so the split regex and added tokens remain exact; otherwise this falls back to the inherently less-complete two-file BPE representation.</summary>
    /// <param name="vocabPath">Path to <c>vocab.json</c>.</param>
    /// <param name="mergesPath">Path to <c>merges.txt</c>.</param>
    /// <param name="maxLength">Truncation cap. Default 512 (matches typical diffusion text-encoder windows; Qwen3 itself supports up to 40,960).</param>
    public Qwen3Tokenizer(string vocabPath, string mergesPath, int maxLength = 512)
    {
        if (maxLength <= 0)
            throw new ArgumentOutOfRangeException(nameof(maxLength));
        _maxLength = maxLength;

        string vocabFullPath = Path.GetFullPath(vocabPath);
        string mergesFullPath = Path.GetFullPath(mergesPath);
        if (!File.Exists(vocabFullPath))
            throw new FileNotFoundException("Qwen3 vocabulary file was not found.", vocabFullPath);
        if (!File.Exists(mergesFullPath))
            throw new FileNotFoundException("Qwen3 merges file was not found.", mergesFullPath);
        string? vocabDirectory = Path.GetDirectoryName(vocabFullPath);
        string? mergesDirectory = Path.GetDirectoryName(mergesFullPath);
        if (vocabDirectory is not null
            && string.Equals(vocabDirectory, mergesDirectory, StringComparison.Ordinal)
            && File.Exists(Path.Combine(vocabDirectory, "tokenizer.json")))
        {
            using Stream jsonStream = File.OpenRead(Path.Combine(vocabDirectory, "tokenizer.json"));
            _exactTokenizer = HfTokenizerJson.LoadByteLevelBpe(jsonStream);
            _bosTokenId = ResolveSpecialId(_exactTokenizer, "<|endoftext|>", BosTokenId);
            _eosTokenId = ResolveSpecialId(_exactTokenizer, "<|im_end|>", EosTokenId);
            _imStartId = ResolveSpecialId(_exactTokenizer, "<|im_start|>", ImStartId);
            _imEndId = _eosTokenId;
            _thinkStartId = ResolveSpecialId(_exactTokenizer, "<think>", ThinkStartId);
            _thinkEndId = ResolveSpecialId(_exactTokenizer, "</think>", ThinkEndId);
            return;
        }

        using Stream vocabStream = File.OpenRead(vocabFullPath);
        using Stream mergesStream = File.OpenRead(mergesFullPath);
        _fallbackTokenizer = BpeTokenizer.Create(vocabStream, mergesStream);
    }

    /// <summary>Encodes text into a fixed-length <see cref="int"/> array of token ids, padded with <c>BosTokenId</c> on the right (Qwen3 has no dedicated pad token; using BOS/<c>&lt;|endoftext|&gt;</c> matches HF Tokenizers' default for Qwen3 instruct models). The first non-pad slots hold the BPE-encoded text; an EOS token is appended unless the prompt fills the entire window.</summary>
    /// <param name="text">Input prompt.</param>
    /// <param name="appendEos">Whether to append <c>EosTokenId</c> after the last real token (default true).</param>
    /// <returns>Array of length <c>maxLength</c>.</returns>
    public int[] Encode(string text, bool appendEos = true)
    {
        ThrowIfDisposed();
        IReadOnlyList<int> tokenIds = EncodeBpe(text);

        int[] result = new int[_maxLength];
        // Pad with BOS (Qwen3's <|endoftext|>) — same convention as HF Qwen3 tokenizer when no
        // explicit pad_token is set.
        for (int i = 0; i < _maxLength; i++)
            result[i] = _bosTokenId;

        int reserveForEos = appendEos ? 1 : 0;
        int tokenCount = Math.Min(tokenIds.Count, _maxLength - reserveForEos);
        for (int i = 0; i < tokenCount; i++)
            result[i] = tokenIds[i];

        if (appendEos && tokenCount < _maxLength)
            result[tokenCount] = _eosTokenId;

        return result;
    }

    /// <summary>Encodes text and returns the raw token ids without padding/truncation/EOS.</summary>
    public IReadOnlyList<int> EncodeRaw(string text)
    {
        ThrowIfDisposed();
        return EncodeBpe(text);
    }

    /// <summary>Token id for <c>&lt;|im_start|&gt;</c>.</summary>
    public const int ImStartId = 151644;

    /// <summary>Token id for <c>&lt;|im_end|&gt;</c>.</summary>
    public const int ImEndId = 151645;

    /// <summary>Token id for Qwen3's assistant reasoning-block opener <c>&lt;think&gt;</c>.</summary>
    public const int ThinkStartId = 151667;

    /// <summary>Token id for Qwen3's assistant reasoning-block closer <c>&lt;/think&gt;</c>.</summary>
    public const int ThinkEndId = 151668;

    /// <summary>Encodes a single user prompt using the Qwen3 chat template (matches <c>apply_chat_template([{role:"user",content:prompt}], add_generation_prompt=True)</c>). The diffusion text encoder receives chat-formatted hidden states; raw prompt encoding produces wrong conditioning.
    /// <para>Two model families differ in the assistant generation prefix:</para>
    /// <list type="bullet">
    /// <item><b>Flux.2 Klein</b> (Qwen3 text, <paramref name="includeThinkBlock"/> = true, default): <c>…&lt;|im_start|&gt;assistant\n&lt;think&gt;\n\n&lt;/think&gt;\n\n</c> (the <c>enable_thinking=False</c> rendering of the Qwen3 text template).</item>
    /// <item><b>Z-Image</b> (Qwen3 text, <paramref name="includeThinkBlock"/> = false): upstream requests <c>enable_thinking=True</c>, so the generation prompt stops at <c>…&lt;|im_start|&gt;assistant\n</c>; no empty reasoning block is prefilled.</item>
    /// <item><b>Ideogram 4</b> (Qwen3-VL-8B-Instruct, <paramref name="includeThinkBlock"/> = false): <c>…&lt;|im_start|&gt;assistant\n</c> with NO think block — the VL-Instruct template emits no <c>&lt;think&gt;</c> tags and no default system message (verified against <c>Qwen/Qwen3-VL-8B-Instruct</c> tokenizer_config + upstream <c>pipeline_ideogram4.py</c>).</item>
    /// </list>
    /// Format: <c>&lt;|im_start|&gt;user\n{prompt}&lt;|im_end|&gt;\n&lt;|im_start|&gt;assistant\n[think]</c>. Right-padded with <see cref="BosTokenId"/> to maxLength. Note that the upstream boolean is inverted relative to this API: <c>enable_thinking=false</c> emits the empty think block, while <c>true</c> does not.</summary>
    public int[] EncodeChat(string prompt, bool includeThinkBlock = true) =>
        EncodeChatWithLength(prompt, includeThinkBlock).TokenIds;

    /// <summary>Encodes the same chat template as <see cref="EncodeChat"/> and also returns the exact number of real tokens before right-padding. Consumers that slice text-encoder hidden states must use this metadata rather than infer length from token values: <see cref="BosTokenId"/> is both legal content and the pad id, so a real trailing <c>&lt;|endoftext|&gt;</c> token is otherwise indistinguishable from padding.</summary>
    /// <returns>The fixed-length token array and its real prefix length, capped at the configured max length.</returns>
    public (int[] TokenIds, int RealLength) EncodeChatWithLength(string prompt, bool includeThinkBlock = true)
    {
        ThrowIfDisposed();

        List<int> ids = new(_maxLength);
        ids.Add(_imStartId);
        // This is one ordinary-text span in the rendered template. Keeping it whole is load-bearing when
        // the prompt begins with whitespace (for example, "\ncat" must merge the two adjacent newlines to
        // Qwen token 271); separate BPE calls create an artificial pre-tokenization boundary.
        AppendBpe(ids, string.Concat("user\n", prompt));
        ids.Add(_imEndId);
        AppendBpe(ids, "\n");
        ids.Add(_imStartId);
        AppendBpe(ids, "assistant\n");
        if (includeThinkBlock)
        {
            // <think> and </think> are added tokens, not entries in the base vocab/merges files. Encoding
            // their literal text as ordinary BPE silently produces multiple punctuation/word tokens.
            ids.Add(_thinkStartId);
            AppendBpe(ids, "\n\n");
            ids.Add(_thinkEndId);
            AppendBpe(ids, "\n\n");
        }

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
        for (int i = realLen; i < _maxLength; i++) result[i] = _bosTokenId; // pad with <|endoftext|>
        return (result, realLen);
    }

    private void AppendBpe(List<int> dst, string text)
    {
        IReadOnlyList<int> ids = EncodeBpe(text);
        for (int i = 0; i < ids.Count; i++) dst.Add(ids[i]);
    }

    private IReadOnlyList<int> EncodeBpe(string text)
    {
        string normalized = text.IsNormalized(NormalizationForm.FormC)
            ? text
            : text.Normalize(NormalizationForm.FormC);

        if (_exactTokenizer is not null)
            // GgufTokenizer's addSpecial flag means "recognize literals already present in the text"; it
            // does not append a BOS/EOS post-processor. HF likewise recognizes Qwen added-token literals
            // even when its add_special_tokens option is false.
            return _exactTokenizer.Encode(normalized, addSpecial: true);

        // BpeTokenizer.Create(vocab, merges) does not install the GPT-2 byte-level pre-tokenizer. Keep the
        // explicit-path compatibility path byte-correct, although only tokenizer.json can also supply the
        // canonical Qwen split regex and full added-token table used by the embedded production path.
        return _fallbackTokenizer!.EncodeToIds(ByteLevelCodec.Encode(normalized));
    }

    /// <summary>Builds a best-effort [seq] attention mask: 1 through the last non-pad token, then 0 for the trailing <see cref="BosTokenId"/> run. For chat encoding, prefer <see cref="EncodeChatWithLength"/>: a real trailing <c>&lt;|endoftext|&gt;</c> token has the same id as padding and cannot be recovered from the padded array alone.</summary>
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
