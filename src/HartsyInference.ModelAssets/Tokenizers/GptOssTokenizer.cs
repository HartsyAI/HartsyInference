using System.Text.RegularExpressions;
using Microsoft.ML.Tokenizers;

namespace HartsyInference.ModelAssets.Tokenizers;

/// <summary>OpenAI GPT-OSS byte-level BPE tokenizer (o200k_harmony) used as the text-encoder front-end for Microsoft Lens. Wraps <see cref="BpeTokenizer"/> with <c>vocab.json</c> + <c>merges.txt</c> exported from <c>openai/gpt-oss-20b</c>'s <c>tokenizer.json</c>. Base vocab 199,998 + 21 added special tokens (199998–200018) = 201,088-slot embedding (with reserved gaps). Encoding follows the upstream <c>tokenizers</c> pipeline exactly: split on special-token markers → o200k pre-tokenizer regex → GPT-2 byte-level remap → BPE merges per pre-token.
///
/// <para><b>Lens chat template</b> (verbatim from the reference pipeline): the full Harmony render — system preamble ("You are ChatGPT, …", knowledge cutoff, pinned date, reasoning level, valid channels), developer block with Lens' image-description instruction, user prompt, assistant <c>analysis</c> turn with the fixed thinking text, and a trailing assistant <c>final</c> header. The wrapper up to and including <c>&lt;|start|&gt;user&lt;|message|&gt;</c> tokenizes to exactly 97 tokens (<c>DEFAULT_TXT_OFFSET = 97</c>); the pipeline strips those 97 positions off the encoder hidden states. <see cref="BuildChatInputs"/> validates that invariant at first use and fails fast if the vocab/template drift.</para></summary>
public sealed class GptOssTokenizer : IDisposable
{
    /// <summary>Embedding-table size per <c>microsoft/Lens/text_encoder/config.json</c> (includes reserved special-token slots).</summary>
    public const int VocabSize = 201088;

    /// <summary>Lens image-description instruction (the Harmony <b>developer</b> block's content). Verbatim from the reference pipeline.</summary>
    public const string ChatSystemPrompt =
        "Describe the image by detailing the color, shape, size, texture, " +
        "quantity, text, spatial relationships of the objects and background.";

    /// <summary>Assistant thinking content Lens uses as the encoder-side prefix. Verbatim from the reference pipeline.</summary>
    public const string ChatAssistantThinking = "Need to generate one image according to the description.";

    /// <summary>Pinned "Current date" the reference pipeline bakes into the Harmony system preamble. Must stay fixed — it is part of the 97-token wrapper the encoder output is offset by.</summary>
    public const string ChatTemplateDate = "2026-05-23";

    /// <summary>Token count of the chat-template wrapper up to and including <c>&lt;|start|&gt;user&lt;|message|&gt;</c> — the offset <see cref="HartsyInference.Diffusion.Pipelines.LensPipeline"/> strips off the encoder hidden states. Constant per upstream pipeline (<c>DEFAULT_TXT_OFFSET</c>).</summary>
    public const int DefaultTxtOffset = 97;

    /// <summary>o200k pre-tokenizer split pattern, translated from the upstream <c>tokenizer.json</c> Split regex.</summary>
    private static readonly Regex _o200kPreTokenRegex = new(
        "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?" +
        "|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?" +
        "|\\p{N}{1,3}" +
        "| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*" +
        "|\\s*[\\r\\n]+" +
        "|\\s+(?!\\S)" +
        "|\\s+",
        RegexOptions.Compiled | RegexOptions.CultureInvariant);

    /// <summary>Matches any input as a single pre-token — the inner <see cref="BpeTokenizer"/> must never re-split the pieces produced by <see cref="_o200kPreTokenRegex"/>.</summary>
    private static readonly Regex _matchWholeInputRegex = new("[\\s\\S]+", RegexOptions.Compiled);

    /// <summary>o200k_harmony added tokens (ids 199998–200018). These live OUTSIDE <c>vocab.json</c> (the HF export keeps them in <c>added_tokens</c>), so they are matched literally before BPE.</summary>
    private static readonly Dictionary<string, int> _specialTokens = new()
    {
        ["<|startoftext|>"] = 199998,
        ["<|endoftext|>"] = 199999,
        ["<|reserved_200000|>"] = 200000,
        ["<|reserved_200001|>"] = 200001,
        ["<|return|>"] = 200002,
        ["<|constrain|>"] = 200003,
        ["<|reserved_200004|>"] = 200004,
        ["<|channel|>"] = 200005,
        ["<|start|>"] = 200006,
        ["<|end|>"] = 200007,
        ["<|message|>"] = 200008,
        ["<|reserved_200009|>"] = 200009,
        ["<|reserved_200010|>"] = 200010,
        ["<|reserved_200011|>"] = 200011,
        ["<|call|>"] = 200012,
        ["<|reserved_200013|>"] = 200013,
        ["<|reserved_200014|>"] = 200014,
        ["<|reserved_200015|>"] = 200015,
        ["<|reserved_200016|>"] = 200016,
        ["<|reserved_200017|>"] = 200017,
        ["<|endofprompt|>"] = 200018,
    };

    private readonly Tokenizer _tokenizer;
    private readonly int _maxLength;
    private readonly int _padTokenId;
    private int _wrapperTokenCount = -1;
    private int _disposed;

    /// <summary>Creates a GPT-OSS tokenizer from <c>vocab.json</c> + <c>merges.txt</c>. No embedded resource — GPT-OSS vocab is ~10 MB and is exported alongside the Lens checkpoint.</summary>
    /// <param name="vocabPath">Path to the exported base vocab (199,998 entries; special tokens excluded).</param>
    /// <param name="mergesPath">Path to <c>merges.txt</c>.</param>
    /// <param name="maxLength">Truncation cap. Default 512 matches <c>LensPipeline.__call__.max_sequence_length</c>.</param>
    /// <param name="padTokenId">Pad token id used when a caller batches uneven rows. Reference pads with <c>&lt;|endoftext|&gt;</c> = 199999.</param>
    public GptOssTokenizer(string vocabPath, string mergesPath, int maxLength = 512, int padTokenId = 199999)
    {
        if (maxLength <= 0)
            throw new ArgumentOutOfRangeException(nameof(maxLength));
        _maxLength = maxLength;
        _padTokenId = padTokenId;
        using Stream vocabStream = File.OpenRead(vocabPath);
        using Stream mergesStream = File.OpenRead(mergesPath);
        // Pre-tokenization is done by _o200kPreTokenRegex in EncodePlainSegment, so the inner BPE must
        // treat each piece as ONE word. `preTokenizer: null` silently falls back to ML.Tokenizers'
        // whitespace-or-punctuation default, which re-splits byte-level pieces (".Ċ" → "." + "Ċ") and
        // drops every punctuation↔letter merge — hence the explicit match-everything pre-tokenizer.
        _tokenizer = BpeTokenizer.Create(
            vocabStream,
            mergesStream,
            preTokenizer: new RegexPreTokenizer(_matchWholeInputRegex, null),
            normalizer: null,
            specialTokens: null,
            unknownToken: null,
            continuingSubwordPrefix: null,
            endOfWordSuffix: null,
            fuseUnknownTokens: false);
    }

    /// <summary>Pad token id (<c>&lt;|endoftext|&gt;</c> for the reference pipeline).</summary>
    public int PadTokenId => _padTokenId;

    /// <summary>Renders the Lens Harmony chat template around a user prompt. Verbatim structure from the reference pipeline — every byte matters because the encoder output is offset by the wrapper's fixed 97-token length.</summary>
    public static string RenderChatTemplate(string prompt) =>
        "<|start|>system<|message|>" +
        "You are ChatGPT, a large language model trained by OpenAI.\n" +
        "Knowledge cutoff: 2024-06\n" +
        $"Current date: {ChatTemplateDate}\n\n" +
        "Reasoning: medium\n\n" +
        "# Valid channels: analysis, commentary, final. " +
        "Channel must be included for every message.<|end|>" +
        "<|start|>developer<|message|># Instructions\n\n" +
        $"{ChatSystemPrompt}\n\n<|end|>" +
        $"<|start|>user<|message|>{prompt}<|end|>" +
        "<|start|>assistant<|channel|>analysis<|message|>" +
        $"{ChatAssistantThinking}<|end|>" +
        "<|start|>assistant<|channel|>final<|message|>";

    /// <summary>Encodes a user prompt through Lens' Harmony chat template. Returns the true-length token ids (truncated to the max length; NO padding — the reference tokenizes without padding and the engine pipeline runs each prompt at its natural length) plus an all-ones attention mask of the same length. Fails fast if the fixed wrapper does not tokenize to exactly <see cref="DefaultTxtOffset"/> tokens (vocab/merges drift would silently mis-align the encoder offset otherwise).</summary>
    public (int[] tokenIds, int[] attentionMask) BuildChatInputs(string prompt)
    {
        ThrowIfDisposed();
        ValidateWrapperTokenCount();

        List<int> ids = EncodeWithSpecials(RenderChatTemplate(prompt));
        int realLen = Math.Min(ids.Count, _maxLength);
        int[] tokenIds = new int[realLen];
        int[] mask = new int[realLen];
        for (int i = 0; i < realLen; i++)
        {
            tokenIds[i] = ids[i];
            mask[i] = 1;
        }
        return (tokenIds, mask);
    }

    /// <summary>Encodes raw text without applying the chat template (special-token markers in the text ARE matched). Useful for tokenizer round-trip tests against the upstream Python loader.</summary>
    public IReadOnlyList<int> EncodeRaw(string text)
    {
        ThrowIfDisposed();
        return EncodeWithSpecials(text);
    }

    /// <summary>Splits on special-token markers, then runs the o200k pre-tokenizer regex + GPT-2 byte-level remap + BPE on each plain segment — mirroring the upstream <c>tokenizers</c> pipeline stage-for-stage.</summary>
    private List<int> EncodeWithSpecials(string text)
    {
        List<int> ids = new(text.Length / 3 + 8);
        int pos = 0;
        while (pos < text.Length)
        {
            int nextSpecial = -1;
            int nextSpecialId = 0;
            int nextSpecialLen = 0;
            foreach (KeyValuePair<string, int> kvp in _specialTokens)
            {
                int idx = text.IndexOf(kvp.Key, pos, StringComparison.Ordinal);
                if (idx >= 0 && (nextSpecial < 0 || idx < nextSpecial))
                {
                    nextSpecial = idx;
                    nextSpecialId = kvp.Value;
                    nextSpecialLen = kvp.Key.Length;
                }
            }

            if (nextSpecial < 0)
            {
                EncodePlainSegment(text.AsSpan(pos), ids);
                break;
            }

            if (nextSpecial > pos)
                EncodePlainSegment(text.AsSpan(pos, nextSpecial - pos), ids);
            ids.Add(nextSpecialId);
            pos = nextSpecial + nextSpecialLen;
        }
        return ids;
    }

    /// <summary>Regex pre-tokenize → byte-level remap → per-piece BPE. Running BPE per regex match is what makes the merge boundaries match upstream (merges never cross pre-token boundaries).</summary>
    private void EncodePlainSegment(ReadOnlySpan<char> segment, List<int> ids)
    {
        if (segment.IsEmpty) return;
        string s = segment.ToString();
        foreach (Match m in _o200kPreTokenRegex.Matches(s))
        {
            string byteLevel = ByteLevelCodec.Encode(m.Value);
            IReadOnlyList<int> pieceIds = _tokenizer.EncodeToIds(byteLevel);
            for (int i = 0; i < pieceIds.Count; i++)
                ids.Add(pieceIds[i]);
        }
    }

    /// <summary>Tokenizes the template wrapper once and asserts it is exactly <see cref="DefaultTxtOffset"/> tokens — the pipeline strips that many positions off the encoder output, so drift here corrupts conditioning silently.</summary>
    private void ValidateWrapperTokenCount()
    {
        if (_wrapperTokenCount >= 0)
        {
            if (_wrapperTokenCount != DefaultTxtOffset)
                throw new InvalidOperationException(
                    $"GPT-OSS chat-template wrapper tokenizes to {_wrapperTokenCount} tokens; Lens requires exactly {DefaultTxtOffset}.");
            return;
        }

        string rendered = RenderChatTemplate(string.Empty);
        const string userMarker = "<|start|>user<|message|>";
        string prefix = rendered[..(rendered.IndexOf(userMarker, StringComparison.Ordinal) + userMarker.Length)];
        _wrapperTokenCount = EncodeWithSpecials(prefix).Count;
        if (_wrapperTokenCount != DefaultTxtOffset)
            throw new InvalidOperationException(
                $"GPT-OSS chat-template wrapper tokenizes to {_wrapperTokenCount} tokens; Lens requires exactly {DefaultTxtOffset}. " +
                "The exported vocab.json/merges.txt likely do not match openai/gpt-oss-20b's tokenizer.json.");
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
