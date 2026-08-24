using System.Text;
using System.Text.RegularExpressions;

namespace HartsyInference.ModelAssets.Tokenizers;

/// <summary>A byte-level BPE tokenizer built entirely from a GGUF file's embedded tokenizer metadata (<c>tokenizer.ggml.tokens</c> + <c>merges</c> + <c>token_type</c> + bos/eos ids) — no external tokenizer files needed. Covers the byte-level BPE family (Llama-3, Qwen, Mistral-v3, DeepSeek, GPT-OSS). SentencePiece GGUFs (Gemma, Llama-2) are a follow-up.
///
/// <para>Implements GPT-2/Llama byte-level BPE directly (regex pre-tokenization → byte-level remap → rank-based pair merges → vocab lookup) rather than going through a general tokenizer library, so word-boundary spaces and newline runs encode correctly (matching llama.cpp). Special/control tokens (type CONTROL or USER_DEFINED) are split out before BPE so a rendered chat template's literals map to their ids, and skipped on decode. Byte-level reversal via <see cref="ByteLevelCodec"/>.</para></summary>
public sealed class GgufTokenizer : ILlmTokenizer
{
    private const int TypeControl = 3;
    private const int TypeUserDefined = 4;

    // GPT-2 byte-level pre-token regex (the default): contraction suffixes, optional-leading-space letter/
    // number/symbol runs, and whitespace runs (so a word keeps its leading space as Ġ and newline runs
    // survive). Newer families (Llama-3) use a different split regex — pass it to the constructor.
    private static readonly Regex DefaultPreTokenRegex = new(
        @"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
        RegexOptions.Compiled);

    private readonly string[] _tokens;                       // id → byte-level token string
    private readonly Dictionary<string, int> _tokenToId;     // byte-level token string → id
    private readonly Dictionary<string, int> _mergeRank;     // "left right" → rank (priority)
    private readonly Dictionary<string, int> _specialByLiteral;
    private readonly string[] _specialLiterals;              // sorted by length desc (longest-match)
    private readonly HashSet<int> _specialIds;
    private readonly Regex _preTokenRegex;                   // family-specific pre-token split
    private readonly bool _ignoreMerges;                     // emit a whole pre-token directly if it's in vocab

    public int? BosId { get; }
    public int? EosId { get; }
    public string? BosToken { get; }
    public string? EosToken { get; }
    public IReadOnlyList<int> StopIds { get; }

    /// <summary>Builds from GGUF metadata arrays. <paramref name="tokens"/> is the full vocab (index = id); <paramref name="merges"/> are "left right" byte-level pairs in rank order; <paramref name="tokenType"/> (optional) marks control/user-defined tokens; <paramref name="extraStopIds"/> adds end-of-turn ids. <paramref name="preTokenizerRegex"/> overrides the default GPT-2 split regex (Llama-3 differs in digit grouping and contraction casing); <paramref name="ignoreMerges"/> mirrors HF's <c>ignore_merges</c> — a pre-token that is itself a vocab entry is emitted directly instead of being re-derived by BPE.</summary>
    public GgufTokenizer(string[] tokens, string[]? merges, int[]? tokenType,
        int? bosId, int? eosId, IReadOnlyList<int>? extraStopIds = null,
        string? preTokenizerRegex = null, bool ignoreMerges = false)
    {
        _preTokenRegex = preTokenizerRegex is null ? DefaultPreTokenRegex
            : new Regex(preTokenizerRegex, RegexOptions.Compiled);
        _ignoreMerges = ignoreMerges;
        if (tokens is null || tokens.Length == 0) throw new ArgumentException("GGUF tokens array is empty.", nameof(tokens));
        if (merges is null || merges.Length == 0)
            throw new NotSupportedException("GGUF has no BPE merges (tiktoken/SentencePiece-style); Phase 1 supports byte-level BPE only.");

        _tokens = tokens;
        _tokenToId = new Dictionary<string, int>(tokens.Length, StringComparer.Ordinal);
        for (int i = 0; i < tokens.Length; i++) _tokenToId.TryAdd(tokens[i], i);

        _mergeRank = new Dictionary<string, int>(merges.Length, StringComparer.Ordinal);
        for (int i = 0; i < merges.Length; i++) _mergeRank.TryAdd(merges[i], i);

        _specialByLiteral = new Dictionary<string, int>(StringComparer.Ordinal);
        _specialIds = [];
        for (int i = 0; i < tokens.Length; i++)
        {
            bool isSpecial = tokenType is not null && i < tokenType.Length
                ? tokenType[i] == TypeControl || tokenType[i] == TypeUserDefined : LooksSpecial(tokens[i]);
            // Conversion fixup: some GGUFs (leafspark mllama) mis-type template tokens as NORMAL, so
            // Decode's special-skip missed them and `<|eot_id|>`-family markers leaked into generated
            // text. Tokens shaped EXACTLY like `<|...|>` are template machinery in every supported vocab
            // (llama-3, ChatML, qwen, deepseek) — register them regardless of the declared type. Strictly
            // narrower than LooksSpecial (`<div>`/`[INST]`-shaped content tokens are NOT affected).
            if (!isSpecial) isSpecial = LooksTemplateSpecial(tokens[i]);
            if (isSpecial && tokens[i].Length > 0)
            {
                _specialByLiteral[tokens[i]] = i;
                _specialIds.Add(i);
            }
        }
        _specialLiterals = _specialByLiteral.Keys.OrderByDescending(s => s.Length).ToArray();

        BosId = bosId;
        EosId = eosId;
        BosToken = bosId is int b && b >= 0 && b < tokens.Length ? tokens[b] : null;
        EosToken = eosId is int e && e >= 0 && e < tokens.Length ? tokens[e] : null;

        HashSet<int> stops = [];
        if (eosId is int eid) stops.Add(eid);
        if (extraStopIds is not null) foreach (int s in extraStopIds) stops.Add(s);
        foreach (string lit in new[] { "<|eot_id|>", "<|eom_id|>", "<|im_end|>", "<|end|>", "<end_of_turn>" })
            if (_specialByLiteral.TryGetValue(lit, out int id)) stops.Add(id);
        StopIds = stops.ToArray();
    }

    public int? SpecialId(string token) => _specialByLiteral.TryGetValue(token, out int id) ? id : null;

    public int[] EncodeOrdinary(string text)
    {
        if (text.Length == 0) return [];
        List<int> ids = [];
        foreach (Match m in _preTokenRegex.Matches(text))
            EncodePieceBpe(ByteLevelCodec.Encode(m.Value), ids);
        return [.. ids];
    }

    public int[] Encode(string text, bool addSpecial)
    {
        if (!addSpecial || _specialLiterals.Length == 0) return EncodeOrdinary(text);

        List<int> ids = new(text.Length / 3 + 8);
        int i = 0;
        while (i < text.Length)
        {
            string? matched = MatchSpecialAt(text, i);
            if (matched is not null) { ids.Add(_specialByLiteral[matched]); i += matched.Length; continue; }
            int next = NextSpecialIndex(text, i);
            ids.AddRange(EncodeOrdinary(text[i..next]));
            i = next;
        }
        return [.. ids];
    }

    public string Decode(IReadOnlyList<int> ids)
    {
        StringBuilder raw = new();
        foreach (int id in ids)
        {
            if (id < 0 || id >= _tokens.Length || _specialIds.Contains(id)) continue;
            raw.Append(_tokens[id]);
        }
        return ByteLevelCodec.Decode(raw.ToString());
    }

    /// <summary>Standard BPE on one byte-level pre-token: start from single characters, repeatedly merge the adjacent pair with the lowest merge rank, then map the resulting pieces to ids.</summary>
    private void EncodePieceBpe(string piece, List<int> ids)
    {
        if (piece.Length == 0) return;
        // HF ignore_merges: a pre-token that is itself a vocab entry maps to that id without re-deriving it.
        if (_ignoreMerges && _tokenToId.TryGetValue(piece, out int wholeId)) { ids.Add(wholeId); return; }
        List<string> symbols = new(piece.Length);
        foreach (char c in piece) symbols.Add(c.ToString());

        while (symbols.Count > 1)
        {
            int bestRank = int.MaxValue, bestIdx = -1;
            for (int i = 0; i < symbols.Count - 1; i++)
            {
                if (_mergeRank.TryGetValue(symbols[i] + " " + symbols[i + 1], out int rank) && rank < bestRank)
                {
                    bestRank = rank; bestIdx = i;
                }
            }
            if (bestIdx < 0) break;
            symbols[bestIdx] += symbols[bestIdx + 1];
            symbols.RemoveAt(bestIdx + 1);
        }

        foreach (string sym in symbols)
            if (_tokenToId.TryGetValue(sym, out int id)) ids.Add(id);
            // A byte-level char not in vocab should never happen (all 256 byte glyphs are tokens); skip if so.
    }

    private string? MatchSpecialAt(string text, int i)
    {
        foreach (string lit in _specialLiterals)
            if (i + lit.Length <= text.Length && string.CompareOrdinal(text, i, lit, 0, lit.Length) == 0)
                return lit;
        return null;
    }

    private int NextSpecialIndex(string text, int from)
    {
        int best = text.Length;
        foreach (string lit in _specialLiterals)
        {
            int idx = text.IndexOf(lit, from, StringComparison.Ordinal);
            if (idx >= 0 && idx < best) best = idx;
        }
        return best;
    }

    private static bool LooksSpecial(string token)
        => token.Length >= 2 && ((token[0] == '<' && token[^1] == '>') || (token[0] == '[' && token[^1] == ']'));

    /// <summary>Strict template-token shape: <c>&lt;|...|&gt;</c> with no nested angle brackets or pipes.</summary>
    private static bool LooksTemplateSpecial(string token)
    {
        if (token.Length < 4 || token[0] != '<' || token[1] != '|' || token[^2] != '|' || token[^1] != '>') return false;
        for (int i = 2; i < token.Length - 2; i++)
            if (token[i] == '<' || token[i] == '>' || token[i] == '|') return false;
        return true;
    }
}
