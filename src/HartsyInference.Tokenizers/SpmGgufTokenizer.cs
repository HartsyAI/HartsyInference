using System.Text;

namespace HartsyInference.Tokenizers;

/// <summary>A SentencePiece (LLaMA/Gemma "SPM") tokenizer built entirely from a GGUF file's embedded tokenizer
/// metadata (<c>tokenizer.ggml.tokens</c> + <c>scores</c> + <c>token_type</c> + bos/eos ids) — the counterpart
/// to <see cref="GgufTokenizer"/> for models whose <c>tokenizer.ggml.model</c> is <c>llama</c> (SPM) rather than
/// byte-level BPE. Covers Gemma 1/2/3 and Llama-1/2.
///
/// <para>Implements the SentencePiece segmentation llama.cpp uses: spaces are replaced by the ▁ meta-symbol
/// (U+2581), the text is split into single Unicode code points, then the adjacent pair whose merged string is a
/// vocab entry with the <i>highest</i> score is merged repeatedly until none remain (score-driven, unlike BPE's
/// rank-driven merges). Code points with no vocab token fall back to per-byte <c>&lt;0xNN&gt;</c> tokens.
/// Control / user-defined tokens (the chat template's <c>&lt;bos&gt;</c>, <c>&lt;start_of_turn&gt;</c>, …) are
/// split out before segmentation so their literals map to ids, and are skipped on decode.</para></summary>
public sealed class SpmGgufTokenizer : ILlmTokenizer
{
    private const int TypeUnknown = 2;
    private const int TypeControl = 3;
    private const int TypeUserDefined = 4;
    private const int TypeByte = 6;
    private const char SpaceMeta = '▁';   // ▁ — SentencePiece space marker

    private readonly string[] _tokens;
    private readonly float[] _scores;
    private readonly Dictionary<string, int> _tokenToId;
    private readonly Dictionary<byte, int> _byteToId;          // byte value → <0xNN> token id (byte fallback)
    private readonly Dictionary<string, int> _specialByLiteral;
    private readonly string[] _specialLiterals;               // longest-first for greedy literal matching
    private readonly HashSet<int> _specialIds;
    private readonly int _unkId;
    private readonly bool _addSpacePrefix;

    public int? BosId { get; }
    public int? EosId { get; }
    public string? BosToken { get; }
    public string? EosToken { get; }
    public IReadOnlyList<int> StopIds { get; }

    /// <summary>Builds from GGUF metadata arrays. <paramref name="scores"/> is the per-token SentencePiece score
    /// (parallel to <paramref name="tokens"/>); <paramref name="tokenType"/> marks control/user-defined/byte
    /// tokens. <paramref name="addSpacePrefix"/> mirrors <c>tokenizer.ggml.add_space_prefix</c> (Llama-2 true,
    /// Gemma-3 false) — when true a leading ▁ is added so the first word is treated like any space-prefixed word.</summary>
    public SpmGgufTokenizer(string[] tokens, float[] scores, int[]? tokenType,
        int? bosId, int? eosId, IReadOnlyList<int>? extraStopIds = null, bool addSpacePrefix = false)
    {
        if (tokens is null || tokens.Length == 0) throw new ArgumentException("GGUF tokens array is empty.", nameof(tokens));
        if (scores is null || scores.Length != tokens.Length)
            throw new ArgumentException("SPM tokenizer needs a tokenizer.ggml.scores array parallel to tokens.", nameof(scores));

        _tokens = tokens;
        _scores = scores;
        _addSpacePrefix = addSpacePrefix;
        _tokenToId = new Dictionary<string, int>(tokens.Length, StringComparer.Ordinal);
        for (int i = 0; i < tokens.Length; i++) _tokenToId.TryAdd(tokens[i], i);

        _byteToId = new Dictionary<byte, int>(256);
        _specialByLiteral = new Dictionary<string, int>(StringComparer.Ordinal);
        _specialIds = [];
        _unkId = -1;
        for (int i = 0; i < tokens.Length; i++)
        {
            int type = tokenType is not null && i < tokenType.Length ? tokenType[i] : 1;
            if (type == TypeByte)
            {
                if (TryParseByteToken(tokens[i], out byte b)) _byteToId[b] = i;
            }
            else if (type == TypeControl || type == TypeUserDefined)
            {
                if (tokens[i].Length > 0) { _specialByLiteral[tokens[i]] = i; _specialIds.Add(i); }
            }
            else if (type == TypeUnknown && _unkId < 0)
            {
                _unkId = i;
            }
        }
        _specialLiterals = _specialByLiteral.Keys.OrderByDescending(s => s.Length).ToArray();

        BosId = bosId;
        EosId = eosId;
        BosToken = bosId is int bi && bi >= 0 && bi < tokens.Length ? tokens[bi] : null;
        EosToken = eosId is int ei && ei >= 0 && ei < tokens.Length ? tokens[ei] : null;

        HashSet<int> stops = [];
        if (eosId is int eid) stops.Add(eid);
        if (extraStopIds is not null) foreach (int s in extraStopIds) stops.Add(s);
        foreach (string lit in new[] { "<end_of_turn>", "<eos>", "<|im_end|>", "<|end|>", "<|eot_id|>" })
            if (_specialByLiteral.TryGetValue(lit, out int id)) stops.Add(id);
        StopIds = stops.ToArray();
    }

    public int? SpecialId(string token) => _specialByLiteral.TryGetValue(token, out int id) ? id : null;

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

    public int[] EncodeOrdinary(string text)
    {
        if (text.Length == 0) return [];

        // SentencePiece normalization: spaces → ▁ (and an optional leading ▁ for add_space_prefix).
        string normalized = text.Replace(' ', SpaceMeta);
        if (_addSpacePrefix && (normalized.Length == 0 || normalized[0] != SpaceMeta))
            normalized = SpaceMeta + normalized;

        // Initial symbols = single Unicode code points.
        List<string> symbols = [];
        for (int i = 0; i < normalized.Length;)
        {
            int len = char.IsHighSurrogate(normalized[i]) && i + 1 < normalized.Length ? 2 : 1;
            symbols.Add(normalized.Substring(i, len));
            i += len;
        }

        // Repeatedly merge the adjacent pair whose concatenation is the highest-scoring vocab token.
        while (symbols.Count > 1)
        {
            float bestScore = float.NegativeInfinity;
            int bestIdx = -1;
            for (int i = 0; i < symbols.Count - 1; i++)
            {
                if (_tokenToId.TryGetValue(symbols[i] + symbols[i + 1], out int id) && _scores[id] > bestScore)
                {
                    bestScore = _scores[id];
                    bestIdx = i;
                }
            }
            if (bestIdx < 0) break;
            symbols[bestIdx] += symbols[bestIdx + 1];
            symbols.RemoveAt(bestIdx + 1);
        }

        List<int> ids = new(symbols.Count);
        foreach (string sym in symbols)
        {
            if (_tokenToId.TryGetValue(sym, out int id)) { ids.Add(id); continue; }
            // Byte fallback: emit the symbol's UTF-8 bytes as <0xNN> tokens.
            foreach (byte b in Encoding.UTF8.GetBytes(sym))
            {
                if (_byteToId.TryGetValue(b, out int bid)) ids.Add(bid);
                else if (_unkId >= 0) ids.Add(_unkId);
            }
        }
        return [.. ids];
    }

    public string Decode(IReadOnlyList<int> ids)
    {
        List<byte> bytes = new(ids.Count * 2);
        foreach (int id in ids)
        {
            if (id < 0 || id >= _tokens.Length || _specialIds.Contains(id)) continue;
            if (_byteToId.Count > 0 && TryParseByteToken(_tokens[id], out byte raw)) { bytes.Add(raw); continue; }
            // Normal piece: ▁ → space, then its UTF-8 bytes (so multi-byte chars split across byte tokens rejoin).
            bytes.AddRange(Encoding.UTF8.GetBytes(_tokens[id].Replace(SpaceMeta, ' ')));
        }
        return Encoding.UTF8.GetString(bytes.ToArray());
    }

    /// <summary>Parses a SentencePiece byte token literal <c>&lt;0xNN&gt;</c> into its byte value.</summary>
    private static bool TryParseByteToken(string token, out byte value)
    {
        value = 0;
        if (token.Length != 6 || token[0] != '<' || token[1] != '0' || token[2] != 'x' || token[5] != '>')
            return false;
        return byte.TryParse(token.AsSpan(3, 2), System.Globalization.NumberStyles.HexNumber, null, out value);
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
}
