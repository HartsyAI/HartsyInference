using System.Text;
using System.Text.Json;

namespace HartsyInference.ModelAssets.Tokenizers;

/// <summary>A Gemma-family BPE tokenizer read from a HuggingFace <c>tokenizer.json</c>: space → <c>▁</c> normalization,
/// character-level BPE with merge ranks, UTF-8 byte fallback (<c>&lt;0xNN&gt;</c>) for characters outside the vocabulary,
/// and added tokens (<c>&lt;bos&gt;</c>, <c>&lt;|AUDIO|&gt;</c>, <c>[S0]</c>, …) matched literally anywhere in the text,
/// longest first. Unlike <see cref="HfTokenizerJson"/>'s byte-level BPE this is the SentencePiece-style layout Gemma,
/// T5Gemma and PaliGemma ship.</summary>
public sealed class SentencePieceBpeJson
{
    private const char Space = '▁';
    private readonly Dictionary<string, int> _vocab = new(StringComparer.Ordinal);
    private readonly Dictionary<(string, string), int> _ranks = [];
    private readonly Dictionary<string, int> _added = new(StringComparer.Ordinal);
    private readonly string[] _addedByLength;
    private readonly int _unk;
    private readonly int[] _byteIds = new int[256];

    /// <summary>Token id of <c>&lt;bos&gt;</c> (2 in Gemma), or -1.</summary>
    public int BosId { get; }

    public SentencePieceBpeJson(Stream json)
    {
        using JsonDocument doc = JsonDocument.Parse(json);
        JsonElement root = doc.RootElement, model = root.GetProperty("model");
        if (model.GetProperty("type").GetString() != "BPE")
            throw new NotSupportedException("Only BPE tokenizer.json files are supported.");
        foreach (JsonProperty kv in model.GetProperty("vocab").EnumerateObject()) _vocab[kv.Name] = kv.Value.GetInt32();
        int rank = 0;
        foreach (JsonElement m in model.GetProperty("merges").EnumerateArray())
        {
            string a, b;
            if (m.ValueKind == JsonValueKind.Array) { a = m[0].GetString()!; b = m[1].GetString()!; }
            else { string[] p = m.GetString()!.Split(' ', 2); a = p[0]; b = p[1]; }
            _ranks.TryAdd((a, b), rank++);
        }
        _unk = model.TryGetProperty("unk_token", out JsonElement u) && u.ValueKind == JsonValueKind.String
            && _vocab.TryGetValue(u.GetString()!, out int uid) ? uid : -1;
        for (int i = 0; i < 256; i++) _byteIds[i] = _vocab.TryGetValue($"<0x{i:X2}>", out int id) ? id : _unk;
        if (root.TryGetProperty("added_tokens", out JsonElement added))
            foreach (JsonElement t in added.EnumerateArray())
            {
                string content = t.GetProperty("content").GetString()!;
                if (content.Length > 0) _added[content] = t.GetProperty("id").GetInt32();
                _vocab[content] = t.GetProperty("id").GetInt32();
            }
        _addedByLength = [.. _added.Keys.OrderByDescending(k => k.Length)];
        BosId = _added.TryGetValue("<bos>", out int bos) ? bos : -1;
    }

    /// <summary>Tokenizes <paramref name="text"/>, parsing added tokens inline. No BOS is added; callers put
    /// <c>&lt;bos&gt;</c> in the text when they want it (HF's template adds it only for <c>add_special_tokens=True</c>).</summary>
    public int[] Encode(string text)
    {
        List<int> ids = [];
        StringBuilder plain = new();
        void Flush() { if (plain.Length > 0) { EncodeOrdinary(plain.ToString(), ids); plain.Clear(); } }
        for (int i = 0; i < text.Length;)
        {
            string? hit = null;
            foreach (string lit in _addedByLength)
                if (string.CompareOrdinal(text, i, lit, 0, lit.Length) == 0) { hit = lit; break; }
            if (hit is not null) { Flush(); ids.Add(_added[hit]); i += hit.Length; }
            else { plain.Append(text[i]); i++; }
        }
        Flush();
        return [.. ids];
    }

    private void EncodeOrdinary(string text, List<int> ids)
    {
        string normalized = text.Replace(' ', Space);
        List<string> symbols = [];
        foreach (Rune r in normalized.EnumerateRunes())
        {
            string c = r.ToString();
            if (_vocab.ContainsKey(c)) symbols.Add(c);
            else foreach (byte b in Encoding.UTF8.GetBytes(c)) symbols.Add($"<0x{b:X2}>");
        }
        while (symbols.Count > 1)
        {
            int best = -1, bestRank = int.MaxValue;
            for (int i = 0; i + 1 < symbols.Count; i++)
                if (_ranks.TryGetValue((symbols[i], symbols[i + 1]), out int r) && r < bestRank) { bestRank = r; best = i; }
            if (best < 0) break;
            symbols[best] = symbols[best] + symbols[best + 1];
            symbols.RemoveAt(best + 1);
        }
        foreach (string s in symbols) ids.Add(_vocab.TryGetValue(s, out int id) ? id : _unk);
    }
}
