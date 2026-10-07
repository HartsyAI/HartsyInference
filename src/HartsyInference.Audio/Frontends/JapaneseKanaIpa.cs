using System.Text;

namespace HartsyInference.Audio.Frontends;

/// <summary>misaki's adaptation of cutlet's Hepburn table (cutlet MIT, misaki Apache-2.0): hiragana, digraphs and
/// Japanese punctuation to the IPA-like symbols Kokoro was trained on, plus cutlet's context rules for sokuon,
/// moraic ん, small kana, long-vowel mark and iteration marks.</summary>
internal static class JapaneseKanaIpa
{
    private const string Sutegana = "ゃゅょぁぃぅぇぉ";
    private const string Odori = "〃々ゝゞヽ";

    /// <summary>cutlet's <c>HEPBURN</c> table as misaki modifies it.</summary>
    public static readonly IReadOnlyDictionary<string, string> Table = BuildTable();

    /// <summary>cutlet <c>_get_single_mapping</c>: the symbols for kana <paramref name="kk"/> between its neighbours
    /// <paramref name="pk"/> and <paramref name="nk"/> (null at the word edges).</summary>
    public static string Map(string? pk, string kk, string? nk)
    {
        if (Odori.Contains(kk, StringComparison.Ordinal))
        {
            if ("ゝヽ".Contains(kk, StringComparison.Ordinal)) return pk ?? "";
            if ("ゞヾ".Contains(kk, StringComparison.Ordinal))
            {
                if (pk is null) return "";
                string? voiced = AddDakuten(pk);
                return voiced is null ? "" : Table[voiced];
            }
            return "";
        }
        if (pk is not null && Table.TryGetValue(pk + kk, out string? digraph)) return digraph;
        if (nk is not null && Table.ContainsKey(kk + nk)) return "";
        if (nk is not null && Sutegana.Contains(nk, StringComparison.Ordinal))
        {
            if (kk == "っ") return "";
            // cutlet raises KeyError for a kana outside the table here; dropping its symbols is the graceful reading.
            string self = Table.TryGetValue(kk, out string? s) ? s : "";
            return (self.Length == 0 ? "" : self[..^1]) + Table[nk];
        }
        if (Sutegana.Contains(kk, StringComparison.Ordinal)) return "";
        if (kk == "ー") return "ː";
        if (kk == "っ") return "ʔ";
        if (kk == "ん")
        {
            if (nk is not null && Table.TryGetValue(nk, out string? next) && next.Length > 0)
            {
                char c = next[0];
                if (c is 'm' or 'p' or 'b') return "m";
                if (c is 'k' or 'ɡ') return "ŋ";
                if (next.StartsWith('ɲ') || next.StartsWith('ʨ') || next.StartsWith('ʥ')) return "ɲ";
                if (c is 'n' or 't' or 'd' or 'ɾ' or 'z') return "n";
            }
            return "ɴ";
        }
        return Table.TryGetValue(kk, out string? mapped) ? mapped : "";
    }

    /// <summary>The symbols for a whole hiragana reading, kana by kana with its in-word neighbours.</summary>
    public static string MapReading(string hira)
    {
        List<string> chars = [];
        foreach (Rune r in hira.EnumerateRunes()) chars.Add(r.ToString());
        StringBuilder sb = new();
        for (int i = 0; i < chars.Count; i++)
            sb.Append(Map(i > 0 ? chars[i - 1] : null, chars[i], i < chars.Count - 1 ? chars[i + 1] : null));
        return sb.ToString();
    }

    private static string? AddDakuten(string kana)
    {
        const string Plain = "かきくけこさしすせそたちつてとはひふへほ";
        const string Voiced = "がぎぐげござじずぜぞだぢづでどばびぶべぼ";
        int i = kana.Length == 1 ? Plain.IndexOf(kana[0]) : -1;
        return i < 0 ? null : Voiced[i].ToString();
    }

    private static Dictionary<string, string> BuildTable()
    {
        Dictionary<string, string> t = new(StringComparer.Ordinal);
        // ぁ (U+3041) .. ゖ (U+3096), っ and ん left to the context rules.
        string[] kana =
        [
            "a", "a", "i", "i", "ɯ", "ɯ", "e", "e", "o", "o", "ka", "ɡa", "kʲi", "ɡʲi", "kɯ", "ɡɯ", "ke", "ɡe", "ko", "ɡo",
            "sa", "ʣa", "ɕi", "ʥi", "sɨ", "zɨ", "se", "ʣe", "so", "ʣo", "ta", "da", "ʨi", "ʥi", "", "ʦɨ", "zɨ", "te", "de",
            "to", "do", "na", "ɲi", "nɯ", "ne", "no", "ha", "ba", "pa", "çi", "bʲi", "pʲi", "ɸɯ", "bɯ", "pɯ", "he", "be",
            "pe", "ho", "bo", "po", "ma", "mʲi", "mɯ", "me", "mo", "ja", "ja", "jɯ", "jɯ", "jo", "jo", "ɾa", "ɾʲi", "ɾɯ",
            "ɾe", "ɾo", "βa", "βa", "i", "e", "o", "", "vɯ", "ka", "ke",
        ];
        for (int i = 0; i < kana.Length; i++)
        {
            int cp = 0x3041 + i;
            if (cp is 0x3063 or 0x3093) continue;
            t[((char)cp).ToString()] = kana[i];
        }
        t["ヷ"] = "va";
        t["ヸ"] = "vʲi";
        t["ヹ"] = "ve";
        t["ヺ"] = "vo";
        string[] digraphs =
        [
            "いぇ", "je", "うぃ", "βi", "うぇ", "βe", "うぉ", "βo", "きぇ", "kʲe", "きゃ", "kʲa", "きゅ", "kʲɨ", "きょ", "kʲo",
            "ぎゃ", "ɡʲa", "ぎゅ", "ɡʲɨ", "ぎょ", "ɡʲo", "くぁ", "kᵝa", "くぃ", "kᵝi", "くぇ", "kᵝe", "くぉ", "kᵝo",
            "ぐぁ", "ɡᵝa", "ぐぃ", "ɡᵝi", "ぐぇ", "ɡᵝe", "ぐぉ", "ɡᵝo", "しぇ", "ɕe", "しゃ", "ɕa", "しゅ", "ɕɨ", "しょ", "ɕo",
            "じぇ", "ʥe", "じゃ", "ʥa", "じゅ", "ʥɨ", "じょ", "ʥo", "ちぇ", "ʨe", "ちゃ", "ʨa", "ちゅ", "ʨɨ", "ちょ", "ʨo",
            "ぢゃ", "ʥa", "ぢゅ", "ʥɨ", "ぢょ", "ʥo", "つぁ", "ʦa", "つぃ", "ʦʲi", "つぇ", "ʦe", "つぉ", "ʦo", "てぃ", "tʲi",
            "てゅ", "tʲɨ", "でぃ", "dʲi", "でゅ", "dʲɨ", "とぅ", "tɯ", "どぅ", "dɯ", "にぇ", "ɲe", "にゃ", "ɲa", "にゅ", "ɲɨ",
            "にょ", "ɲo", "ひぇ", "çe", "ひゃ", "ça", "ひゅ", "çɨ", "ひょ", "ço", "びゃ", "bʲa", "びゅ", "bʲɨ", "びょ", "bʲo",
            "ぴゃ", "pʲa", "ぴゅ", "pʲɨ", "ぴょ", "pʲo", "ふぁ", "ɸa", "ふぃ", "ɸʲi", "ふぇ", "ɸe", "ふぉ", "ɸo", "ふゅ", "ɸʲɨ",
            "ふょ", "ɸʲo", "みゃ", "mʲa", "みゅ", "mʲɨ", "みょ", "mʲo", "りゃ", "ɾʲa", "りゅ", "ɾʲɨ", "りょ", "ɾʲo", "ゔぁ", "va",
            "ゔぃ", "vʲi", "ゔぇ", "ve", "ゔぉ", "vo", "ゔゅ", "bʲɨ", "ゔょ", "bʲo",
            "。", ".", "、", ",", "？", "?", "！", "!", "「", "“", "」", "”", "『", "“", "』", "”", "：", ":", "；", ";",
            "（", "(", "）", ")", "《", "(", "》", ")", "【", "[", "】", "]", "・", " ", "，", ",", "～", "—", "〜", "—",
            "—", "—", "«", "“", "»", "”", "゚", "", "゙", "",
        ];
        for (int i = 0; i < digraphs.Length; i += 2) t[digraphs[i]] = digraphs[i + 1];
        return t;
    }
}
