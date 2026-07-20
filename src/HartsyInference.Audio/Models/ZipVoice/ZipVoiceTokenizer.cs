using System.Globalization;
using HartsyInference.Phonemizer;
using HartsyInference.Phonemizer.Espeak;

namespace HartsyInference.Audio.Models.ZipVoice;

/// <summary>Ports ZipVoice's <c>EmiliaTokenizer.tokenize_EN</c> (English-only — this port does not implement
/// the Chinese/pinyin branch, which needs jieba + pypinyin word segmentation with no equivalent in this
/// engine). Upstream calls <c>piper_phonemize.phonemize_espeak(text, "en-us")</c> then flattens per-word
/// phoneme-symbol lists into one token stream; the real checkpoint's <c>tokens.txt</c> vocab is single
/// Unicode-codepoint IPA symbols plus punctuation/space/BOS/EOS/pad markers. This port uses the engine's own
/// pure-C# espeak-ng phonemizer (<see cref="EspeakPhonemizer.PhonemizeToIpa(string,string,bool)"/>, same
/// underlying espeak engine piper_phonemize wraps) with prosody punctuation preserved, then splits the
/// resulting IPA string into individual Unicode text elements and maps each through <c>tokens.txt</c>,
/// silently skipping any out-of-vocabulary symbol (matches upstream's <c>tokens_to_token_ids</c>).</summary>
public sealed class ZipVoiceTokenizer
{
    private readonly Dictionary<string, int> _token2id;

    public int PadId { get; }
    public int VocabSize => _token2id.Count;

    private ZipVoiceTokenizer(Dictionary<string, int> token2id)
    {
        _token2id = token2id;
        PadId = token2id["_"];
    }

    /// <summary>Loads <c>tokens.txt</c> (<c>{token}\t{id}</c> per line, as shipped with the real
    /// <c>k2-fsa/ZipVoice</c> checkpoint).</summary>
    public static ZipVoiceTokenizer FromFile(string tokensPath)
    {
        Dictionary<string, int> map = new();
        foreach (string line in File.ReadLines(tokensPath))
        {
            if (line.Length == 0) continue;
            int tab = line.LastIndexOf('\t');
            if (tab < 0) continue;
            string token = line[..tab];
            if (!int.TryParse(line[(tab + 1)..], out int id)) continue;
            map[token] = id;
        }
        return new ZipVoiceTokenizer(map);
    }

    /// <summary>Basic ASCII/CJK-punctuation normalization mirroring upstream <c>map_punctuations</c> (fullwidth
    /// punctuation → ASCII, ellipsis variants → <c>…</c>). English-only, so the CJK punctuation cases are
    /// mostly inert but cheap to keep for mixed-input robustness.</summary>
    public static string NormalizePunctuation(string text) => text
        .Replace('，', ',').Replace('。', '.').Replace('！', '!').Replace('？', '?')
        .Replace('；', ';').Replace('：', ':').Replace('、', ',')
        .Replace('‘', '\'').Replace('’', '\'').Replace('“', '"').Replace('”', '"')
        .Replace("⋯", "…").Replace("···", "…").Replace("・・・", "…").Replace("...", "…");

    /// <summary>Text → token ids, via espeak IPA phonemization + per-codepoint vocab lookup.
    /// <paramref name="phonemizer"/> is typically <see cref="EspeakPhonemizer.FromCache"/>.</summary>
    public int[] Encode(string text, EspeakPhonemizer phonemizer, string language = "en-us")
    {
        string ipa = phonemizer.PhonemizeToIpa(NormalizePunctuation(text), language, preservePunctuation: true);
        List<int> ids = new();
        TextElementEnumerator e = StringInfo.GetTextElementEnumerator(ipa);
        while (e.MoveNext())
        {
            string sym = (string)e.Current;
            if (_token2id.TryGetValue(sym, out int id)) ids.Add(id);
        }
        return [.. ids];
    }
}
