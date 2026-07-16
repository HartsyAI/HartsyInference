using System.Collections.Generic;
using System.Globalization;

namespace HartsyInference.Audio.Models.StyleTts2;

/// <summary>The StyleTTS2 (LibriTTS) phoneme symbol set — <c>[pad] + punctuation + letters + IPA letters</c> from
/// the upstream <c>text_utils.py</c>, in order. The token id of each symbol is its index (pad = 0), so this is the
/// authoritative <c>{symbol: id}</c> map the LibriTTS text-encoder/PLBERT embeddings are indexed by (178 symbols).
/// Embedded here (byte-exact, including the combining mark and curly quotes) so the pipeline is self-contained.</summary>
public static class StyleTts2Symbols
{
    /// <summary>All 178 symbols concatenated in id order (each Unicode code point is one token).</summary>
    public const string Symbols = "$;:,.!?¡¿—…\"«»“” ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ";

    /// <summary>Builds the <c>{symbol: id}</c> vocab (id = code-point index; later duplicates win, matching upstream).</summary>
    public static Dictionary<string, int> BuildVocab()
    {
        Dictionary<string, int> vocab = new(StringComparer.Ordinal);
        int id = 0;
        for (int i = 0; i < Symbols.Length;)
        {
            int cp = char.ConvertToUtf32(Symbols, i);
            vocab[char.ConvertFromUtf32(cp)] = id;
            i += char.IsSurrogatePair(Symbols, i) ? 2 : 1;
            id++;
        }
        return vocab;
    }
}
