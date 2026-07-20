using System.Text;

namespace HartsyInference.Audio.Phonemizer.Espeak;

/// <summary>Renders a clause phoneme list to an IPA string, porting <c>GetTranslatedPhonemeString</c> +
/// <c>WritePhMnemonic</c> from dictionary.c. Each phoneme's IPA comes from its bytecode program's <c>i_IPA_NAME</c>
/// (via the interpreter), or, when none is defined, from converting its mnemonic through espeak's <c>ipa1</c> ascii
/// table. Stress marks (ˈ/ˌ) are emitted from each syllable's stress level.</summary>
internal sealed class EspeakPhonemeRenderer
{
    // ipa1[96]: ascii 0x20..0x7f -> unicode codepoint (dictionary.c).
    private static readonly ushort[] Ipa1 =
    [
        0x20, 0x21, 0x22, 0x2b0, 0x24, 0x25, 0x0e6, 0x2c8, 0x28, 0x29, 0x27e, 0x2b, 0x2cc, 0x2d, 0x2e, 0x2f,
        0x252, 0x31, 0x32, 0x25c, 0x34, 0x35, 0x36, 0x37, 0x275, 0x39, 0x2d0, 0x2b2, 0x3c, 0x3d, 0x3e, 0x294,
        0x259, 0x251, 0x3b2, 0xe7, 0xf0, 0x25b, 0x46, 0x262, 0x127, 0x26a, 0x25f, 0x4b, 0x26b, 0x271, 0x14b, 0x254,
        0x3a6, 0x263, 0x280, 0x283, 0x3b8, 0x28a, 0x28c, 0x153, 0x3c7, 0xf8, 0x292, 0x32a, 0x5c, 0x5d, 0x5e, 0x5f,
        0x60, 0x61, 0x62, 0x63, 0x64, 0x65, 0x66, 0x261, 0x68, 0x69, 0x6a, 0x6b, 0x6c, 0x6d, 0x6e, 0x6f,
        0x70, 0x71, 0x72, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x7b, 0x7c, 0x7d, 0x303, 0x7f
    ];

    private readonly EspeakPhonemeInterpreter _interp;
    private readonly EspeakPhonemeData _phdata = new();

    public EspeakPhonemeRenderer(EspeakPhonemeInterpreter interpreter) => _interp = interpreter;

    /// <summary>Renders the list (built by <see cref="EspeakPhonemeList"/>, with two guard entries at each end) to IPA,
    /// separating words with a space.</summary>
    public string Render(IReadOnlyList<EspeakPhonemeListEntry> list)
    {
        StringBuilder sb = new();
        bool firstWord = true;
        for (int ix = 2; ix < list.Count - 2; ix++)
        {
            EspeakPhonemeListEntry e = list[ix];
            if (e.Type == EspeakPhoneme.TypePause) continue;

            if (e.SourceIx != 0)
            {
                if (!firstWord) sb.Append(' ');
                firstWord = false;
            }

            if ((e.SynthFlags & EspeakProgram.SflagSyllable) != 0 && e.StressLevel > 1)
                sb.Append(e.StressLevel > EspeakProgram.StressSecondary ? 'ˈ' : 'ˌ');

            sb.Append(Mnemonic(list, ix));
        }
        return sb.ToString();
    }

    // WritePhMnemonic (IPA mode): program ipa_string, else mnemonic via ipa1.
    private string Mnemonic(IReadOnlyList<EspeakPhonemeListEntry> list, int ix)
    {
        EspeakPhonemeListEntry e = list[ix];
        _interp.Interpret(list, ix, 0, tr: false, _phdata);
        string ipa = _phdata.IpaString;
        if (ipa.Length > 0)
        {
            if (ipa[0] == ' ') return string.Empty;     // explicit "no name"
            if (ipa[0] < 0x20) ipa = ipa[1..];           // leading flags byte
            return ipa;
        }

        // Fallback: convert the mnemonic characters.
        EspeakPhoneme ph = e.Ph;
        StringBuilder sb = new();
        bool first = true;
        uint mnem = ph.Mnemonic;
        for (; (mnem & 0xff) != 0; mnem >>= 8)
        {
            int c = (int)(mnem & 0xff);
            if (c == '/') break;                         // variant indicator
            if (first && c == '_') break;                // pause phoneme: no output
            if (c == '#' && ph.Type == EspeakPhoneme.TypeVowel) break; // subscript-h is consonant-only
            if (!first && c is >= '0' and <= '9') { first = false; continue; } // drop non-initial digits
            if (c is >= 0x20 and < 128) c = Ipa1[c - 0x20];
            sb.Append(char.ConvertFromUtf32(c));
            first = false;
        }
        return sb.ToString();
    }
}
