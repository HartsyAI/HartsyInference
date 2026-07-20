using HartsyInference.Audio.Phonemizer.Espeak;
using Xunit;

namespace HartsyInference.Audio.Phonemizer.Tests;

/// <summary>Validates the phoneme-program interpreter's IPA extraction (the <c>i_IPA_NAME</c> path) against known
/// values from espeak's real en-us <c>phonindex</c>. Gated on <c>ESPEAK_DATA_DIR</c>.</summary>
public sealed class EspeakInterpreterTests
{
    private static (EspeakPhonemeTable, EspeakPhonemeInterpreter)? Build()
    {
        string? dir = Environment.GetEnvironmentVariable("ESPEAK_DATA_DIR");
        if (string.IsNullOrEmpty(dir)) return null;
        string phontab = Path.Combine(dir, "phontab");
        string phonindex = Path.Combine(dir, "phonindex");
        if (!File.Exists(phontab) || !File.Exists(phonindex)) return null;
        EspeakPhonemeTable table = EspeakPhonemeTable.Load(phontab, "en-us");
        EspeakPhonemeInterpreter interp = new(table, EspeakPhonemeIndex.Load(phonindex));
        return (table, interp);
    }

    private static string IpaFor(EspeakPhonemeTable table, EspeakPhonemeInterpreter interp, int code)
    {
        table.TryGet(code, out EspeakPhoneme ph);
        EspeakPhonemeListEntry pause = new(1, table.TryGet(1, out EspeakPhoneme p) ? p : default);
        List<EspeakPhonemeListEntry> list = [pause, pause, new EspeakPhonemeListEntry(code, ph), pause, pause];
        EspeakPhonemeData data = new();
        interp.Interpret(list, 2, 0, tr: false, data);
        return data.IpaString;
    }

    [Fact]
    public void ExtractsIpaNamesFromPrograms()
    {
        (EspeakPhonemeTable table, EspeakPhonemeInterpreter interp)? built = Build();
        if (built is null) return; // gated
        (EspeakPhonemeTable table, EspeakPhonemeInterpreter interp) = built.Value;

        // Codes confirmed by directly parsing the en-us phonindex.
        Assert.Equal("ɾ", IpaFor(table, interp, 25));   // t# flap
        Assert.Equal("æ", IpaFor(table, interp, 35));   // a (American)
        Assert.Equal("i", IpaFor(table, interp, 37));   // i
        Assert.Equal("ɚ", IpaFor(table, interp, 111));  // 3 (rhotic schwa)

        // Phonemes with no i_IPA_NAME emit nothing (the renderer converts the mnemonic instead).
        Assert.Equal("", IpaFor(table, interp, 47));    // t
        Assert.Equal("", IpaFor(table, interp, 90));    // s
    }
}
