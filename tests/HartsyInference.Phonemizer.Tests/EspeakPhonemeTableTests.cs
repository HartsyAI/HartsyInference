using HartsyInference.Phonemizer.Espeak;
using Xunit;

namespace HartsyInference.Phonemizer.Tests;

/// <summary>Validates the compiled <c>phontab</c> reader and the <c>includes</c>-chain assembly of the active English
/// phoneme table against the genuine v1.50 data file. Gated on <c>ESPEAK_DATA_DIR</c> pointing at an
/// <c>espeak-ng-data</c> directory.</summary>
public sealed class EspeakPhonemeTableTests
{
    private static string? PhontabPath()
    {
        string? dir = Environment.GetEnvironmentVariable("ESPEAK_DATA_DIR");
        if (string.IsNullOrEmpty(dir)) return null;
        string path = Path.Combine(dir, "phontab");
        return File.Exists(path) ? path : null;
    }

    [Fact]
    public void BuildsEnglishTableFromIncludesChain()
    {
        string? path = PhontabPath();
        if (path is null) return; // gated

        EspeakPhonemeTable table = EspeakPhonemeTable.Load(path, "en");

        // The base -> base1 -> en overlay yields 165 codes, highest code 164 (matches reference parse of v1.50).
        Assert.Equal(164, table.MaxCode);

        // Code 1 is the pause phoneme; its mnemonic begins with '_' (followed by espeak's 0x01 control byte).
        Assert.True(table.TryGet(1, out EspeakPhoneme pause));
        Assert.Equal('_', pause.MnemonicText[0]);
        Assert.Equal(EspeakPhoneme.TypePause, pause.Type);

        // The vowel "a" sits at code 35 and is typed as a vowel.
        Assert.True(table.TryGet(35, out EspeakPhoneme a));
        Assert.Equal("a", a.MnemonicText);
        Assert.Equal(EspeakPhoneme.TypeVowel, a.Type);

        // Reverse lookup round-trips.
        Assert.Equal(35, table.CodeForMnemonic(a.Mnemonic));
    }
}
