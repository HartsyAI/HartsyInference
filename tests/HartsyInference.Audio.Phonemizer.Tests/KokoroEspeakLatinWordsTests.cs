using HartsyInference.Audio.Frontends;
using HartsyInference.Audio.Phonemizer.Espeak;
using Xunit;

namespace HartsyInference.Audio.Phonemizer.Tests;

/// <summary>Hindi's script has no Latin letter rules, so espeak-ng reads Latin words with its English dictionary (its
/// <c>(en)</c> language-switch spans, which misaki's <c>remove-flags</c> drops). Kokoro's Hindi voices (<c>hf_*</c>,
/// <c>hm_*</c>) need the same, or an English sentence reaches the model as near-silence. Expected strings are espeak-ng
/// 1.52's own output for these words with the switch flags removed, then misaki's single-symbol spellings. Gated on
/// <c>ESPEAK_DATA_DIR</c>.</summary>
public sealed class KokoroEspeakLatinWordsTests
{
    private static string? DataDir()
    {
        string? dir = Environment.GetEnvironmentVariable("ESPEAK_DATA_DIR");
        return string.IsNullOrEmpty(dir) || !File.Exists(Path.Combine(dir, "phontab")) ? null : dir;
    }

    [Fact]
    public void LatinWords_InHindi_ReadWithTheEnglishDictionary()
    {
        string? dir = DataDir();
        if (dir is null) return; // gated
        KokoroEspeakG2P g2p = new(EspeakPhonemizer.FromDataDirectory(dir, "hi"), EspeakPhonemizer.FromDataDirectory(dir, "en"));

        Assert.Equal("ðə kwˈɪk bɹˈWn fˈɒks ʤˈʌmps ˌQvə ðə lˈAzi dˈɒɡ ðˈɛn nˈaps ɪnðə wˈɔːm ˌaftənˈuːn sˈʌn",
            g2p.ToIpa("The quick brown fox jumps over the lazy dog then naps in the warm afternoon sun"));
    }

    [Fact]
    public void MixedScript_SwitchesPerRun()
    {
        string? dir = DataDir();
        if (dir is null) return; // gated
        KokoroEspeakG2P g2p = new(EspeakPhonemizer.FromDataDirectory(dir, "hi"), EspeakPhonemizer.FromDataDirectory(dir, "en"));

        Assert.Equal("mɛ̃ ðə kwˈɪk ˈaːɟ", g2p.ToIpa("मैं The quick आज"));
    }
}
