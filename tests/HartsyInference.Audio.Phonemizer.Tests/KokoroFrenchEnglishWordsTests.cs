using System.Text.Json;
using HartsyInference.Audio.Frontends;
using HartsyInference.Audio.Phonemizer.Espeak;
using Xunit;

namespace HartsyInference.Audio.Phonemizer.Tests;

/// <summary>French espeak-ng switches some English words to English: the word's dictionary entry, or its rules, emit
/// the language-switch phoneme (<c>_^_</c>), and espeak then reads the word with its English voice. Kokoro's French
/// voice (<c>ff_siwis</c>) needs the same, or those words reach the model as a blank. The expected strings are
/// espeak-ng 1.52's own output for these words (libespeak-ng 1.52.0.1, voice <c>fr</c>) with the switch spans
/// removed, then misaki's single-symbol spellings. Gated on <c>ESPEAK_DATA_DIR</c>.</summary>
public sealed class KokoroFrenchEnglishWordsTests
{
    private const string Speech = "The quick brown fox jumps over the lazy dog, then naps in the warm afternoon sun.";

    // espeak-ng 1.52 (voice fr) switches these to English, one word at a time.
    private static readonly string[] SwitchedWords =
        ["The", "quick", "brown", "over", "the", "lazy", "then", "in", "warm", "afternoon", "sun"];

    // espeak-ng 1.52 keeps these French.
    private static readonly string[] FrenchWords = ["fox", "jumps", "dog", "naps"];

    private static string? DataDir()
    {
        string? dir = Environment.GetEnvironmentVariable("ESPEAK_DATA_DIR");
        return string.IsNullOrEmpty(dir) || !File.Exists(Path.Combine(dir, "phontab")) ? null : dir;
    }

    [Fact]
    public void Sentence_SwitchesTheWordsEspeakSwitches()
    {
        string? dir = DataDir();
        if (dir is null) return; // gated
        EspeakPhonemizer fr = EspeakPhonemizer.FromDataDirectory(dir, "fr");
        KokoroEspeakG2P g2p = new(fr, EspeakPhonemizer.FromDataDirectory(dir, "en"), fr.SwitchesToEnglish);

        Assert.Equal("ðə kwˈɪk bɹˈWn fˈɔks ʒˈœ̃ ˌQvə ðə lˈAzi dˈɔɡ, ðˈɛn nˈap ɪnðə wˈɔːm ˌaftənˈuːn sˈʌn.", g2p.ToIpa(Speech));
    }

    [Fact]
    public void Words_PartitionAsEspeakDoes()
    {
        string? dir = DataDir();
        if (dir is null) return; // gated
        EspeakPhonemizer fr = EspeakPhonemizer.FromDataDirectory(dir, "fr");

        Assert.All(SwitchedWords, w => Assert.True(fr.SwitchesToEnglish(w), $"'{w}' should switch to English"));
        Assert.All(FrenchWords, w => Assert.False(fr.SwitchesToEnglish(w), $"'{w}' should stay French"));
    }

    [Fact]
    public void NativeFrench_IsUnchanged()
    {
        // Native French has no switching word, so the new predicate must leave its output byte for byte as it was.
        string? dir = DataDir();
        if (dir is null) return; // gated
        EspeakPhonemizer fr = EspeakPhonemizer.FromDataDirectory(dir, "fr");
        KokoroEspeakG2P before = new(fr);
        KokoroEspeakG2P after = new(fr, EspeakPhonemizer.FromDataDirectory(dir, "en"), fr.SwitchesToEnglish);

        using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(
            Path.Combine(AppContext.BaseDirectory, "Fixtures", "kokoro_espeak_parity.json")));
        string[] corpus = doc.RootElement.GetProperty("corpus").GetProperty("fr-fr").EnumerateArray()
            .Select(e => e.GetString()!).ToArray();
        foreach (string sentence in corpus)
        {
            Assert.Equal(before.ToIpa(sentence), after.ToIpa(sentence));
        }
    }
}
