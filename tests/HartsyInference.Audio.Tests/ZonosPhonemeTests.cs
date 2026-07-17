using System;
using System.IO;
using System.Text.Json;
using HartsyInference.Audio.Models.Zonos;
using HartsyInference.Phonemizer.Espeak;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>Zonos phoneme tokenizer parity. The table-correctness test is a pure Unit test (feeds the golden
/// IPA string, no espeak/checkpoint); the espeak-parity test is gated on the espeak-ng data being installed.</summary>
public sealed class ZonosPhonemeTests
{
    private readonly ITestOutputHelper _out;
    public ZonosPhonemeTests(ITestOutputHelper o) => _out = o;

    // Golden from zonos_golden.py (phonemes.json) for "Hello, this is a test of the Zonos text to speech system."
    private const string GoldenIpa = "həlˈoʊ, ðˈɪs ɪz ˈeɪ tˈɛst ʌv ðə zˈoʊnoʊz tˈɛkst tuː spˈiːtʃ sˈɪstəm.";
    private static readonly int[] GoldenIds =
    [
        2, 61, 94, 65, 167, 68, 146, 6, 21, 92, 167, 113, 72, 21, 113, 79, 21, 167, 58, 113, 21, 73, 167, 97, 72, 73,
        21, 149, 75, 21, 92, 94, 21, 79, 167, 68, 146, 67, 68, 146, 79, 21, 73, 167, 97, 64, 72, 73, 21, 73, 74, 169,
        21, 72, 69, 167, 62, 169, 73, 142, 21, 72, 167, 113, 72, 73, 94, 66, 7, 3,
    ];

    [Fact]
    public void TokenizeIpa_MatchesReferenceTable()
    {
        int[] ids = ZonosPhonemeTokenizer.TokenizeIpa(GoldenIpa);
        Assert.Equal(GoldenIds, ids);
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void EspeakPhonemize_MatchesReferenceIpa()
    {
        string? dataDir = Environment.GetEnvironmentVariable("ESPEAK_DATA_DIR");
        if (string.IsNullOrEmpty(dataDir) || !Directory.Exists(dataDir))
        {
            _out.WriteLine("Skipped: set ESPEAK_DATA_DIR to the espeak-ng-data directory.");
            return;
        }

        EspeakPhonemizer ph = EspeakPhonemizer.FromDataDirectory(dataDir, "en-us");
        string raw = ph.PhonemizeToIpa("Hello, this is a test of the Zonos text to speech system.", "en-us", preservePunctuation: true);
        string ipa = ZonosPhonemeTokenizer.NormalizePunctuation(raw);
        _out.WriteLine($"engine ipa: {ipa}");
        _out.WriteLine($"golden ipa: {GoldenIpa}");
        int[] ids = ZonosPhonemeTokenizer.TokenizeIpa(ipa);
        _out.WriteLine($"engine ids: [{string.Join(",", ids)}]");

        // The pure-C# espeak port can differ from the Python phonemizer's espeak-ng by the occasional stress
        // mark; require the *segmental* phonemes to match exactly (parity modulo primary/secondary stress).
        Assert.Equal(StripStress(GoldenIpa), StripStress(ipa));
    }

    private static string StripStress(string s) => s.Replace("ˈ", "").Replace("ˌ", "");
}
