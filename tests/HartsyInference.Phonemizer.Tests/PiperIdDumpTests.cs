using HartsyInference.Phonemizer;
using HartsyInference.Phonemizer.Espeak;
using Xunit;

namespace HartsyInference.Phonemizer.Tests;

/// <summary>Dumps the C# espeak→Piper token ids + IPA for a fixed sentence (for offline comparison to piper). Gated on
/// <c>ESPEAK_DATA_DIR</c> and <c>PIPER_CFG</c> (a Piper <c>.onnx.json</c>).</summary>
public sealed class PiperIdDumpTests
{
    [Fact]
    public void DumpIds()
    {
        string? dir = Environment.GetEnvironmentVariable("ESPEAK_DATA_DIR");
        string? cfg = Environment.GetEnvironmentVariable("PIPER_CFG");
        if (string.IsNullOrEmpty(dir) || string.IsNullOrEmpty(cfg) || !File.Exists(cfg)) return;

        EspeakPhonemizer phon = EspeakPhonemizer.FromDataDirectory(dir, "en");
        using FileStream fs = File.OpenRead(cfg);
        PhonemeIdMap idMap = PhonemeIdMap.FromPiperConfig(fs);

        const string text = "Hello world. This is a test of the speech synthesizer.";
        string ipa = phon.PhonemizeToIpa(text, "en-us");
        int[] ids = phon.PhonemizeToIds(text, "en-us", idMap);

        File.WriteAllText("/tmp/cs_piper_ids.txt", ipa + "\n" + string.Join(",", ids));
    }
}
