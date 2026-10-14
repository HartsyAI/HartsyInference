using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using HartsyInference.Audio.Frontends.WeText;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary><see cref="WeTextNormalizer"/> against wetext 0.0.4 (CosyVoice 2's pin) on every line of
/// <c>Fixtures/wetext_parity.json</c> (from tools/parity/wetext_parity_reference.py): tagger output and final text must
/// match exactly, and lines 0.0.4 raises on must fail here too. Needs the wetext FSTs: <c>WETEXT_FST_DIR</c>, else a
/// local wetext wheel's <c>wetext/fsts</c> or ModelScope snapshot; the files must hash to the fixture's SHA-256s.</summary>
public sealed class WeTextParityTests
{
    private static readonly string[] LocalFstDirs =
    [
        "/mnt/model-storage/scratch/tts-ref/fe-venv/lib/python3.12/site-packages/wetext/fsts",
        "/mnt/model-storage/scratch/tts-ref/cache/modelscope/models/pengzhendong--wetext/snapshots/master",
    ];

    private readonly ITestOutputHelper _out;

    public WeTextParityTests(ITestOutputHelper output) => _out = output;

    [Fact]
    [Trait("Category", "Integration")]
    public void EveryFixtureLine_MatchesWetext004()
    {
        string? dir = FstDirectory();
        if (dir is null)
        {
            _out.WriteLine("SKIPPED: set WETEXT_FST_DIR to a folder holding en/tn and zh/tn tagger.fst + verbalizer.fst");
            return;
        }
        using JsonDocument fixture = JsonDocument.Parse(File.ReadAllText(
            Path.Combine(AppContext.BaseDirectory, "Fixtures", "wetext_parity.json")));
        foreach (JsonProperty file in fixture.RootElement.GetProperty("fst_sha256").EnumerateObject())
        {
            string actual = Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(Path.Combine(dir, file.Name))));
            Assert.True(actual == file.Value.GetString(), $"{file.Name} in {dir} is not the FST the fixture was made from");
        }

        WeTextNormalizer normalizer = WeTextNormalizer.Load(dir);
        Assert.False(normalizer.TryNormalize("5\0 cats", out _));
        Assert.False(normalizer.TryNormalize("5 \ud800 cats", out _));
        int total = 0, matched = 0;
        Dictionary<string, (int Total, int Matched)> perLanguage = [];
        StringBuilder misses = new();
        foreach (JsonElement entry in fixture.RootElement.GetProperty("entries").EnumerateArray())
        {
            string text = entry.GetProperty("text").GetString()!;
            string lang = entry.GetProperty("lang").GetString()!;
            string? wantTagged = entry.GetProperty("tagged").GetString();
            string? wantOutput = entry.GetProperty("output").GetString();
            WeTextLanguage language = lang == "zh" ? WeTextLanguage.Chinese : WeTextLanguage.English;
            string? haveTagged = wantTagged is null ? null : normalizer.Tag(text, language);
            string? haveOutput = normalizer.TryNormalize(text, out string? normalized) ? normalized : null;
            bool ok = haveTagged == wantTagged && haveOutput == wantOutput;
            if (wantOutput is null) Assert.Throws<FormatException>(() => normalizer.Normalize(text));

            total++;
            matched += ok ? 1 : 0;
            (int t, int m) = perLanguage.GetValueOrDefault(lang);
            perLanguage[lang] = (t + 1, m + (ok ? 1 : 0));
            if (!ok)
                misses.AppendLine($"{text}\n want tag {wantTagged}\n have tag {haveTagged}\n"
                    + $" want '{wantOutput}'\n have '{haveOutput}'");
        }
        foreach ((string lang, (int t, int m)) in perLanguage) _out.WriteLine($"{lang}: {m}/{t} lines match");
        Assert.True(total >= 150, $"the fixture holds only {total} lines");
        Assert.True(matched == total, $"{matched}/{total} lines match wetext 0.0.4\n{misses}");
    }

    private static string? FstDirectory()
    {
        string? env = Environment.GetEnvironmentVariable("WETEXT_FST_DIR");
        IEnumerable<string> candidates = string.IsNullOrEmpty(env) ? LocalFstDirs : [env];
        foreach (string dir in candidates)
        {
            if (WeTextNormalizer.RequiredFiles().All(f => File.Exists(Path.Combine(dir, f)))) return dir;
        }
        return null;
    }
}
