using System.Text.Json;
using HartsyInference.Audio.Frontends.WeText;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary><see cref="IndexTtsFstNormalizers.FromWeText"/> against wetext 0.1.8: every English/Chinese call upstream
/// IndexTTS made on the corpus in <c>Fixtures/Frontend/indextts_frontend.json</c> must give the recorded output, and the
/// whole 1.5/2.0/2.5 normalization must then match with the real FSTs in place of the recorded lookup. Needs the wetext
/// grammars: <c>WETEXT_FST_DIR</c>, else the reference venv's wetext 0.1.8 wheel.</summary>
[Trait("Category", "Integration")]
public sealed class IndexTtsWeTextParityTests(ITestOutputHelper output)
{
    private static string FstDirectory =>
        Environment.GetEnvironmentVariable("WETEXT_FST_DIR") is { Length: > 0 } dir
            ? dir
            : "/mnt/model-storage/scratch/tts-ref/fe-venv/lib/python3.12/site-packages/wetext/fsts";

    [Fact]
    public void RecordedFstCalls_AndFullNormalization_MatchUpstream()
    {
        if (!RealWeightGate.Require(output.WriteLine, Path.Combine(FstDirectory, "en", "tn", "tagger.fst"),
                Path.Combine(FstDirectory, "zh", "tn", "verbalizer.fst"))) return;
        IndexTtsFstNormalizers fst = IndexTtsFstNormalizers.FromWeText(WeTextNormalizer.Load(FstDirectory));
        List<string> failures = [];
        int calls = 0;
        foreach (JsonElement c in IndexTtsFrontendFixture.Cases)
        {
            foreach (string version in IndexTtsFrontendFixture.Versions)
            {
                foreach (JsonElement call in c.GetProperty(version).GetProperty("fst_calls").EnumerateArray())
                {
                    calls++;
                    string input = call.GetProperty("input").GetString()!;
                    Func<string, string> run = call.GetProperty("lang").GetString() == "en" ? fst.English : fst.Chinese;
                    string actual = run(input);
                    if (actual != call.GetProperty("output").GetString()) failures.Add($"fst [{input}]: '{actual}'");
                }
            }
        }
        output.WriteLine($"fst calls: {calls - failures.Count}/{calls} match");

        int lines = 0, lineFailures = 0;
        foreach (string version in IndexTtsFrontendFixture.Versions)
        {
            IndexTtsTextFrontend frontend = new(new IndexTtsFrontNormalizer(fst, enableGlossary: version != "v1_5"));
            foreach (JsonElement c in IndexTtsFrontendFixture.Cases)
            {
                lines++;
                string text = c.GetProperty("text").GetString()!;
                JsonElement v = c.GetProperty(version);
                string? expected = version == "v2_5"
                    ? v.GetProperty("prepared").GetString()
                    : IndexTtsFrontendFixture.NullableString(v, "cjk_tokenized");
                string? actual = version == "v2_5"
                    ? frontend.PrepareTiktoken(text, v.GetProperty("lang").GetString()!).Prepared
                    : frontend.PrepareSentencePiece(text) is { } p && IndexTtsFrontendFixture.NullableString(v, "normalized") is not null ? p : null;
                if (actual != expected)
                {
                    lineFailures++;
                    failures.Add($"{version} [{text}]: '{actual}'");
                }
            }
        }
        output.WriteLine($"normalized lines: {lines - lineFailures}/{lines} match");
        foreach (string f in failures) output.WriteLine(f);
        Assert.True(failures.Count == 0, $"{failures.Count} mismatches; first: {failures.FirstOrDefault()}");
    }
}
