using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using HartsyInference.Audio.Pipelines;

namespace HartsyInference.Audio.Tests;

/// <summary><c>Fixtures/Frontend/indextts_frontend.json</c> (tools/parity/indextts_frontend_dump.py) and a lookup FST
/// normalizer built from the wetext calls it recorded, so the frontend around the FST can be checked exactly without a
/// WeText runtime.</summary>
internal static partial class IndexTtsFrontendFixture
{
    public const string ModelsRootVariable = "HARTSY_INDEXTTS_MODELS_ROOT";
    public static readonly string[] Versions = ["v1_5", "v2_0", "v2_5"];

    private static readonly Lazy<JsonDocument> Document = new(() => JsonDocument.Parse(File.ReadAllText(
        Path.Combine(AppContext.BaseDirectory, "Fixtures", "Frontend", "indextts_frontend.json"))));

    private static readonly Lazy<Dictionary<string, (string? Output, string? Error)>> Calls = new(BuildCalls);

    public static JsonElement Root => Document.Value.RootElement;

    public static IEnumerable<JsonElement> Cases => Root.GetProperty("cases").EnumerateArray();

    public static string ModelsRoot =>
        Environment.GetEnvironmentVariable(ModelsRootVariable) is { Length: > 0 } root ? root : "/mnt/model-storage/Models/audio/tts";

    /// <summary>FST normalizers answering from the recorded wetext calls; an unrecorded input fails the test.</summary>
    public static IndexTtsFstNormalizers Lookup { get; } = new()
    {
        English = text => Answer("en", text),
        Chinese = text => Answer("zh", text),
    };

    public static IndexTtsTextFrontend Frontend(string version) =>
        new(new IndexTtsFrontNormalizer(Lookup, enableGlossary: version != "v1_5"));

    public static string[] Strings(JsonElement array) => [.. array.EnumerateArray().Select(static e => e.GetString()!)];

    public static int[] Ints(JsonElement array) => [.. array.EnumerateArray().Select(static e => e.GetInt32())];

    public static string? NullableString(JsonElement e, string name) =>
        e.GetProperty(name).ValueKind == JsonValueKind.Null ? null : e.GetProperty(name).GetString();

    private static string Answer(string lang, string text)
    {
        (string canonical, List<string> letters) = Canonicalize(text);
        if (!Calls.Value.TryGetValue(lang + "\u0001" + canonical, out (string? Output, string? Error) answer))
            throw new KeyNotFoundException($"wetext {lang} input not recorded in the fixture: {text}");
        if (answer.Error is not null) throw new InvalidOperationException($"recorded wetext failure: {answer.Error}");
        return Rename(answer.Output!, letters);
    }

    private static Dictionary<string, (string?, string?)> BuildCalls()
    {
        Dictionary<string, (string?, string?)> calls = new(StringComparer.Ordinal);
        foreach (JsonElement c in Cases)
        {
            foreach (string version in Versions)
            {
                foreach (JsonElement call in c.GetProperty(version).GetProperty("fst_calls").EnumerateArray())
                {
                    (string canonical, List<string> letters) = Canonicalize(call.GetProperty("input").GetString()!);
                    string? output = call.TryGetProperty("output", out JsonElement o) ? ToCanonical(o.GetString()!, letters) : null;
                    string? error = call.TryGetProperty("error", out JsonElement e) ? e.GetString() : null;
                    calls[call.GetProperty("lang").GetString()! + "\u0001" + canonical] = (output, error);
                }
            }
        }
        return calls;
    }

    // Upstream's placeholder letters follow Python set order; key them by order of first appearance instead.
    private static (string Canonical, List<string> Letters) Canonicalize(string text)
    {
        List<string> seen = [];
        foreach (Match m in PlaceholderPattern().Matches(text))
            if (!seen.Contains(m.Value)) seen.Add(m.Value);
        return (ToCanonical(text, seen), seen);
    }

    private static string ToCanonical(string text, List<string> placeholders)
    {
        Dictionary<string, string> map = new(StringComparer.Ordinal);
        Dictionary<string, int> perTag = new(StringComparer.Ordinal);
        foreach (string p in placeholders)
        {
            string tag = PlaceholderPattern().Match(p).Groups[1].Value;
            int n = perTag.GetValueOrDefault(tag);
            perTag[tag] = n + 1;
            map[p] = "<" + tag + "_#" + n + ">";
        }
        return PlaceholderPattern().Replace(text, m => map.TryGetValue(m.Value, out string? c) ? c : m.Value);
    }

    private static string Rename(string canonical, List<string> placeholders)
    {
        StringBuilder sb = new(canonical);
        Dictionary<string, int> perTag = new(StringComparer.Ordinal);
        foreach (string p in placeholders)
        {
            string tag = PlaceholderPattern().Match(p).Groups[1].Value;
            int n = perTag.GetValueOrDefault(tag);
            perTag[tag] = n + 1;
            sb.Replace("<" + tag + "_#" + n + ">", p);
        }
        return sb.ToString();
    }

    [GeneratedRegex(@"<(pinyin|n)_[a-z{|}~]>")]
    private static partial Regex PlaceholderPattern();
}
