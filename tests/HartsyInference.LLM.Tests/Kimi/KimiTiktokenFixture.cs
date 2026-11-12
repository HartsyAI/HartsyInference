using System.Globalization;
using System.Text;
using System.Text.Json;
using HartsyInference.LLM.Kimi;

namespace HartsyInference.LLM.Tests.Kimi;

/// <summary>A tiny tiktoken directory in the Moonshot layout: 256 single bytes, three merges, and the named Kimi specials right above them, with the
/// rest of the reserved range left as <c>&lt;|reserved_token_N|&gt;</c>. Enough to exercise the loader without the real 163,584-rank file.</summary>
internal static class KimiTiktokenFixture
{
    /// <summary>The number of rank lines; the first special id.</summary>
    internal const int BaseRanks = 259;

    internal const int BosId = 259, EosId = 261, OpenId = 262, CloseId = 263, SepId = 264;

    /// <summary>The named specials, in the order the fixture's <c>tokenizer_config.json</c> lists them.</summary>
    private static readonly (int Id, string Literal)[] Named =
    [
        (BosId, "[BOS]"), (260, "[EOS]"), (EosId, "<|end_of_msg|>"), (OpenId, "<|open|>"), (CloseId, "<|close|>"), (SepId, "<|sep|>"),
    ];

    /// <summary>Writes <c>tiktoken.model</c> and <c>tokenizer_config.json</c> into <paramref name="directory"/>.</summary>
    internal static void Write(string directory)
    {
        Directory.CreateDirectory(directory);
        List<string> lines = [];
        for (int b = 0; b < 256; b++)
            lines.Add($"{Convert.ToBase64String([(byte)b])} {b}");
        lines.Add($"{Convert.ToBase64String(Encoding.UTF8.GetBytes("he"))} 256");
        lines.Add($"{Convert.ToBase64String(Encoding.UTF8.GetBytes("ll"))} 257");
        lines.Add($"{Convert.ToBase64String(Encoding.UTF8.GetBytes("hell"))} 258");
        File.WriteAllLines(Path.Combine(directory, KimiK31Tokenizer.TiktokenFile), lines);

        Dictionary<string, object> decoder = [];
        foreach ((int id, string literal) in Named)
            decoder[id.ToString(System.Globalization.CultureInfo.InvariantCulture)] = new Dictionary<string, object> { ["content"] = literal, ["special"] = true };
        File.WriteAllText(Path.Combine(directory, "tokenizer_config.json"),
            JsonSerializer.Serialize(new Dictionary<string, object> { ["added_tokens_decoder"] = decoder }));
    }
}
