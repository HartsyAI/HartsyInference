using System.Text.RegularExpressions;
using HartsyInference.LLM.Kimi;
using HartsyInference.ModelAssets.Tokenizers;
using Xunit;

namespace HartsyInference.LLM.Tests.Kimi;

/// <summary>The Kimi K3.1 tiktoken tokenizer: its pre-tokenizer against splits of the published <c>tokenization_kimi.py</c> pattern, and the special-token
/// table built from a Moonshot-layout directory.</summary>
public sealed class KimiK31TokenizerTests : IDisposable
{
    // Pre-token splits of tokenization_kimi.py's pat_str (run with the Python regex module, V1 sets), one row per input.
    private static readonly (string Text, string[] Pieces)[] PreTokenSplits =
    [
        ("Hello world", ["Hello", " world"]),
        ("Don't stop", ["Don't", " stop"]),
        ("I'M HERE, you're THE best've", ["I'M", " HERE", ",", " you're", " THE", " best've"]),
        ("x = 12345678", ["x", " =", " ", "123", "456", "78"]),
        ("日本語のテキスト", ["日本語", "のテキスト"]),
        ("中文和English混合", ["中文和", "English", "混合"]),
        ("  leading spaces and\n\nnewlines\r\n", [" ", " leading", " spaces", " and", "\n\n", "newlines", "\r\n"]),
        ("café naïve Ångström", ["café", " naïve", " Ångström"]),
        ("emoji 😀 done!!", ["emoji", " 😀", " done", "!!"]),
        ("a\tb", ["a", "\tb"]),
        ("price: $3.50 (approx.)", ["price", ":", " $", "3", ".", "50", " (", "approx", ".)"]),
        ("Ⅻ 한국어 Ελληνικά", ["Ⅻ", " 한국어", " Ελληνικά"]),
    ];

    private readonly string _root = Path.Combine(Path.GetTempPath(), "hartsy-kimi-tok-" + Guid.NewGuid().ToString("N"));

    public KimiK31TokenizerTests() => KimiTiktokenFixture.Write(_root);

    public void Dispose()
    {
        if (Directory.Exists(_root)) Directory.Delete(_root, recursive: true);
    }

    [Theory]
    [MemberData(nameof(SplitRows))]
    public void Pre_tokenizer_splits_match_the_published_pattern(string text, string[] pieces)
    {
        Regex pattern = new(KimiK31Tokenizer.PreTokenPattern);

        string[] actual = pattern.Matches(text).Select(m => m.Value).ToArray();

        Assert.Equal(pieces, actual);
    }

    public static IEnumerable<object[]> SplitRows() => PreTokenSplits.Select(row => new object[] { row.Text, row.Pieces });

    [Fact]
    public void Loading_names_the_specials_from_the_config_and_fills_the_rest_of_the_reserved_range()
    {
        ILlmTokenizer tokenizer = KimiK31Tokenizer.Load(_root, [KimiTiktokenFixture.EosId], KimiTiktokenFixture.BosId, KimiTiktokenFixture.EosId);

        Assert.Equal(KimiTiktokenFixture.BosId, tokenizer.BosId);
        Assert.Equal(KimiTiktokenFixture.EosId, tokenizer.EosId);
        Assert.Equal("[BOS]", tokenizer.BosToken);
        Assert.Equal("<|end_of_msg|>", tokenizer.EosToken);
        Assert.Equal(KimiTiktokenFixture.OpenId, tokenizer.SpecialId("<|open|>"));
        Assert.Equal(KimiTiktokenFixture.SepId, tokenizer.SpecialId("<|sep|>"));
        Assert.Equal(KimiTiktokenFixture.BaseRanks + 40, tokenizer.SpecialId("<|reserved_token_299|>"));
        int lastReserved = KimiTiktokenFixture.BaseRanks + KimiK31Tokenizer.ReservedSpecialCount - 1;
        Assert.Equal(lastReserved, tokenizer.SpecialId($"<|reserved_token_{lastReserved}|>"));
    }

    [Fact]
    public void Config_ids_that_name_no_special_are_refused_as_malformed()
    {
        Assert.Throws<InvalidDataException>(() => KimiK31Tokenizer.Load(_root, [], bosId: 100_000, eosId: null));
    }

    [Fact]
    public void Ordinary_encode_leaves_control_literals_as_text_and_special_encode_recognises_them()
    {
        ILlmTokenizer tokenizer = KimiK31Tokenizer.Load(_root, [], KimiTiktokenFixture.BosId, KimiTiktokenFixture.EosId);

        int[] text = tokenizer.EncodeOrdinary("<|open|>");
        int[] special = tokenizer.Encode("<|open|>", addSpecial: true);

        Assert.DoesNotContain(KimiTiktokenFixture.OpenId, text);
        Assert.Equal([KimiTiktokenFixture.OpenId], special);
    }

    [Fact]
    public void A_directory_without_a_tiktoken_file_is_refused()
    {
        string empty = Path.Combine(_root, "no-tiktoken");
        Directory.CreateDirectory(empty);

        Assert.Throws<FileNotFoundException>(() => KimiK31Tokenizer.Load(empty, [], null, null));
    }
}
