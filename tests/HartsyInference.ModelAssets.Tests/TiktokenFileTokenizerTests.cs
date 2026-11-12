using System.Text;
using HartsyInference.ModelAssets.Tokenizers;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>The shared tiktoken-format loader on a small synthetic rank file: rank-ordered BPE, special tokens, reserved fill and validation.</summary>
public sealed class TiktokenFileTokenizerTests : IDisposable
{
    // Rank-file layout: 256 single bytes at ranks 0..255 (rank = byte value), then "he" (256), "ll" (257) and "hell" (258) = "he" + "ll".
    private const string Pattern = @"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";
    private const int HeId = 256, HellId = 258;
    private const int OpenId = 259, EndId = 260;

    private readonly string _root = Path.Combine(Path.GetTempPath(), "hartsy-tiktoken-" + Guid.NewGuid().ToString("N"));

    public TiktokenFileTokenizerTests() => Directory.CreateDirectory(_root);

    public void Dispose()
    {
        if (Directory.Exists(_root)) Directory.Delete(_root, recursive: true);
    }

    [Fact]
    public void Byte_pairs_merge_in_rank_order()
    {
        GgufTokenizer tokenizer = Load(Specials());

        Assert.Equal([HellId, (int)'o'], tokenizer.EncodeOrdinary("hello"));
    }

    [Fact]
    public void Ordinary_encode_then_decode_round_trips_text_with_multibyte_characters()
    {
        GgufTokenizer tokenizer = Load(Specials());
        const string text = "hello, wörld 😀 — 日本語";

        int[] ids = tokenizer.Encode(text, addSpecial: false);

        Assert.Equal(text, tokenizer.Decode(ids));
    }

    [Fact]
    public void Special_literals_map_to_their_ids_only_when_special_handling_is_on()
    {
        GgufTokenizer tokenizer = Load(Specials());

        Assert.Equal(OpenId, tokenizer.SpecialId("<|open|>"));
        Assert.Equal([OpenId, HellId], tokenizer.Encode("<|open|>hell", addSpecial: true));
        Assert.DoesNotContain(OpenId, tokenizer.EncodeOrdinary("<|open|>"));
    }

    [Fact]
    public void Begin_and_end_ids_take_their_literals_from_the_special_table()
    {
        GgufTokenizer tokenizer = Load(Specials(), bosId: OpenId, eosId: EndId);

        Assert.Equal(OpenId, tokenizer.BosId);
        Assert.Equal(EndId, tokenizer.EosId);
        Assert.Equal("<|open|>", tokenizer.BosToken);
        Assert.Contains(EndId, tokenizer.StopIds);
    }

    [Fact]
    public void A_begin_id_that_names_no_special_is_refused()
    {
        Assert.Throws<InvalidDataException>(() => Load(Specials(), bosId: 999));
    }

    [Fact]
    public void Non_dense_ranks_are_refused()
    {
        string path = WriteRanks("gaps.tiktoken", [(0, "AA=="), (2, "AQ==")]);

        Assert.Throws<InvalidDataException>(() => TiktokenFileTokenizer.Load(path, Pattern, Specials()));
    }

    [Fact]
    public void A_duplicate_special_literal_or_id_is_refused()
    {
        Assert.Throws<InvalidDataException>(() => Load([("<|a|>", OpenId), ("<|a|>", OpenId + 1)]));
        Assert.Throws<InvalidDataException>(() => Load([("<|a|>", OpenId), ("<|b|>", OpenId)]));
    }

    [Fact]
    public void A_missing_rank_file_is_reported_by_path()
    {
        string path = Path.Combine(_root, "absent.tiktoken");

        FileNotFoundException missing = Assert.Throws<FileNotFoundException>(() => TiktokenFileTokenizer.Load(path, Pattern, Specials()));
        Assert.Equal(path, missing.FileName);
    }

    [Fact]
    public void Count_ranks_counts_the_non_blank_lines()
    {
        string path = WriteRanks("count.tiktoken", Ranks());

        Assert.Equal(HellId + 1, TiktokenFileTokenizer.CountRanks(path));
    }

    [Fact]
    public void Reserved_range_names_unnamed_ids_and_drops_named_ids_outside_it()
    {
        Dictionary<int, string> named = new() { [5] = "<|a|>", [9] = "<|outside|>" };

        IReadOnlyList<(string Token, int Id)> specials = TiktokenFileTokenizer.WithReservedRange(4, 3, named);

        Assert.Equal([("<|reserved_token_4|>", 4), ("<|a|>", 5), ("<|reserved_token_6|>", 6)], specials);
    }

    private GgufTokenizer Load(IReadOnlyList<(string Token, int Id)> specials, int? bosId = null, int? eosId = null) =>
        TiktokenFileTokenizer.Load(WriteRanks("synthetic.tiktoken", Ranks()), Pattern, specials, bosId, eosId);

    private static IReadOnlyList<(string Token, int Id)> Specials() => [("<|open|>", OpenId), ("[EOS]", EndId)];

    private static List<(int Rank, string Base64)> Ranks()
    {
        List<(int, string)> ranks = [];
        for (int b = 0; b < 256; b++)
            ranks.Add((b, Convert.ToBase64String([(byte)b])));
        ranks.Add((HeId, Convert.ToBase64String(Encoding.UTF8.GetBytes("he"))));
        ranks.Add((257, Convert.ToBase64String(Encoding.UTF8.GetBytes("ll"))));
        ranks.Add((HellId, Convert.ToBase64String(Encoding.UTF8.GetBytes("hell"))));
        return ranks;
    }

    private string WriteRanks(string name, IEnumerable<(int Rank, string Base64)> ranks)
    {
        string path = Path.Combine(_root, name);
        File.WriteAllLines(path, ranks.Select(r => $"{r.Base64} {r.Rank}"));
        return path;
    }
}
