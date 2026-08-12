using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using HartsyInference.ModelAssets.Tokenizers;
using Xunit;

namespace HartsyInference.ModelAssets.Tokenizers.Tests;

/// <summary>Tests <see cref="Gemma4Tokenizer"/> against a hand-built miniature of the real Gemma 4
/// <c>tokenizer.json</c> (same declared pipeline: BPE + byte fallback + the <c>' ' → '▁'</c> Replace normalizer).
/// The real blob is a 32 MB tensor inside the LTX-2.5 checkpoint and cannot ship with the repo, so the
/// conditioning-assembly rules that the pipeline depends on — BOS exactly once, no EOS, right-pad to 1024 — are
/// covered here on data that does.</summary>
public sealed class Gemma4TokenizerTests
{
    /// <summary>Builds a tokenizer.json with all 256 byte-fallback pieces, a handful of ordinary pieces, the four
    /// Gemma specials, and merges whose rank order decides the segmentation.</summary>
    private static string BuildTokenizerJson()
    {
        Dictionary<string, int> vocab = new()
        {
            ["<pad>"] = 0,
            ["<eos>"] = 1,
            ["<bos>"] = 2,
            ["<unk>"] = 3,
        };
        int next = 4;
        for (int b = 0; b < 256; b++) vocab[$"<0x{b:X2}>"] = next++;
        foreach (string piece in new[] { "a", "b", "c", "▁", "▁a", "▁ab", "ab", "abc", "▁abc" })
        {
            if (!vocab.ContainsKey(piece)) vocab[piece] = next++;
        }

        // Rank order is the whole point: "a"+"b" must win over "▁"+"a" so "▁ab" only forms via "▁"+"ab".
        string[][] merges =
        [
            ["a", "b"],
            ["ab", "c"],
            ["▁", "a"],
            ["▁", "ab"],
            ["▁", "abc"],
        ];

        object model = new
        {
            type = "BPE",
            dropout = (string?)null,
            unk_token = "<unk>",
            continuing_subword_prefix = (string?)null,
            end_of_word_suffix = (string?)null,
            fuse_unk = true,
            byte_fallback = true,
            ignore_merges = false,
            vocab,
            merges,
        };
        object root = new
        {
            version = "1.0",
            added_tokens = new object[]
            {
                new { id = 0, content = "<pad>", special = true },
                new { id = 1, content = "<eos>", special = true },
                new { id = 2, content = "<bos>", special = true },
                new { id = 3, content = "<unk>", special = true },
            },
            normalizer = new { type = "Replace", pattern = new { String = " " }, content = "▁" },
            pre_tokenizer = new { type = "Split", pattern = new { String = " " }, behavior = "MergedWithPrevious", invert = false },
            model,
        };
        return JsonSerializer.Serialize(root);
    }

    private static Gemma4Tokenizer Make() =>
        Gemma4Tokenizer.FromTokenizerJson(new MemoryStream(Encoding.UTF8.GetBytes(BuildTokenizerJson())));

    [Fact]
    public void Encode_AddsNoSpecialTokens()
    {
        int[] ids = Make().Encode("abc");
        Assert.DoesNotContain(Gemma4Tokenizer.BosTokenId, ids);
        Assert.DoesNotContain(Gemma4Tokenizer.EosTokenId, ids);
        Assert.NotEmpty(ids);
    }

    [Fact]
    public void Encode_ReplacesSpacesWithMetaSymbol()
    {
        Gemma4Tokenizer tokenizer = Make();
        int[] withSpace = tokenizer.Encode(" abc");
        int[] withoutSpace = tokenizer.Encode("abc");
        Assert.NotEqual(withoutSpace, withSpace);
        Assert.Equal(" abc", tokenizer.Decode(withSpace));
        Assert.Equal("abc", tokenizer.Decode(withoutSpace));
    }

    [Fact]
    public void Encode_AppliesMergesInRankOrder()
    {
        // "a"+"b" (rank 0) then "ab"+"c" (rank 1) beats "▁"+"a" (rank 2), leaving "▁" + "abc" -> "▁abc" (rank 4).
        Gemma4Tokenizer tokenizer = Make();
        Assert.Single(tokenizer.Encode(" abc"));
        Assert.Single(tokenizer.Encode("abc"));
    }

    [Fact]
    public void Encode_UnknownCharacterFallsBackToBytes()
    {
        Gemma4Tokenizer tokenizer = Make();
        // 'é' is U+00E9 -> UTF-8 C3 A9, so two byte pieces and no <unk>.
        int[] ids = tokenizer.Encode("é");
        Assert.Equal(2, ids.Length);
        Assert.DoesNotContain(3, ids);
        Assert.Equal("é", tokenizer.Decode(ids));
    }

    [Fact]
    public void Encode_SpecialLiteralsResolveToTheirIds()
    {
        int[] ids = Make().Encode("<bos>abc");
        Assert.Equal(Gemma4Tokenizer.BosTokenId, ids[0]);
    }

    [Fact]
    public void Conditioning_PrependsBosExactlyOnce_NoEos_PaddedTo1024()
    {
        int[] sequence = Make().EncodeForConditioning("abc");
        Assert.Equal(1024, sequence.Length);
        Assert.Equal(Gemma4Tokenizer.BosTokenId, sequence[0]);
        Assert.Equal(1, sequence.Count(id => id == Gemma4Tokenizer.BosTokenId));
        Assert.DoesNotContain(Gemma4Tokenizer.EosTokenId, sequence);
        Assert.Equal(Gemma4Tokenizer.PadTokenId, sequence[^1]);
    }

    [Fact]
    public void Conditioning_DoesNotDuplicateAnExistingBos()
    {
        int[] sequence = Gemma4Tokenizer.BuildConditioningSequence([Gemma4Tokenizer.BosTokenId, 42, 43]);
        Assert.Equal(1024, sequence.Length);
        Assert.Equal([Gemma4Tokenizer.BosTokenId, 42, 43], sequence[..3]);
        Assert.Equal(1, sequence.Count(id => id == Gemma4Tokenizer.BosTokenId));
    }

    [Fact]
    public void Conditioning_LongerThanMinimumIsNotTruncated()
    {
        int[] ids = Enumerable.Range(10, 20).ToArray();
        int[] sequence = Gemma4Tokenizer.BuildConditioningSequence(ids, minLength: 8);
        Assert.Equal(21, sequence.Length);
        Assert.Equal(Gemma4Tokenizer.BosTokenId, sequence[0]);
    }

    [Fact]
    public void Parse_RejectsANonBpeModel()
    {
        string json = BuildTokenizerJson().Replace("\"type\":\"BPE\"", "\"type\":\"Unigram\"");
        Assert.Throws<NotSupportedException>(() =>
            Gemma4Tokenizer.FromTokenizerJson(new MemoryStream(Encoding.UTF8.GetBytes(json))));
    }
}
