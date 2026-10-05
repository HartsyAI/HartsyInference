using HartsyInference.Audio.Frontends;
using HartsyInference.Audio.Phonemizer.MeCab;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>The dictionary-free parts of Kokoro's Japanese front-end. Expected strings are what misaki's
/// <c>num2kana.Convert</c>, <c>Cutlet._normalize_text</c>, <c>jaconv.kata2hira</c> and <c>Cutlet._romaji_word</c>
/// print; the UniDic-backed whole pipeline is covered by <see cref="KokoroJapaneseParityTests"/>.</summary>
public sealed class KokoroJapaneseG2PTests
{
    [Theory]
    [InlineData("0", "ゼロ")]
    [InlineData("007", "なな")]
    [InlineData("11", "じゅういち")]
    [InlineData("35", "さんじゅうご")]
    [InlineData("105", "ひゃくご")]
    [InlineData("310", "さんびゃくじゅう")]
    [InlineData("600", "ろっぴゃく")]
    [InlineData("800", "はっぴゃく")]
    [InlineData("1005", "せんご")]
    [InlineData("3000", "さんぜん")]
    [InlineData("8000", "はっせん")]
    [InlineData("2024", "にせんにじゅうよん")]
    [InlineData("10001", "いちまんいち")]
    [InlineData("100000", "じゅうまん")]
    [InlineData("10000000", "いっせんまん")]
    [InlineData("300000000", "さんおく")]
    [InlineData("123456789", "いちおくにせんさんびゃくよんじゅうごまんろくせんななひゃくはちじゅうきゅう")]
    public void NumberKana_MatchesNum2Kana(string digits, string expected)
        => Assert.Equal(expected, JapaneseNumberKana.Convert(digits));

    [Theory]
    [InlineData("ＡＢＣ１２３", "ABC ひゃくにじゅうさん")]
    [InlineData("ｶﾀｶﾅ", "カタカナ")]
    [InlineData("5〜10人", " ごから じゅう人")]
    [InlineData("ㇰㇱ", "クシ")]
    [InlineData("‘a’ “b”", "`a' “b\"")]
    [InlineData("3.14", " さん. じゅうよん")]
    public void Normalize_MatchesCutlet(string text, string expected)
        => Assert.Equal(expected, KokoroJapaneseG2P.Normalize(text));

    [Fact]
    public void KataToHira_MatchesJaconv()
        => Assert.Equal("かたかな・ゔゕゖヷーゝゞ漢字", KokoroJapaneseG2P.KataToHira("カタカナ・ヴヵヶヷーヽヾ漢字"));

    [Theory]
    // ん assimilates to the next kana; っ is a glottal stop; small kana fuse into digraphs; ゞ repeats voiced.
    [InlineData("かんぱい", "kampai")]
    [InlineData("しんぶん", "ɕimbɯɴ")]
    [InlineData("こんにちは", "koɲɲiʨiha")]
    [InlineData("きっぷ", "kʲiʔpɯ")]
    [InlineData("ぎゅうにゅう", "ɡʲɨɯɲɨɯ")]
    [InlineData("ほんや", "hoɴja")]
    [InlineData("てぃー", "tʲiː")]
    [InlineData("かゞみ", "kaɡamʲi")]
    [InlineData("ふぁいる", "ɸaiɾɯ")]
    [InlineData("あゃ", "ja")]
    public void KanaReading_MatchesCutlet(string hira, string expected)
        => Assert.Equal(expected, JapaneseKanaIpa.MapReading(hira));

    [Fact]
    public void KanaTable_HasMisakisEntryCount() => Assert.Equal(189, JapaneseKanaIpa.Table.Count);

    [Fact]
    public void FeatureField_UnquotesCsvAndPadsShortFeatures()
    {
        MeCabToken token = new("た", "助動詞,*,ダ,\"動詞%F2@1,形容詞%F4@-2\",x\"\"y", 6, false);
        Assert.Equal("助動詞", token.Field(0));
        Assert.Equal("ダ", token.Field(2));
        Assert.Equal("動詞%F2@1,形容詞%F4@-2", token.Field(3));
        Assert.Equal("x\"\"y", token.Field(4));
        Assert.Null(token.Field(5));
    }

    [Fact]
    public void CharInfo_DecodesMeCabBitfield()
    {
        // KATAKANA in UniDic's char.bin: category bit 7, default type 7, length 2, group and invoke set.
        MeCabCharInfo info = new(128u | (7u << 18) | (2u << 26) | (1u << 30) | (1u << 31));
        Assert.Equal(128u, info.Type);
        Assert.Equal(7, info.DefaultType);
        Assert.Equal(2, info.Length);
        Assert.True(info.Group);
        Assert.True(info.Invoke);
        Assert.True(info.IsKindOf(new MeCabCharInfo(128u | 1u)));
        Assert.False(info.IsKindOf(new MeCabCharInfo(64u)));
    }
}
