using System.Text;
using HartsyInference.Audio.Frontends;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>The pieces of Kokoro's Mandarin front-end over tiny in-memory dictionaries. Expected strings are what
/// misaki's <c>ZHG2P</c>, cn2an's <c>transform(text, "an2cn")</c> and <c>ZHG2P.py2ipa</c> print.</summary>
public sealed class KokoroMandarinG2PTests
{
    private const string JiebaDict = "我 200 r\n在 200 p\n二 100 m\n银行 100 n\n工作 100 vn\n个人 100 n\n";
    // HMM emissions: 我/在/二 read as single-character words, 韩冰 as one begin-end word.
    private const string ProbEmit = """
        P={'B': {'韩': -1.0},
         'E': {'冰': -1.0},
         'M': {'一': -5.0},
         'S': {'我': -1.0,
               '在': -1.0,
               '二': -1.0}}
        """;
    private const string Phrases = """
        {"银行": [["yín"], ["háng", "xíng"]], "工作": [["gōng"], ["zuò"]], "个人": [["gè"], ["rén"]]}
        """;

    private static KokoroMandarinG2P G2P()
    {
        string singles = $"{{\"{(int)'我'}\": \"wǒ\", \"{(int)'在'}\": \"zài\", \"{(int)'二'}\": \"èr,ér\"}}";
        using MemoryStream dict = new(Encoding.UTF8.GetBytes(JiebaDict));
        using MemoryStream emit = new(Encoding.UTF8.GetBytes(ProbEmit));
        using MemoryStream phrases = new(Encoding.UTF8.GetBytes(Phrases));
        using MemoryStream single = new(Encoding.UTF8.GetBytes(singles));
        return KokoroMandarinG2P.FromStreams(dict, emit, phrases, single);
    }

    [Fact]
    public void ToIpa_MatchesMisakiZhG2P()
        => Assert.Equal("wo↓ ʦai↘ i↗nxa↗ŋ kʊ→ŋʦwo↘, ɚ↘ kɤ↘ɻə↗n.", G2P().ToIpa("我在银行工作，2个人。"));

    [Fact]
    public void Segment_RoutesThroughDictionaryAndHmm()
    {
        KokoroMandarinG2P g2p = G2P();
        Assert.Equal(["我", "在", "银行", "工作"], g2p.Segment("我在银行工作"));
        Assert.Equal(["韩冰", "工作"], g2p.Segment("韩冰工作"));
    }

    [Theory]
    [InlineData("10", "十")]
    [InlineData("1010", "一千零一十")]
    [InlineData("100010000", "一亿零一万")]
    [InlineData("3.50", "三点五零")]
    [InlineData("-0", "负零")]
    [InlineData("１２３", "一百二十三")]
    [InlineData("2008年8月8日", "二零零八年八月八日")]
    [InlineData("12345年5月", "一二三四五年五月")]
    [InlineData("2024-01-01", "二千零二十四负一负一")]
    [InlineData("3-5个", "三到五个")]
    [InlineData("1/3", "三分之一")]
    [InlineData("30℃", "三十摄氏度")]
    [InlineData("-5%", "百分之负五")]
    [InlineData("12345678901234567", "12345678901234567")]
    [InlineData("0.12345678901234567891", "零点一二三四五六七八九零一二三四五六")]
    public void Numbers_ReadAsCn2an(string text, string expected)
        => Assert.Equal(expected, ChineseNumberNormalizer.Transform(text));

    [Theory]
    [InlineData("zhong1", "ꭧʊ→ŋ")]
    [InlineData("lv4", "ly↘")]
    [InlineData("ma5", "ma")]
    [InlineData("shi4", "ʂɨ↘")]
    [InlineData("zi3", "ʦɨ↓")]
    [InlineData("er2", "ɚ↗")]
    [InlineData("yo1", "wo→")]
    [InlineData("n2", "n↗")]
    [InlineData("hm5", "hm")]
    [InlineData("jun1", "ʨy→n")]
    [InlineData("gui4", "kwei̯↘")]
    [InlineData("liu2", "ljou̯↗")]
    [InlineData("wen4", "wə↘n")]
    [InlineData("yu3", "y↓")]
    [InlineData("xue2", "ɕɥe↗")]
    [InlineData("yuan2", "ɥɛ↗n")]
    public void Syllables_ReadAsPy2Ipa(string tone3, string expected) => Assert.Equal(expected, PinyinToIpa.Convert(tone3));

    [Theory]
    [InlineData("zhōng", "zhong1")]
    [InlineData("lǜ", "lv4")]
    [InlineData("lü", "lv5")]
    [InlineData("de", "de5")]
    [InlineData("ḿ", "m2")]
    [InlineData("m̀", "m4")]
    [InlineData("ê̄", "ê1")]
    public void ToneMarks_BecomeTone3(string pinyin, string expected) => Assert.Equal(expected, PinyinConverter.ToTone3(pinyin));
}
