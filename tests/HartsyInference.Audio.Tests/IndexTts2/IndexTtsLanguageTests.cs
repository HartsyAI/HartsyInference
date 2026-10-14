using HartsyInference.Audio.Pipelines;
using HartsyInference.ModelAssets.Tokenizers;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests.IndexTts2;

/// <summary>IndexTTS-2.5's language when the caller names none: Chinese text must be tagged <c>&lt;|zh|&gt;</c>, not the
/// <c>&lt;|en|&gt;</c> every request used to get.</summary>
public sealed class IndexTtsLanguageTests(ITestOutputHelper output)
{
    [Theory]
    [InlineData("你好，世界。", "zh")]
    [InlineData("Hello 你好 world", "zh")]
    [InlineData("The weather is nice today.", "en")]
    [InlineData("1,250.", "en")]
    [InlineData("そうですね。", "en")]
    public void DetectLanguage_IsChineseOnlyForCjkIdeographs(string text, string expected) =>
        Assert.Equal(expected, IndexTtsTextFrontend.DetectLanguage(text));

    [Fact]
    [Trait("Category", "Integration")]
    public void ChineseText_IsEncodedWithTheZhTag()
    {
        string path = Path.Combine(IndexTtsFrontendFixture.ModelsRoot, "IndexTeam--IndexTTS-2.5", "multilingual_zh_ja_yue_char_del.tiktoken");
        if (!RealWeightGate.Require(output.WriteLine, path)) return;
        IndexTts2TiktokenTokenizer tokenizer = new(path);
        IndexTtsTextFrontend frontend = new(new IndexTtsFrontNormalizer(IndexTtsFrontendFixture.Lookup, enableGlossary: true));
        string text = "你好，世界！今天天气很好。";
        List<int[]> segments = frontend.SegmentTiktoken(text, IndexTtsTextFrontend.DetectLanguage(text), tokenizer.Encode, 602);
        Assert.Equal(tokenizer.Encode("<|zh|>")[0], segments[0][0]);
        Assert.NotEqual(tokenizer.Encode("<|en|>")[0], segments[0][0]);
    }
}
