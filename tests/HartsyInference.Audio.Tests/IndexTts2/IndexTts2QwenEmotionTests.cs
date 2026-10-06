using HartsyInference.Audio.Models.IndexTts2;
using Xunit;

namespace HartsyInference.Audio.Tests.IndexTts2;

/// <summary>Exercises <see cref="IndexTts2QwenEmotion.ParseAndConvert"/> — the pure parsing/normalization/
/// conversion logic ported from the real <c>QwenEmotion</c> Python class — without a real model. Says nothing
/// about the actual Qwen3 classifier's text-understanding quality.</summary>
public sealed class IndexTts2QwenEmotionTests
{
    private static int IndexOf(string category) => IndexTts2QwenEmotion.EnglishCategoryOrder.ToList().IndexOf(category);

    [Fact]
    public void EnglishCategoryOrder_MatchesTheRealDesiredVectorOrder()
    {
        Assert.Equal(["happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm"],
            IndexTts2QwenEmotion.EnglishCategoryOrder);
    }

    [Fact]
    public void ParseAndConvert_PlainJsonObject_MapsChineseKeysToTheRightSlots()
    {
        string json = """{"高兴": 0.8, "愤怒": 0.1}""";
        float[] result = IndexTts2QwenEmotion.ParseAndConvert(json, "a happy sentence");

        Assert.Equal(0.8f, result[IndexOf("happy")], 4);
        Assert.Equal(0.1f, result[IndexOf("angry")], 4);
        Assert.Equal(0f, result[IndexOf("sad")], 4);
    }

    [Fact]
    public void ParseAndConvert_ClampsScoresToTheRealMinMaxRange()
    {
        string json = """{"高兴": 5.0, "愤怒": -3.0}""";
        float[] result = IndexTts2QwenEmotion.ParseAndConvert(json, "");

        Assert.Equal(1.2f, result[IndexOf("happy")], 4);
        Assert.Equal(0f, result[IndexOf("angry")], 4);
    }

    [Fact]
    public void ParseAndConvert_AllZero_DefaultsToCalm()
    {
        string json = """{"高兴": 0.0, "愤怒": 0.0}""";
        float[] result = IndexTts2QwenEmotion.ParseAndConvert(json, "");

        Assert.Equal(1f, result[IndexOf("calm")], 4);
        for (int i = 0; i < result.Length - 1; i++) Assert.Equal(0f, result[i], 4);
    }

    [Fact]
    public void ParseAndConvert_MalformedJson_FallsBackToRegexParsing()
    {
        // Not valid JSON (missing braces/quotes around keys) — real fallback: re.finditer(r'([^\s":.,]+?)"?\s*:\s*([\d.]+)').
        string malformed = "高兴: 0.9 愤怒: 0.2";
        float[] result = IndexTts2QwenEmotion.ParseAndConvert(malformed, "");

        Assert.Equal(0.9f, result[IndexOf("happy")], 4);
        Assert.Equal(0.2f, result[IndexOf("angry")], 4);
    }

    [Fact]
    public void ParseAndConvert_BareStringLabel_SetsThatCategoryToOne()
    {
        string json = "\"happy\"";
        float[] result = IndexTts2QwenEmotion.ParseAndConvert(json, "");

        Assert.Equal(1f, result[IndexOf("happy")], 4);
        for (int i = 0; i < result.Length; i++) if (i != IndexOf("happy")) Assert.Equal(0f, result[i], 4);
    }

    [Fact]
    public void ParseAndConvert_AliasLabelKey_IsRecognized()
    {
        string json = """{"label": "angry"}""";
        float[] result = IndexTts2QwenEmotion.ParseAndConvert(json, "");

        Assert.Equal(1f, result[IndexOf("angry")], 4);
    }

    [Theory]
    [InlineData("melancholic")]
    [InlineData("depression")]
    [InlineData("低落")]
    public void ParseAndConvert_MelancholicWordInInput_SwapsSadAndMelancholicVectors(string word)
    {
        // Model detected "sad" (悲伤) with high confidence, but the input text names a melancholic word —
        // the real workaround swaps the two vectors so the OUTPUT ends up "melancholic", not "sad".
        string json = """{"悲伤": 0.9, "低落": 0.0}""";
        float[] result = IndexTts2QwenEmotion.ParseAndConvert(json, $"I feel so {word} today");

        Assert.Equal(0f, result[IndexOf("sad")], 4);
        Assert.Equal(0.9f, result[IndexOf("melancholic")], 4);
    }

    [Fact]
    public void ParseAndConvert_StripsAThinkBlock_BeforeParsing()
    {
        string withThink = "<think>reasoning about the emotion here</think>{\"高兴\": 0.7}";
        float[] result = IndexTts2QwenEmotion.ParseAndConvert(withThink, "");

        Assert.Equal(0.7f, result[IndexOf("happy")], 4);
    }

    [Fact]
    public void ParseAndConvert_NoWordMatch_DoesNotSwap()
    {
        string json = """{"悲伤": 0.6, "低落": 0.1}""";
        float[] result = IndexTts2QwenEmotion.ParseAndConvert(json, "a perfectly ordinary sentence");

        Assert.Equal(0.6f, result[IndexOf("sad")], 4);
        Assert.Equal(0.1f, result[IndexOf("melancholic")], 4);
    }
}
