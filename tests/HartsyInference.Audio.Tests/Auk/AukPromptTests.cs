using System.Text;
using Xunit;
using HartsyInference.Audio.Models.Auk;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Audio.Tests.Auk;

/// <summary>Golden prompt strings, token-id layout and template substitution for AuK instructions.</summary>
public sealed class AukPromptTests
{
    private static IReadOnlyList<int> Utf8(string text) => [.. Encoding.UTF8.GetBytes(text).Select(b => (int)b)];

    [Fact]
    public void BuildText_WithAudio_MatchesTheChatTemplate()
    {
        string text = AukPrompt.BuildText("Raise the pitch by 2 semitones.", 3);
        Assert.Equal(
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nRaise the pitch by 2 semitones."
            + "<|audio_bos|><|AUDIO|><|AUDIO|><|AUDIO|><|audio_eos|><|im_end|>\n<|im_start|>assistant\n",
            text);
    }

    [Fact]
    public void BuildIds_SplicesSpecialIdsAroundEncodedSegments()
    {
        int[] ids = AukPrompt.BuildIds("Hi", 2, Utf8);
        List<int> expected = [151644];
        expected.AddRange(Utf8("system\nYou are a helpful assistant."));
        expected.AddRange([151645, 10, 151644]);
        expected.AddRange(Utf8("user\nHi"));
        expected.AddRange([151647, 151646, 151646, 151648, 151645, 10, 151644]);
        expected.AddRange(Utf8("assistant\n"));
        Assert.Equal([.. expected], ids);
        Assert.Equal(2, ids.Count(i => i == 151646));
    }

    [Fact]
    public void BuildIds_RejectsNegativeAudioTokens() =>
        Assert.Throws<ArgumentOutOfRangeException>(() => AukPrompt.BuildIds("x", -1, Utf8));

    [Fact]
    public void Templates_RejectWrongValueCount_AndEveryTaskHasATemplate()
    {
        Assert.Throws<ArgumentException>(() => AukTemplates.Format(AukTask.ReplaceText, "only one"));
        foreach (AukTask task in Enum.GetValues<AukTask>())
        {
            string[] values = [.. AukTemplates.Placeholders(task).Select(p => "<" + p + ">")];
            string formatted = AukTemplates.Format(task, values);
            Assert.DoesNotContain("{", formatted);
        }
        Assert.False(AukTemplates.RequiresAudio(AukTask.InstructTts));
        Assert.True(AukTemplates.RequiresAudio(AukTask.Denoise));
    }

    /// <summary>Opt-in: needs the Qwen2.5-Omni tokenizer.json path in HARTSY_QWEN_OMNI_TOKENIZER.</summary>
    [Fact]
    [Trait("Category", "Integration")]
    public void BuildIds_WithTheRealTokenizer_KeepsNewlinesAndSpecials()
    {
        string? path = Environment.GetEnvironmentVariable("HARTSY_QWEN_OMNI_TOKENIZER");
        if (path is null || !File.Exists(path))
        {
            return;
        }
        using FileStream json = File.OpenRead(path);
        GgufTokenizer tok = HfTokenizerJson.LoadByteLevelBpe(json);
        int[] ids = AukPrompt.BuildIds("Raise the pitch by 2 semitones.", 4, tok.EncodeOrdinary);
        int[] viaSpecialAware = tok.Encode(AukPrompt.BuildText("Raise the pitch by 2 semitones.", 4), addSpecial: true);
        Assert.Equal(viaSpecialAware, ids);
    }
}
