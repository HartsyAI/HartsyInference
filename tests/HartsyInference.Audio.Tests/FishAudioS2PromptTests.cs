using System.Text.Json;
using HartsyInference.Audio.Models.FishAudio;
using Xunit;

namespace HartsyInference.Audio.Tests;

public sealed class FishAudioS2PromptTests
{
    [Fact]
    public void SplitBatches_MatchesFishSpeech()
    {
        // Expected values come from running fish-speech's split_text_by_speaker + group_turns_into_batches.
        using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(
            Path.Combine(AppContext.BaseDirectory, "Fixtures", "FishAudioS2", "s2_batches.json")));
        foreach (JsonElement item in doc.RootElement.EnumerateArray())
        {
            string[] expected = item.GetProperty("batches").EnumerateArray().Select(e => e.GetString()!).ToArray();
            Assert.Equal(expected, FishAudioS2Prompt.SplitBatches(item.GetProperty("text").GetString()!));
        }
    }

    [Fact]
    public void Prompt_TokenizesEachPartSeparately_AndCarriesCodesOnSemanticPositions()
    {
        // One id per UTF-16 unit so part boundaries are visible in the token list.
        FishAudioS2Prompt prompt = new(text => text.Select(c => (int)c).ToArray(), semanticBeginId: 1000);
        int[,] reference = { { 3, 4 }, { 7, 8 } };
        prompt.System("hi", reference).User("x").OpenAssistant();

        string text = "<|im_start|>system\nconvert the provided text to speech reference to the following:\n\nText:\n"
            + "<|speaker:0|>hi\n\nSpeech:\n";
        int[] tokens = prompt.Tokens;
        Assert.Equal(text.Select(c => (int)c), tokens.Take(text.Length));
        Assert.Equal([1003, 1004], tokens.Skip(text.Length).Take(2));
        Assert.Equal([3, 7], prompt.Codes[text.Length]!);
        Assert.Equal([4, 8], prompt.Codes[text.Length + 1]!);
        Assert.Null(prompt.Codes[0]);

        string tail = "<|im_end|>\n<|im_start|>user\nx<|im_end|>\n<|im_start|>assistant\n<|voice|>";
        Assert.Equal(tail.Select(c => (int)c), tokens.Skip(text.Length + 2));
    }

}
