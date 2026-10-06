using System.Text.Json;
using HartsyInference.ModelAssets.Tokenizers;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Real-weight checks for Breeze TTS 2; they need <c>HARTSY_BREEZE_DIR</c> to name the downloaded
/// <c>BreezeBlue/Breeze-TTS-2</c> folder (plus <c>ref_ids.json</c> from HF <c>tokenizers</c>) and otherwise do nothing.</summary>
[Trait("Category", "RealWeights")]
public sealed class BreezeTts2RealWeightTests
{
    private static string? Dir => Environment.GetEnvironmentVariable("HARTSY_BREEZE_DIR");

    [Fact]
    public void Tokenizer_MatchesHuggingFaceTokenizers()
    {
        if (Dir is not { Length: > 0 } dir) return;
        using FileStream json = File.OpenRead(Path.Combine(dir, "tokenizer.json"));
        SentencePieceBpeJson tokenizer = new(json);
        using JsonDocument reference = JsonDocument.Parse(File.ReadAllText(Path.Combine(dir, "ref_ids.json")));
        foreach (JsonElement item in reference.RootElement.EnumerateArray())
        {
            int[] expected = item.GetProperty("ids").EnumerateArray().Select(e => e.GetInt32()).ToArray();
            Assert.Equal(expected, tokenizer.Encode(item.GetProperty("text").GetString()!));
        }
    }
}
