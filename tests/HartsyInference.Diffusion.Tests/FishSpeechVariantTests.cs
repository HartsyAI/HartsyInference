using HartsyInference.Engine.Audio;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Fish-Speech's variant picks the checkpoint. The bare id and the 1.5 names reach fish-speech-1.5, the 1.4 names reach the
/// 1.4 layout, an org/repo keeps its own repo, and any other name is refused rather than loaded as 1.5.</summary>
public sealed class FishSpeechVariantTests
{
    [Theory]
    [InlineData("", "fishaudio/fish-speech-1.5", false)]
    [InlineData("fishspeech", "fishaudio/fish-speech-1.5", false)]
    [InlineData("fish-speech-1.5", "fishaudio/fish-speech-1.5", false)]
    [InlineData("1.5", "fishaudio/fish-speech-1.5", false)]
    [InlineData("FISH-SPEECH-1.4", "fishaudio/fish-speech-1.4", true)]
    [InlineData("1.4", "fishaudio/fish-speech-1.4", true)]
    [InlineData("fishaudio/fish-speech-1.4", "fishaudio/fish-speech-1.4", true)]
    [InlineData("someone/finetune", "someone/finetune", false)]
    public void Parse_PicksRepoAndLayout(string variant, string repo, bool layout14)
    {
        FishSpeechModel.Variant parsed = FishSpeechModel.Variant.Parse(variant);
        Assert.Equal(repo, parsed.Repo);
        Assert.Equal(layout14, parsed.Layout14);
    }

    [Theory]
    [InlineData("fish-speech-1.2")]
    [InlineData("fish-speech-1")]
    [InlineData("s1-mini")]
    public void Parse_RefusesCheckpointsTheEngineDoesNotRun(string variant) =>
        Assert.Throws<ArgumentException>(() => FishSpeechModel.Variant.Parse(variant));

    [Fact]
    public void Files_LeaveTheWeightsLast()
    {
        IReadOnlyList<HartsyInference.Audio.Cache.AudioModelFile> v15 = FishSpeechModel.Variant.Parse("fish-speech-1.5").Files();
        Assert.Equal("model.pth", v15[^1].Name);
        Assert.Contains(v15, f => f.Name == "tokenizer.tiktoken");

        IReadOnlyList<HartsyInference.Audio.Cache.AudioModelFile> v14 = FishSpeechModel.Variant.Parse("fish-speech-1.4").Files();
        Assert.Equal("model.pth", v14[^1].Name);
        Assert.Contains(v14, f => f.Name == "tokenizer.json");
    }
}
