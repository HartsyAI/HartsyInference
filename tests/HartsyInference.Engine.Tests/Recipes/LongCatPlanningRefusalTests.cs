using HartsyInference.Engine.Planning;
using HartsyInference.Engine.Recipes.Video;
using HartsyInference.Engine.Requests;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Engine.Tests;

/// <summary>Planning refusals of the LongCat recipes: what can only fail after the weights load is refused before any weight is read.</summary>
[Trait("Category", "SyntheticSmoke")]
public sealed class LongCatPlanningRefusalTests
{
    private static readonly AudioClip Speech = new() { Data = [] };

    private static VideoEffectiveSettings Settings(int width = 832, int height = 480, int frames = 93, int fps = 25) => new()
    {
        Width = width, Height = height, Frames = frames, Fps = fps, Steps = 8, CfgScale = 1f, Seed = 3,
        ReferenceSizing = VideoReferenceSizing.Native, LockedFields = default,
    };

    [Fact]
    public void TheAvatarRefusesARequestWithoutDrivingSpeech()
    {
        string? refusal = new LongCatAvatarRecipe().RequestRefusal(new VideoRequest { Prompt = "p" }, Settings());
        Assert.NotNull(refusal);
        Assert.Contains("driving speech", refusal, StringComparison.Ordinal);
    }

    [Fact]
    public void TheAvatarRefusesAChainOverTheBudgetAndAcceptsAValidOne()
    {
        LongCatAvatarRecipe avatar = new();
        string? over = avatar.RequestRefusal(new VideoRequest { Prompt = "p", VideoAudioReference = Speech, ChainTotalFrames = 300 }, Settings());
        Assert.NotNull(over);
        Assert.Contains("budget", over, StringComparison.Ordinal);
        Assert.Null(avatar.RequestRefusal(new VideoRequest { Prompt = "p", VideoAudioReference = Speech, ChainTotalFrames = 150 }, Settings()));
        Assert.Null(avatar.RequestRefusal(new VideoRequest { Prompt = "p", VideoAudioReference = Speech }, Settings()));
    }

    [Fact]
    public void TheAvatarRefusesAWrongFrameRate()
    {
        string? refusal = new LongCatAvatarRecipe().RequestRefusal(new VideoRequest { Prompt = "p", VideoAudioReference = Speech },
            Settings(fps: 15));
        Assert.NotNull(refusal);
        Assert.Contains("fps", refusal, StringComparison.Ordinal);
    }

    [Fact]
    public void TheBaseRefusesDrivingSpeechAndUnwiredLongVideoAndAcceptsAPlainRequest()
    {
        LongCatVideoRecipe baseRecipe = new();
        Assert.NotNull(baseRecipe.RequestRefusal(new VideoRequest { Prompt = "p", VideoAudioReference = Speech }, Settings()));
        Assert.NotNull(baseRecipe.RequestRefusal(new VideoRequest { Prompt = "p", ChainTotalFrames = 300 }, Settings()));
        Assert.Null(baseRecipe.RequestRefusal(new VideoRequest { Prompt = "p" }, Settings()));
    }

    [Fact]
    public void TheDistilledBaseRefusesWhatTheBaseRefuses()
    {
        LongCatVideoDistilledRecipe distilled = new();
        Assert.NotNull(distilled.RequestRefusal(new VideoRequest { Prompt = "p", VideoAudioReference = Speech }, Settings()));
        Assert.NotNull(distilled.RequestRefusal(new VideoRequest { Prompt = "p", ChainTotalFrames = 300 }, Settings()));
        Assert.Null(distilled.RequestRefusal(new VideoRequest { Prompt = "p" }, Settings()));
    }

    [Fact]
    public void TheBaseAcceptsAContinuationAndTheDistilledBuildAndAnInitImageAreRefused()
    {
        VideoRequest continuation = new() { Prompt = "p", VideoContinuation = new VideoClip { Data = [1] } };
        Assert.Null(new LongCatVideoRecipe().RequestRefusal(continuation, Settings(fps: 15)));
        Assert.NotNull(new LongCatVideoRecipe().RequestRefusal(continuation, Settings(fps: 25)));
        Assert.NotNull(new LongCatVideoDistilledRecipe().RequestRefusal(continuation, Settings()));
        Assert.NotNull(new LongCatVideoRecipe().RequestRefusal(continuation with { InitImage = new ImageData { Rgb = [1, 2, 3], Width = 1, Height = 1 } },
            Settings()));
    }
}
