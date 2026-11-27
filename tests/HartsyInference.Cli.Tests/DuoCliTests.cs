using HartsyInference.Cli.Commands;
using HartsyInference.Cli.Dispatch;
using HartsyInference.Cli.Infra;
using HartsyInference.Engine;
using HartsyInference.Engine.Recipes.Video;
using Xunit;

namespace HartsyInference.Cli.Tests;

/// <summary>What <c>hartsy video</c> puts on a request for a Duo run, and that other families' requests are unchanged.</summary>
public sealed class DuoCliTests
{
    private static ParamState Parameters(params (string Key, string Value)[] values)
    {
        ParamState parameters = new(Modality.Video);
        parameters.Put("negative", "");
        foreach ((string key, string value) in values)
        {
            parameters.Put(key, value);
        }
        return parameters;
    }

    [Fact]
    public void BothRunOptionsBecomeExtraKeysWithAFullFolderPathAndNoDefaultNegative()
    {
        ParamState parameters = Parameters(("duo-run-folder", "runs/a"), ("duo-window-frames", "34"));

        IReadOnlyDictionary<string, object> extra = GenerationDispatch.BuildVideoExtra(parameters);

        Assert.Equal(Path.GetFullPath("runs/a"), extra[DuoRecipePipeline.RunFolderKey]);
        Assert.Equal(34, extra[DuoRecipePipeline.WindowFramesKey]);
        Assert.Equal(2, extra.Count);
        Assert.Null(GenerationDispatch.VideoNegativePrompt(parameters));
    }

    [Fact]
    public void ARunWithoutDuoOptionsKeepsItsDefaultNegativeAndNoExtras()
    {
        Assert.Empty(GenerationDispatch.BuildVideoExtra(Parameters()));
        Assert.Equal("blurry, low quality, distorted, watermark", GenerationDispatch.VideoNegativePrompt(Parameters()));
    }

    [Fact]
    public void ATypedNegativeIsPassedOnSoTheRecipeCanRefuseIt()
    {
        ParamState parameters = Parameters(("duo-run-folder", "runs/a"));
        parameters.Put("negative", "grain");

        Assert.Equal("grain", GenerationDispatch.VideoNegativePrompt(parameters));
    }

    [Theory]
    [InlineData("duo-v1-causal", true)]
    [InlineData("duo-v1-2-causal", true)]
    [InlineData("duo-v1-distill", false)]
    [InlineData("minimax-h3", false)]
    [InlineData("", false)]
    public void OnlyTheCausalDuoModelsTakeTheRunOptions(string model, bool expected) =>
        Assert.Equal(expected, DuoCliGate.IsDuo(model));

    [Fact]
    public void ADuoRunWithoutAFolderAndAnotherFamilyWithDuoOptionsAreRefusedUpFront()
    {
        Assert.Contains("--duo-run-folder", DuoCliGate.Refusal("duo-v1-causal", null, null), StringComparison.Ordinal);
        Assert.Contains("--duo-run-folder", DuoCliGate.Refusal("duo-v1-causal", null, 34), StringComparison.Ordinal);
        Assert.Null(DuoCliGate.Refusal("duo-v1-causal", "run", null));
        Assert.Null(DuoCliGate.Refusal("minimax-h3", null, null));
        Assert.NotNull(DuoCliGate.Refusal("minimax-h3", "run", null));
        Assert.NotNull(DuoCliGate.Refusal("", null, 34));
    }
}
