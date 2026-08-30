using HartsyInference.Core.Exceptions;
using HartsyInference.ModelAssets.MiniMaxH3;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

public sealed class MiniMaxH3PddScheduleTests
{
    [Theory]
    [InlineData(4, 8, 8, 8, 8)]
    [InlineData(6, 8, 8, 4, 4)]
    [InlineData(8, 4, 4, 4, 4)]
    public void Create_UsesOnlyPublishedPartitions(int nfe, int first, int second, int third, int fourth)
    {
        MiniMaxH3PddSchedule schedule = MiniMaxH3PddSchedule.Create(Settings(nfe));

        Assert.Equal(nfe, schedule.Nfe);
        Assert.Equal(first, schedule.Steps[0].FineCount);
        Assert.Equal(second, schedule.Steps[1].FineCount);
        Assert.Equal(third, schedule.Steps[2].FineCount);
        Assert.Equal(fourth, schedule.Steps[3].FineCount);
        Assert.Equal(1.0, schedule.Sigmas[0]);
        Assert.Equal(0.0, schedule.Sigmas[^1]);
        foreach (MiniMaxH3PddStep step in schedule.Steps)
        {
            Assert.Equal(1.0f, step.VideoWeights.Sum(), 6);
            Assert.Equal(1.0f, step.AudioWeights.Sum(), 6);
        }
    }

    [Fact]
    public void Create_RejectsEveryUncertifiedRecipeDimension()
    {
        Assert.Throws<HartsyInferenceException>(() => MiniMaxH3PddSchedule.Create(Settings(5)));
        Assert.Throws<HartsyInferenceException>(() => MiniMaxH3PddSchedule.Create(Settings(8) with { Sampler = "dpmpp_2m" }));
        Assert.Throws<HartsyInferenceException>(() => MiniMaxH3PddSchedule.Create(Settings(8) with { CfgScale = 1.1f }));
        Assert.Throws<HartsyInferenceException>(() => MiniMaxH3PddSchedule.Create(Settings(8) with { VideoFlowShift = 11.0 }));
        Assert.Throws<HartsyInferenceException>(() => MiniMaxH3PddSchedule.Create(Settings(8) with { Strength = 0.5f }));
        Assert.Throws<HartsyInferenceException>(() => MiniMaxH3PddSchedule.Create(Settings(8) with { HasVsa = true }));
    }

    [Fact]
    public void Resolve_RejectsOffGridOrNonAdjacentEvaluations()
    {
        MiniMaxH3PddSchedule schedule = MiniMaxH3PddSchedule.Create(Settings(8));
        MiniMaxH3PddStep expected = schedule.Steps[2];

        Assert.Equal(expected, schedule.Resolve(expected.Sigma, expected.SigmaNext));
        Assert.Throws<HartsyInferenceException>(() => schedule.Resolve(expected.Sigma - 0.01, expected.SigmaNext));
        Assert.Throws<HartsyInferenceException>(() => schedule.Resolve(expected.Sigma, schedule.Steps[3].SigmaNext));
    }

    private static MiniMaxH3PddExecutionSettings Settings(int nfe) => new()
    {
        Nfe = nfe,
        Sampler = "euler",
        CfgScale = 1.0f,
        VideoFlowShift = 12.0,
        AudioFlowShift = 3.0,
        Strength = 1.0f,
    };
}
