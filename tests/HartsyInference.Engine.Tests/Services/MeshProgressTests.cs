using HartsyInference.Diffusion.Requests;
using HartsyInference.Engine.Services;

namespace HartsyInference.Engine.Tests.Services;

public sealed class MeshProgressTests
{
    private static GenerationProgress Tick(int step, int total, string stage, int stageStep, int stageSteps)
        => new(step, total, 0) { Stage = stage, StageStep = stageStep, StageSteps = stageSteps };

    /// <summary>The ticks of a shape-only Pixal3D run with 3 structure and 2 + 2 shape steps, as the pipeline reports them.</summary>
    private static List<GenerationProgress> PixalRun()
    {
        List<GenerationProgress> run = [];
        int step = 0;
        void Add(string stage, int steps)
        {
            for (int i = 1; i <= steps; i++)
                run.Add(Tick(++step, 13, stage, i, steps));
        }

        Add("Structure conditioning", 1);
        Add("Structure flow", 3);
        Add("Coarse conditioning", 1);
        Add("Coarse shape flow", 2);
        Add("Upsample", 1);
        Add("Fine conditioning", 1);
        Add("Fine shape flow", 2);
        Add("Shape decode", 1);
        Add("Mesh extraction", 1);
        return run;
    }

    [Fact]
    public void TheOverallFractionNeverMovesBackwardsAndEndsAtOne()
    {
        float[] fractions = [.. PixalRun().Select(p => MeshProgress.Map("pixal3d", false, 1024, p).OverallFraction!.Value)];

        Assert.Equal(fractions.Order(), fractions);
        Assert.Equal(1f, fractions[^1], 3);
        Assert.True(fractions[0] > 0f);
    }

    [Fact]
    public void TheFractionFollowsTheWeightsNotTheTickCount()
    {
        List<GenerationProgress> run = PixalRun();
        GenerationProgress beforeFine = run.Single(p => p.Stage == "Fine conditioning");
        GenerationProgress afterFine = run.Last(p => p.Stage == "Fine shape flow");

        float early = MeshProgress.Map("pixal3d", false, 1024, beforeFine).OverallFraction!.Value;
        float late = MeshProgress.Map("pixal3d", false, 1024, afterFine).OverallFraction!.Value;

        // Seven of thirteen ticks have passed at the fine conditioning, but the fine flow is most of the time, so the bar is still under half.
        Assert.True(beforeFine.Step >= 7);
        Assert.InRange(early, 0f, 0.3f);
        Assert.InRange(late, 0.8f, 1f);
    }

    [Fact]
    public void AStageCarriesItsIndexAndTheStageCount()
    {
        List<GenerationProgress> run = PixalRun();

        StepPreview structure = MeshProgress.Map("pixal3d", false, 1024, run.First(p => p.Stage == "Structure flow"));
        StepPreview textured = MeshProgress.Map("pixal3d", true, 1024, run.First(p => p.Stage == "Structure flow"));

        Assert.Equal("Structure flow", structure.Stage);
        Assert.Equal(2, structure.StageIndex);
        Assert.Equal(9, structure.StageCount);
        Assert.Equal(13, textured.StageCount);
        Assert.Equal(2, structure.Step);
        Assert.Equal(13, structure.TotalSteps);
    }

    [Fact]
    public void TheTRELLIS2StagesDependOnTheGrid()
    {
        GenerationProgress flow = Tick(5, 20, "Shape flow 512", 2, 4);

        StepPreview coarse = MeshProgress.Map("trellis2", false, 512, flow);
        StepPreview cascade = MeshProgress.Map("trellis2", false, 1024, flow);

        Assert.Equal(5, coarse.StageCount);
        Assert.Equal(7, cascade.StageCount);
        Assert.True(coarse.OverallFraction > cascade.OverallFraction);
    }

    [Fact]
    public void AnUnnamedOrUnknownStageFallsBackToTheStepCount()
    {
        StepPreview unnamed = MeshProgress.Map("trellis2", false, 1024, new GenerationProgress(5, 20, 0));
        StepPreview unknown = MeshProgress.Map("trellis2", false, 1024, Tick(5, 20, "Nonsense", 1, 1));

        Assert.Equal(0.25f, unnamed.OverallFraction);
        Assert.Equal(0.25f, unknown.OverallFraction);
        Assert.Equal(0, unnamed.StageIndex);
    }

    [Fact]
    public void PlainReportsTheStepCountAndAStageWhenGiven()
    {
        StepPreview preview = MeshProgress.Plain(3, 12, "Card sampler", 1, 5, 3, 4);

        Assert.Equal((3, 12, "Card sampler", 3, 4), (preview.Step, preview.TotalSteps, preview.Stage, preview.StageIndex, preview.StageCount));
        Assert.Equal(0.25f, preview.OverallFraction);
        Assert.Null(MeshProgress.Plain(0, 0).OverallFraction);
    }
}
