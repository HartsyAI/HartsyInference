using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Planning;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;

namespace HartsyInference.Engine.Tests.Services;

public sealed class MeshGridLimitTests
{
    private static MeshRequest Request(int grid) => new()
    {
        Image = new ImageData { Rgb = new byte[4 * 4 * 3], Width = 4, Height = 4 },
        GridResolution = grid,
    };

    [Fact]
    public void ATurboGridWhoseFinestLevelCannotFitOneArrayIsRefusedBeforeAnythingLoads()
    {
        ArgumentOutOfRangeException error = Assert.Throws<ArgumentOutOfRangeException>(() => MeshRequestValidator.Validate("hunyuan3d-turbo", Request(1536), matterAvailable: false));

        Assert.Equal(nameof(MeshRequest.GridResolution), MeshFieldTag.FieldOf(error));
        Assert.Contains("largest this decode takes", error.Message);
    }

    [Fact]
    public void TheGridsTheTurboModelRunsAreAccepted()
    {
        MeshRequestValidator.Validate("hunyuan3d-turbo", Request(0), matterAvailable: false);
        MeshRequestValidator.Validate("hunyuan3d-turbo", Request(512), matterAvailable: false);
        MeshRequestValidator.Validate("hunyuan3d-turbo", Request(1024), matterAvailable: false);
    }

    [Fact]
    public void ADenseModelsLimitIsItsOwnCube()
    {
        MeshRequestValidator.Validate("hunyuan3d", Request(1289), matterAvailable: false);
        Assert.Throws<ArgumentOutOfRangeException>(() => MeshRequestValidator.Validate("hunyuan3d", Request(1291), matterAvailable: false));
    }

    [Fact]
    public async Task APlanReportsTheRefusalInsteadOfPlanningARunThatWouldOverflow()
    {
        using InferenceEngine engine = new("cpu");
        ModelSpec spec = new() { Requested = "hunyuan3d-turbo", Modality = Modality.Mesh, Catalog = ModelCatalog.Find("hunyuan3d-turbo"), LocalPath = Path.GetTempPath() };

        MeshPlan plan = await engine.MeshPlanning.PlanAsync(spec, Request(1536));

        Assert.False(plan.IsValid);
        Assert.Contains(plan.Issues, i => i.Field == nameof(MeshRequest.GridResolution));
    }
}
