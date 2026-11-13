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
        Assert.Contains("largest that fits one", error.Message);
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
        MeshRequestValidator.Validate("hunyuan3d", Request(1290), matterAvailable: false);
        Assert.Throws<ArgumentOutOfRangeException>(() => MeshRequestValidator.Validate("hunyuan3d", Request(1291), matterAvailable: false));
    }

    [Theory]
    [InlineData("hunyuan3d-turbo", 2_097_152)]
    [InlineData("hunyuan3d-turbo", int.MaxValue)]
    [InlineData("hunyuan3d", 2_097_152)]
    [InlineData("hunyuan3d", int.MaxValue)]
    public void AGridSoLargeItsCubeWouldWrapIsRefusedToo(string id, int grid)
    {
        ArgumentOutOfRangeException error = Assert.Throws<ArgumentOutOfRangeException>(() => MeshRequestValidator.Validate(id, Request(grid), matterAvailable: false));

        Assert.Equal(nameof(MeshRequest.GridResolution), MeshFieldTag.FieldOf(error));
    }

    [Fact]
    public void TripoSrsDensityCubeHasTheSameLimitAndIsRefusedBeforeAnythingLoads()
    {
        MeshRequestValidator.Validate("triposr", Request(0), matterAvailable: false);
        MeshRequestValidator.Validate("triposr", Request(256), matterAvailable: false);
        MeshRequestValidator.Validate("triposr", Request(1290), matterAvailable: false);
        foreach (int grid in new[] { 1291, 2_097_152, int.MaxValue })
        {
            ArgumentOutOfRangeException error = Assert.Throws<ArgumentOutOfRangeException>(() => MeshRequestValidator.Validate("triposr", Request(grid), matterAvailable: false));

            Assert.Equal(nameof(MeshRequest.GridResolution), MeshFieldTag.FieldOf(error));
            Assert.Contains("largest that fits one is 1290", error.Message);
        }
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    public void ATurboGridWithNoFlashVdmLevelIsRefusedUpFront(int grid)
    {
        ArgumentOutOfRangeException error = Assert.Throws<ArgumentOutOfRangeException>(() => MeshRequestValidator.Validate("hunyuan3d-turbo", Request(grid), matterAvailable: false));

        Assert.Equal(nameof(MeshRequest.GridResolution), MeshFieldTag.FieldOf(error));
        MeshRequestValidator.Validate("hunyuan3d-turbo", Request(3), matterAvailable: false);
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
