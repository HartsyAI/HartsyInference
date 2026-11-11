using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.ThreeD.Pipelines.Requests;

namespace HartsyInference.Engine.Tests.Services;

public sealed class Pixal3DRequestMapperTests
{
    [Fact]
    public void ToOptions_MapsGridStepsSeedAndFieldOfView()
    {
        Pixal3DOptions options = Pixal3DRequestMapper.ToOptions(new MeshRequest { GridResolution = 1536, Steps = 8, Seed = 42, FieldOfViewRadians = 0.5 });

        Assert.Equal(1536, options.Grid);
        Assert.Equal(8, options.Steps);
        Assert.Equal(42, options.Seed);
        Assert.Equal(0.5, options.FovRadians);
    }

    [Fact]
    public void ToOptions_LeavesTheDefaultsToThePipeline()
    {
        Pixal3DOptions options = Pixal3DRequestMapper.ToOptions(new MeshRequest());

        Assert.Equal(0, options.Grid);
        Assert.Null(options.Seed);
        Assert.Null(options.FovRadians);
    }

    [Fact]
    public void ToOptions_RejectsASeedBeyond32Bits()
        => Assert.Throws<ArgumentOutOfRangeException>(() => Pixal3DRequestMapper.ToOptions(new MeshRequest { Seed = (long)int.MaxValue + 1 }));

    [Theory]
    [InlineData("hairdeck")]
    [InlineData("cola3d")]
    [InlineData("unirig")]
    [InlineData("trellis2")]
    [InlineData("hunyuan3d")]
    public void RejectForOtherModels_FailsForEveryModelButPixal3D(string id)
        => Assert.Throws<NotSupportedException>(() => Pixal3DRequestMapper.RejectForOtherModels(new MeshRequest { FieldOfViewRadians = 0.5 }, id));

    [Fact]
    public void RejectForOtherModels_AllowsPixal3DAndRequestsWithoutAFieldOfView()
    {
        Pixal3DRequestMapper.RejectForOtherModels(new MeshRequest { FieldOfViewRadians = 0.5 }, MeshModelIds.Pixal3D);
        Pixal3DRequestMapper.RejectForOtherModels(new MeshRequest(), "trellis2");
    }
}
