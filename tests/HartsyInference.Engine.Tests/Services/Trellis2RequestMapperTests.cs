using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.ThreeD.Pipelines.Requests;

namespace HartsyInference.Engine.Tests.Services;

public sealed class Trellis2RequestMapperTests
{
    [Fact]
    public void ToOptions_MapsAShapeOnlyRequestAndLeavesTextureOff()
    {
        Trellis2Options options = Trellis2RequestMapper.ToOptions(new MeshRequest { GridResolution = 512, Steps = 8, Seed = 5 });

        Assert.Equal(512, options.Grid);
        Assert.Equal(8, options.Steps);
        Assert.Equal(5, options.Seed);
        Assert.False(options.Texture);
        Assert.Equal(0, options.TextureSize);
    }

    [Fact]
    public void ToOptions_CarriesTheTextureRequest()
    {
        Trellis2Options options = Trellis2RequestMapper.ToOptions(new MeshRequest { Texture = true, TextureSize = 1024 });

        Assert.True(options.Texture);
        Assert.Equal(1024, options.TextureSize);
        Assert.Null(options.Seed);
    }

    [Fact]
    public void ToOptions_RejectsASeedThatDoesNotFitTheSampler()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => Trellis2RequestMapper.ToOptions(new MeshRequest { Seed = (long)int.MaxValue + 1 }));
    }

    [Fact]
    public void RejectForOtherModels_AllowsADefaultRequest()
    {
        Trellis2RequestMapper.RejectForOtherModels(new MeshRequest(), "triposr");
    }

    [Theory]
    [InlineData(true, 0)]
    [InlineData(false, 2048)]
    public void RejectForOtherModels_RefusesTextureOptions(bool texture, int size)
    {
        MeshRequest request = new() { Texture = texture, TextureSize = size };

        Assert.Throws<NotSupportedException>(() => Trellis2RequestMapper.RejectForOtherModels(request, "triposr"));
    }
}
