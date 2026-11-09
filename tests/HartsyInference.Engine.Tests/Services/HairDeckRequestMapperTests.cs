using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.ThreeD.Models.HairDeck;

namespace HartsyInference.Engine.Tests.Services;

public sealed class HairDeckRequestMapperTests
{
    [Fact]
    public void ToOptions_MapsADefaultRequestToTheReferenceSettings()
    {
        MeshRequest request = new() { Seed = 7 };

        HairDeckOptions options = HairDeckRequestMapper.ToOptions(request);

        Assert.Equal(7, options.Seed);
        Assert.Equal(HairDeckRequestMapper.DefaultSteps, options.Steps);
        Assert.Equal(HairDeckRequestMapper.DefaultCfg, options.Cfg);
        Assert.True(options.Colors);
        Assert.True(options.Chains);
    }

    [Fact]
    public void ToOptions_RejectsMoreThanOneSample()
    {
        Assert.Throws<NotSupportedException>(() => HairDeckRequestMapper.ToOptions(new MeshRequest { Samples = 2 }));
    }

    [Fact]
    public void ToOptions_RejectsRiggedOutputWithoutChains()
    {
        Assert.Throws<NotSupportedException>(() => HairDeckRequestMapper.ToOptions(new MeshRequest { Rigged = true, Chains = false }));
    }

    [Fact]
    public void ToOptions_PassesRiggedOutputThrough()
    {
        HairDeckOptions options = HairDeckRequestMapper.ToOptions(new MeshRequest { Rigged = true, Seed = 0 });
        Assert.True(options.Rigged);
    }

    [Fact]
    public void ToOptions_HonoursAnExplicitColourAndChainChoice()
    {
        HairDeckOptions options = HairDeckRequestMapper.ToOptions(new MeshRequest { Seed = 0, Colors = false, Chains = false });

        Assert.False(options.Colors);
        Assert.False(options.Chains);
    }

    [Fact]
    public void RejectForOtherModels_AllowsADefaultRequest()
    {
        HairDeckRequestMapper.RejectForOtherModels(new MeshRequest(), "triposr");
    }

    [Theory]
    [InlineData("samples")]
    [InlineData("rigged")]
    [InlineData("colors")]
    [InlineData("chains")]
    public void RejectForOtherModels_RefusesAHairDeckOnlyOption(string option)
    {
        MeshRequest request = option switch
        {
            "samples" => new MeshRequest { Samples = 3 },
            "rigged" => new MeshRequest { Rigged = true },
            "colors" => new MeshRequest { Colors = true },
            _ => new MeshRequest { Chains = false },
        };

        Assert.Throws<NotSupportedException>(() => HairDeckRequestMapper.RejectForOtherModels(request, "triposr"));
    }
}
