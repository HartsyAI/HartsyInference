using HartsyInference.Core.Configuration;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.ThreeD.Pipelines;

namespace HartsyInference.Engine.Tests.Services;

public sealed class MeshServiceXPartOptionsTests
{
    [Fact]
    public void FastDecodeIsAnOffByDefaultSettingReadPerRequest()
    {
        Assert.False(EngineKnobs.XPartFastDecode.Default);
        Assert.Equal(KnobScope.Runtime, EngineKnobs.XPartFastDecode.Scope);
        Assert.Equal("numerics.xpartFastDecode", EngineKnobs.XPartFastDecode.Id);
    }

    [Fact]
    public void XPartOptionsFor_TakesTheDefaultsForUnsetFieldsAndPassesFastDecodeThrough()
    {
        XPartOptions unset = MeshService.XPartOptionsFor(new MeshRequest(), fastDecode: false);
        Assert.Equal(42, unset.Seed);
        Assert.Equal(50, unset.Steps);
        Assert.Equal(256, unset.GridResolution);
        Assert.False(unset.FastDecode);

        XPartOptions set = MeshService.XPartOptionsFor(new MeshRequest { Seed = 7, Steps = 12, GridResolution = 128 }, fastDecode: true);
        Assert.Equal(7, set.Seed);
        Assert.Equal(12, set.Steps);
        Assert.Equal(128, set.GridResolution);
        Assert.True(set.FastDecode);
    }
}
