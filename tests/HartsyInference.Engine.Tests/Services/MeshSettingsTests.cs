using HartsyInference.Engine.Planning;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;

namespace HartsyInference.Engine.Tests.Services;

public sealed class MeshSettingsTests
{
    [Fact]
    public void ARandomSeedIsDrawnNowAndReportedAsDerived()
    {
        MeshSettingsUsed first = MeshService.SettingsOf("trellis2", new MeshRequest { Seed = -1 });
        MeshSettingsUsed second = MeshService.SettingsOf("trellis2", new MeshRequest { Seed = -1 });

        Assert.InRange(first.Seed!.Value, 0, int.MaxValue - 1);
        Assert.Equal(first.Seed.Value.ToString(), first.Parameters[nameof(MeshRequest.Seed)].Value);
        Assert.Equal(ParameterSource.Derived, first.Parameters[nameof(MeshRequest.Seed)].Source);
        Assert.NotEqual(first.Seed, second.Seed);
    }

    [Fact]
    public void AnExplicitSeedIsReportedAsGiven()
    {
        MeshSettingsUsed settings = MeshService.SettingsOf("pixal3d", new MeshRequest { Seed = 42, GridResolution = 1536 });

        Assert.Equal(42, settings.Seed);
        Assert.Equal(new ResolvedParameter("42", ParameterSource.UserOverride), settings.Parameters[nameof(MeshRequest.Seed)]);
        Assert.Equal(new ResolvedParameter("1536", ParameterSource.UserOverride), settings.Parameters[nameof(MeshRequest.GridResolution)]);
        Assert.Equal("pixal3d", settings.ModelId);
        Assert.False(string.IsNullOrEmpty(settings.EngineVersion));
    }

    [Theory]
    [InlineData("triposr")]
    [InlineData("cola3d")]
    public void AModelWithNoSeedReportsNone(string id)
    {
        MeshSettingsUsed settings = MeshService.SettingsOf(id, new MeshRequest { Seed = -1 });

        Assert.Null(settings.Seed);
        Assert.False(settings.Parameters.ContainsKey(nameof(MeshRequest.Seed)));
    }

    [Fact]
    public void AModelThatWrapsALargeSeedReportsTheWrappedValueTheSamplerGets()
    {
        MeshSettingsUsed beyond = MeshService.SettingsOf("hunyuan3d", new MeshRequest { Seed = (1L << 32) + 5 });
        MeshSettingsUsed negative = MeshService.SettingsOf("hunyuan3d", new MeshRequest { Seed = 1L << 31 });

        Assert.Equal(5, beyond.Seed);
        Assert.Equal(int.MinValue, negative.Seed);
    }

    [Fact]
    public void AModelThatChecksTheSeedKeepsTheRequestedValueForTheValidatorToReject()
    {
        MeshSettingsUsed settings = MeshService.SettingsOf("trellis2", new MeshRequest { Seed = 1L << 31 });

        Assert.Equal(1L << 31, settings.Seed);
    }

    [Fact]
    public void AnUnknownModelHasNoParametersAndKeepsTheEngineVersion()
    {
        MeshSettingsUsed settings = MeshService.SettingsOf("hunyuan3d-mv", new MeshRequest());

        Assert.Empty(settings.Parameters);
        Assert.Null(settings.Seed);
    }
}
