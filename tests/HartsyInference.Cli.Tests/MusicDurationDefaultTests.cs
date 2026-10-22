using HartsyInference.Cli.Infra;
using HartsyInference.Engine;
using Xunit;

namespace HartsyInference.Cli.Tests;

/// <summary>An unset music duration stays null in the request, so the engine's recipe duration applies (120 s for
/// ACE-Step 1.5 and MiniMax Music 3, 300 s for YuE2) instead of a CLI-side 10 s.</summary>
public sealed class MusicDurationDefaultTests
{
    [Fact]
    public void MusicParams_LeaveDurationUnset_UntilTheUserSetsIt()
    {
        ParamState parameters = new ParamState(Modality.Music);
        Assert.Null(parameters.GetIntOrNull("duration"));

        Assert.True(parameters.TrySet("duration", "30"));
        Assert.Equal(30, parameters.GetIntOrNull("duration"));
    }
}
