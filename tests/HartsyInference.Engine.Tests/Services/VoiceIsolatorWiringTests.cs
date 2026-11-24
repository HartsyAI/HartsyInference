using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Requests;
using Xunit;

namespace HartsyInference.Engine.Tests.Services;

public sealed class VoiceIsolatorWiringTests
{
    private static readonly AudioClip Clip = new() { Data = [1, 2, 3], Format = "wav" };

    [Theory]
    [InlineData("voice-isolator")]
    [InlineData("voice-isolator-realtime")]
    public async Task DemucsOnlyOptionsAreRefused(string model)
    {
        using InferenceEngine engine = new("cpu");
        ModelSpec spec = new() { Requested = model, Modality = Modality.Fx };

        await Assert.ThrowsAsync<ArgumentException>(() => engine.Fx.SeparateAsync(spec, new FxSeparateRequest { Audio = Clip, Shifts = 2 }));
        await Assert.ThrowsAsync<ArgumentException>(() => engine.Fx.SeparateAsync(spec, new FxSeparateRequest { Audio = Clip, Segment = 7.8 }));
        await Assert.ThrowsAsync<ArgumentException>(() => engine.Fx.SeparateAsync(spec, new FxSeparateRequest { Audio = Clip, Overlap = 0.1 }));
    }

    [Fact]
    public async Task LookaheadBelongsToTheRealtimeModelOnly()
    {
        using InferenceEngine engine = new("cpu");
        ModelSpec spec = new() { Requested = "voice-isolator", Modality = Modality.Fx };

        await Assert.ThrowsAsync<ArgumentException>(() => engine.Fx.SeparateAsync(spec, new FxSeparateRequest { Audio = Clip, LookaheadFrames = 2 }));
    }

    [Theory]
    [InlineData("voice-isolator")]
    [InlineData("voice-isolator-realtime")]
    public void BothModelsAreInTheCatalog(string id)
    {
        CatalogEntry? entry = ModelCatalog.All.FirstOrDefault(e => e.Id == id);
        Assert.NotNull(entry);
        Assert.Equal(Modality.Fx, entry.Modality);
        Assert.True(entry.CliDrivable);
    }
}
