using HartsyInference.Core.Configuration;
using HartsyInference.Engine.Services;

namespace HartsyInference.Engine.Tests.Services;

/// <summary>The Paint 2.1 view upscaler is opt-in: off by default it is never loaded, even when its checkpoint is in the model folder.</summary>
public sealed class MeshServicePaintUpscalerTests : IDisposable
{
    private readonly string _folder = Directory.CreateTempSubdirectory("paint-upscaler").FullName;

    public void Dispose() => Directory.Delete(_folder, recursive: true);

    [Fact]
    public void TheKnobIsOffByDefaultAndFixedAtConstruction()
    {
        Assert.False(EngineKnobs.PaintUpscaleViews.Default);
        Assert.Equal(KnobScope.Construction, EngineKnobs.PaintUpscaleViews.Scope);
        Assert.Equal("numerics.paintUpscaleViews", EngineKnobs.PaintUpscaleViews.Id);
    }

    [Fact]
    public void PaintUpscaler_NeedsBothTheKnobAndTheCheckpoint()
    {
        Assert.Null(MeshService.PaintUpscaler(_folder, enabled: true));
        string checkpoint = Path.Combine(_folder, "RealESRGAN_x4plus.pth");
        File.WriteAllBytes(checkpoint, [0]);

        Assert.Null(MeshService.PaintUpscaler(_folder, enabled: false));
        Assert.Equal(checkpoint, MeshService.PaintUpscaler(_folder, enabled: true));
    }
}
