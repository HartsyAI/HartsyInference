using HartsyInference.Core.Configuration;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.ThreeD.Models.Hunyuan3D.Paint;

namespace HartsyInference.Engine.Tests.Services;

/// <summary>The Paint 2.1 view upscaler is on by default, as upstream; it loads only when its checkpoint is in the model folder, and the knob turns it off.</summary>
public sealed class MeshServicePaintUpscalerTests : IDisposable
{
    private readonly string _folder = Directory.CreateTempSubdirectory("paint-upscaler").FullName;

    public void Dispose() => Directory.Delete(_folder, recursive: true);

    [Fact]
    public void TheKnobIsOnByDefaultAndFixedAtConstruction()
    {
        Assert.True(EngineKnobs.PaintUpscaleViews.Default);
        Assert.Equal(KnobScope.Construction, EngineKnobs.PaintUpscaleViews.Scope);
        Assert.Equal("numerics.paintUpscaleViews", EngineKnobs.PaintUpscaleViews.Id);
    }

    [Fact]
    public void TheReferenceProjectionKnobIsOnByDefaultAndReadPerRun()
    {
        Assert.True(EngineKnobs.PaintProjectReference.Default);
        Assert.Equal(KnobScope.Runtime, EngineKnobs.PaintProjectReference.Scope);
        Assert.Equal("numerics.paintProjectReference", EngineKnobs.PaintProjectReference.Id);
    }

    [Theory]
    [InlineData(0, true, 4096)]
    [InlineData(0, false, 2048)]
    [InlineData(1024, true, 1024)]
    public void PaintTextureSize_IsTheRequestsGridElse4096WhenThePictureIsBakedIn(int grid, bool bakesReference, int expected)
    {
        Assert.Equal(expected, MeshService.PaintTextureSize(grid, bakesReference));
    }

    [Fact]
    public void SidePictures_AreReportedUnbakedWhenTheReferenceProjectionIsOff()
    {
        MeshRequest withViews = new() { Views = [new MeshViewImage { View = MeshView.Left, Image = Picture(alpha: 0) }] };
        Assert.Contains(EngineKnobs.PaintProjectReference.Id, MeshService.UnbakedViewsWarning(withViews, projectReference: false));
        Assert.Null(MeshService.UnbakedViewsWarning(withViews, projectReference: true));
        Assert.Null(MeshService.UnbakedViewsWarning(new MeshRequest(), projectReference: false));
    }

    [Fact]
    public void PaintSides_BakesCutoutsAndWarnsForPicturesWithoutOne()
    {
        List<string> warnings = [];
        List<MeshViewImage> views =
        [
            new() { View = MeshView.Left, Image = Picture(alpha: 0) },
            new() { View = MeshView.Back, Image = Picture(alpha: 255) },
            new() { View = MeshView.Right, Image = Picture(alpha: null) },
        ];

        List<PaintSideReference> sides = MeshService.PaintSides(views, warnings);

        Assert.Equal(90, Assert.Single(sides).Azimuth);
        Assert.Equal(2, warnings.Count);
        Assert.Contains("Back picture has an alpha plane that is opaque everywhere", warnings[0]);
        Assert.Contains("Right picture has no alpha plane", warnings[1]);
    }

    // A 2 x 2 picture whose first pixel has the given alpha and the rest are opaque; null alpha means no alpha plane.
    private static ImageData Picture(byte? alpha) => new()
    {
        Rgb = new byte[2 * 2 * 3],
        Width = 2,
        Height = 2,
        Alpha = alpha is { } first ? [first, 255, 255, 255] : null,
    };

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
