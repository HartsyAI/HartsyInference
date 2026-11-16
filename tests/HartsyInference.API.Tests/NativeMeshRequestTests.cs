using HartsyInference.API.Endpoints;
using HartsyInference.Engine;
using HartsyInference.Engine.Requests;

namespace HartsyInference.API.Tests;

public sealed class NativeMeshRequestTests
{
    private static string Png(byte value) => Convert.ToBase64String(PngEncoder.Encode(Enumerable.Repeat(value, 2 * 2 * 3).ToArray(), [255, 255, 0, 255], 2, 2));

    [Fact]
    public void ViewPngsBecomeTheRequestsViewsWithTheirAlpha()
    {
        NativeMeshRequest native = new()
        {
            Model = "hunyuan3d-mv",
            Request = new MeshRequest(),
            ImagePng = Png(10),
            ViewPngs = new Dictionary<string, string> { ["Left"] = Png(20), ["back"] = Png(30) },
        };

        MeshRequest request = native.ToMeshRequest();

        Assert.Equal(10, request.Image!.Rgb[0]);
        Assert.Equal([MeshView.Left, MeshView.Back], request.Views!.Select(v => v.View));
        Assert.Equal(30, request.Views![1].Image.Rgb[0]);
        Assert.Equal([255, 255, 0, 255], request.Views![0].Image.Alpha);
    }

    [Theory]
    [InlineData("front")]
    [InlineData("2")]
    [InlineData("top")]
    public void AnUnknownSideIsRejected(string side)
    {
        NativeMeshRequest native = new() { Model = "hunyuan3d-mv", Request = new MeshRequest(), ViewPngs = new Dictionary<string, string> { [side] = Png(1) } };

        Assert.Throws<ArgumentException>(native.ToMeshRequest);
    }

    [Fact]
    public void ViewsSentBothWaysAreRejected()
    {
        MeshViewImage view = new() { View = MeshView.Back, Image = new Engine.Requests.ImageData { Rgb = new byte[3], Width = 1, Height = 1 } };
        NativeMeshRequest native = new()
        {
            Model = "hunyuan3d-mv",
            Request = new MeshRequest { Views = [view] },
            ViewPngs = new Dictionary<string, string> { ["left"] = Png(1) },
        };

        Assert.Throws<ArgumentException>(native.ToMeshRequest);
    }
}
