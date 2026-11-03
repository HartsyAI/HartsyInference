using System.Text.Json;
using System.Text.Json.Serialization;
using HartsyInference.API.Endpoints;
using HartsyInference.Engine;
using HartsyInference.Engine.Requests;
using Xunit;

namespace HartsyInference.API.Tests;

/// <summary>The native image response for a layered result: <c>png</c> stays the composite and <c>layers</c> lists
/// <c>{ index, name, png }</c> bottom first, each an RGBA PNG of that layer. A single image carries no layers.</summary>
public sealed class LayeredImageResponseTests
{
    private static readonly JsonSerializerOptions Web = new(JsonSerializerDefaults.Web) { Converters = { new JsonStringEnumConverter() } };

    [Fact]
    public void Layers_TravelBottomFirstWithTheirAlpha()
    {
        ImageLayer bottom = new ImageLayer { Index = 0, Width = 1, Height = 1, Rgb = [10, 20, 30], Alpha = [255], Name = "background" };
        ImageLayer top = new ImageLayer { Index = 1, Width = 1, Height = 1, Rgb = [200, 100, 50], Alpha = [77] };
        ImageResult result = new ImageResult { Rgb = [9, 9, 9], Alpha = [255], Width = 1, Height = 1, Layers = [bottom, top] };

        JsonElement json = JsonSerializer.SerializeToElement(ImageEndpoints.ToResponse(result), Web);

        Assert.Equal(PngEncoder.Encode(result.Rgb, result.Alpha, 1, 1), Convert.FromBase64String(json.GetProperty("png").GetString()!));
        JsonElement layers = json.GetProperty("layers");
        Assert.Equal(2, layers.GetArrayLength());
        Assert.Equal(0, layers[0].GetProperty("index").GetInt32());
        Assert.Equal("background", layers[0].GetProperty("name").GetString());
        Assert.Equal(PngEncoder.Encode(bottom.Rgb, bottom.Alpha, 1, 1), Convert.FromBase64String(layers[0].GetProperty("png").GetString()!));
        Assert.Equal(1, layers[1].GetProperty("index").GetInt32());
        Assert.Equal(PngEncoder.Encode(top.Rgb, top.Alpha, 1, 1), Convert.FromBase64String(layers[1].GetProperty("png").GetString()!));
    }

    [Fact]
    public void SingleImage_HasNoLayers()
    {
        JsonElement json = JsonSerializer.SerializeToElement(
            ImageEndpoints.ToResponse(new ImageResult { Rgb = [0, 0, 0], Width = 1, Height = 1 }), Web);

        Assert.Equal(JsonValueKind.Null, json.GetProperty("layers").ValueKind);
    }
}
