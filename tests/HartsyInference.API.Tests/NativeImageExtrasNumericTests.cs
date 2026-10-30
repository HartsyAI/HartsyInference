using System.Text.Json;
using System.Text.Json.Serialization;
using HartsyInference.Engine.Features;
using HartsyInference.Engine.Requests;
using Xunit;

namespace HartsyInference.API.Tests;

/// <summary>A native image request body binds to <see cref="ImageRequest"/> with the app's JSON options, and its
/// <see cref="ImageRequest.Extra"/> numbers must arrive as numbers. The reader drops a value it cannot parse to its
/// default without an error, so this asserts the parsed value, not just that the request binds.</summary>
public sealed class NativeImageExtrasNumericTests
{
    private static readonly JsonSerializerOptions s_options = new(JsonSerializerDefaults.Web)
    {
        Converters = { new JsonStringEnumConverter() },
    };

    [Fact]
    public void ExtraNumbers_PostedAsJson_ArriveAsTheirValues()
    {
        const string body = """{"prompt":"a cat","extra":{"reference.max_pixels":786432,"reference.ref_scale":3.25}}""";

        ImageRequest request = JsonSerializer.Deserialize<ImageRequest>(body, s_options)!;

        Assert.Equal(786432, RequestExtras.Number(request.Extra, RequestExtras.ReferenceMaxPixels, -1));
        Assert.Equal(3.25, RequestExtras.Number(request.Extra, RequestExtras.ReferenceScale, -1));
    }
}
