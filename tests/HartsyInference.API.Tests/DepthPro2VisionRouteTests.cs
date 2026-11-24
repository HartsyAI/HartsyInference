using System.Net;
using System.Net.Http.Json;
using System.Text.Json;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Hosting.Server;
using Microsoft.AspNetCore.Hosting.Server.Features;
using Microsoft.Extensions.DependencyInjection;

namespace HartsyInference.API.Tests;

/// <summary>Metric depth over HTTP: the focal-length and view-count refusals before any model loads, and (with <c>HARTSY_DEPTHPRO2_WEIGHTS</c> set to the S
/// checkpoint, the focal estimator in a <c>DepthPro2-focal</c> folder beside its folder) one image and two views answered by Depth Pro 2 on CPU.</summary>
public sealed class DepthPro2VisionRouteTests : IAsyncLifetime
{
    private const string WeightsVariable = "HARTSY_DEPTHPRO2_WEIGHTS";
    private const int Width = 64, Height = 48;

    private WebApplication? _app;
    private HttpClient _http = null!;   // set in InitializeAsync

    public async Task InitializeAsync()
    {
        WebApplicationBuilder builder = WebApplication.CreateBuilder();
        builder.WebHost.UseUrls("http://127.0.0.1:0");
        builder.Services.AddHartsyInference();
        _app = builder.Build();
        _app.MapHartsyInferenceEndpoints();
        await _app.StartAsync();
        string address = _app.Services.GetRequiredService<IServer>().Features.Get<IServerAddressesFeature>()!.Addresses.First();
        _http = new HttpClient { BaseAddress = new Uri(address), Timeout = TimeSpan.FromMinutes(5) };
    }

    public async Task DisposeAsync()
    {
        _http?.Dispose();
        if (_app is not null) await _app.DisposeAsync();
    }

    private static object Image(int shift = 0)
    {
        byte[] rgb = new byte[Width * Height * 3];
        for (int i = 0; i < rgb.Length; i++) rgb[i] = (byte)((i * 7 + i / (Width * 3) * 5 + shift) % 251);
        return new { rgb = Convert.ToBase64String(rgb), width = Width, height = Height };
    }

    private static string? ModelPath => Environment.GetEnvironmentVariable(WeightsVariable);

    [Fact]
    public async Task AFocalLengthOnAnotherModeIsA400()
    {
        HttpResponseMessage response = await _http.PostAsJsonAsync("/v1/native/vision",
            new { model = "depthpro2-s", request = new { image = Image(), mode = "Depth", focalLengthPx = 500.0 } });

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        Assert.Contains("only applies to MetricDepth", await response.Content.ReadAsStringAsync());
    }

    [Fact]
    public async Task ANegativeFocalLengthOrASingleViewIsA400()
    {
        HttpResponseMessage negative = await _http.PostAsJsonAsync("/v1/native/vision",
            new { model = "depthpro2-s", request = new { image = Image(), mode = "MetricDepth", focalLengthPx = -1.0 } });
        HttpResponseMessage oneView = await _http.PostAsJsonAsync("/v1/native/vision/sequence",
            new { model = "depthpro2-s", request = new { images = new[] { Image() }, mode = "MetricDepth", kind = "Views" } });

        Assert.Equal(HttpStatusCode.BadRequest, negative.StatusCode);
        Assert.Equal(HttpStatusCode.BadRequest, oneView.StatusCode);
    }

    [Fact]
    public async Task MetricDepthOnAModelThatIsNotDepthPro2IsA501()
    {
        HttpResponseMessage response = await _http.PostAsJsonAsync("/v1/native/vision",
            new { model = "depth-anything", request = new { image = Image(), mode = "MetricDepth" } });

        Assert.Equal(HttpStatusCode.NotImplemented, response.StatusCode);
    }

    [RealWeightsFact(WeightsVariable)]
    public async Task OneImageReturnsMetricDepthAndTheGivenFocal()
    {
        HttpResponseMessage response = await _http.PostAsJsonAsync("/v1/native/vision",
            new { model = "depthpro2-s", modelPath = ModelPath, request = new { image = Image(), mode = "MetricDepth", focalLengthPx = 640.0 } });

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        using JsonDocument body = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
        JsonElement result = body.RootElement;
        Assert.Equal(640, result.GetProperty("camera").GetProperty("fx").GetDouble());
        Assert.Equal(Width / 2.0, result.GetProperty("camera").GetProperty("cx").GetDouble());
        AssertDepth(result.GetProperty("depthMap"));
    }

    [RealWeightsFact(WeightsVariable)]
    public async Task TwoViewsShareOneEstimatedFocal()
    {
        HttpResponseMessage response = await _http.PostAsJsonAsync("/v1/native/vision/sequence", new
        {
            model = "depthpro2-s", modelPath = ModelPath,
            request = new { images = new[] { Image(), Image(shift: 40) }, mode = "MetricDepth", kind = "Views" },
        });

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        using JsonDocument body = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
        JsonElement[] views = [.. body.RootElement.GetProperty("views").EnumerateArray()];
        Assert.Equal(2, views.Length);
        double focal = views[0].GetProperty("camera").GetProperty("fx").GetDouble();
        Assert.True(focal > 0);
        Assert.Equal(focal, views[1].GetProperty("camera").GetProperty("fx").GetDouble());
        foreach (JsonElement view in views) AssertDepth(view.GetProperty("depthMap"));
    }

    private static void AssertDepth(JsonElement depthMap)
    {
        Assert.Equal("MetricMeters", depthMap.GetProperty("kind").GetString());
        JsonElement values = depthMap.GetProperty("values");
        Assert.Equal("float32", values.GetProperty("dtype").GetString());
        Assert.Equal([Height, Width], values.GetProperty("shape").EnumerateArray().Select(e => e.GetInt32()));
        byte[] data = Convert.FromBase64String(values.GetProperty("data").GetString()!);
        Assert.Equal(Width * Height * 4, data.Length);
        for (int i = 0; i < data.Length; i += 4) Assert.True(BitConverter.ToSingle(data, i) > 0);
    }
}
