using System.Net;
using System.Net.Http.Json;
using HartsyInference.Tests;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Hosting.Server;
using Microsoft.AspNetCore.Hosting.Server.Features;
using Microsoft.Extensions.DependencyInjection;

namespace HartsyInference.API.Tests;

/// <summary>Open Vision Banana over HTTP, without weights: a mode the model does not answer is a 501 that names the model, and a
/// segmentation with no class names is a 400, both before any load. The 4B and 9B catalog entries take the same route.</summary>
public sealed class OvbVisionRouteTests : IAsyncLifetime
{
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

    private static object Image()
    {
        byte[] rgb = new byte[Width * Height * 3];
        for (int i = 0; i < rgb.Length; i++) rgb[i] = (byte)(i % 251);
        return new { rgb = Convert.ToBase64String(rgb), width = Width, height = Height };
    }

    [Theory]
    [InlineData("open-vision-banana-4b")]
    [InlineData("open-vision-banana-9b")]
    public async Task ADetectRequestOnOpenVisionBananaIsA501NamingTheModel(string model)
    {
        HttpResponseMessage response = await _http.PostAsJsonAsync("/v1/native/vision",
            new { model, request = new { image = Image(), mode = "Detect", prompt = "car" } });

        Assert.Equal(HttpStatusCode.NotImplemented, response.StatusCode);
        Assert.Contains(model, await response.Content.ReadAsStringAsync());
    }

    [Theory]
    [InlineData("open-vision-banana-4b")]
    [InlineData("open-vision-banana-9b")]
    public async Task ASegmentRequestWithNoClassNamesIsA400(string model)
    {
        HttpResponseMessage response = await _http.PostAsJsonAsync("/v1/native/vision",
            new { model, request = new { image = Image(), mode = "Segment", prompt = " , " } });

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        Assert.Contains("comma-separated", await response.Content.ReadAsStringAsync());
    }
}
