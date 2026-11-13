using System.Net;
using System.Net.Http.Json;
using System.Text.Json;
using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Io;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Hosting.Server;
using Microsoft.AspNetCore.Hosting.Server.Features;
using Microsoft.Extensions.DependencyInjection;

namespace HartsyInference.API.Tests;

/// <summary>The thumbnail route over HTTP: a PNG for a glb, the options, and the 400s.</summary>
public sealed class MeshThumbnailRouteTests : IAsyncLifetime
{
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
        _http = new HttpClient { BaseAddress = new Uri(address), Timeout = TimeSpan.FromSeconds(30) };
    }

    public async Task DisposeAsync()
    {
        _http?.Dispose();
        if (_app is not null) await _app.DisposeAsync();
    }

    private static string Glb() => Convert.ToBase64String(GlbWriter.Write(new Mesh { Vertices = [0, 0, 0, 1, 0, 0, 0, 1, 0], Indices = [0, 1, 2] }));

    [Fact]
    public async Task ThumbnailReturnsAPngOfTheRequestedSize()
    {
        HttpResponseMessage response = await _http.PostAsJsonAsync("/v1/native/mesh/thumbnail",
            new { data = Glb(), format = "glb", options = new { width = 80, height = 64, view = "front", background = "#ffffff" } });

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        using JsonDocument body = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
        byte[] png = Convert.FromBase64String(body.RootElement.GetProperty("png").GetString()!);
        Assert.Equal([0x89, (byte)'P', (byte)'N', (byte)'G'], png[..4]);
        Assert.Equal((80, 64, "mesh"), (body.RootElement.GetProperty("width").GetInt32(), body.RootElement.GetProperty("height").GetInt32(), body.RootElement.GetProperty("kind").GetString()));
    }

    [Theory]
    [InlineData("", "glb", null)]
    [InlineData("not base64!!", "glb", null)]
    [InlineData("AAAA", "fbx", null)]
    [InlineData("AAAA", "glb", null)]
    public async Task ABadRequestIsA4xxWithAMessage(string data, string format, object? options)
    {
        HttpResponseMessage response = await _http.PostAsJsonAsync("/v1/native/mesh/thumbnail", new { data, format, options });

        Assert.InRange((int)response.StatusCode, 400, 499);
        Assert.False(string.IsNullOrWhiteSpace(await response.Content.ReadAsStringAsync()));
    }

    [Fact]
    public async Task AnUnknownViewIsA400NamingTheValidViews()
    {
        HttpResponseMessage response = await _http.PostAsJsonAsync("/v1/native/mesh/thumbnail", new { data = Glb(), format = "glb", options = new { view = "sideways" } });

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        Assert.Contains("front, back", await response.Content.ReadAsStringAsync());
    }
}
