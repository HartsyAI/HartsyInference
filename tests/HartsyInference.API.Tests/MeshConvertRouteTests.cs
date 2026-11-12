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

/// <summary>The mesh conversion routes over HTTP: the format lists, a conversion with its files and archive, and the 400s for a bad request.</summary>
public sealed class MeshConvertRouteTests : IAsyncLifetime
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
    public async Task FormatsListsTheSourcesAndTargets()
    {
        using JsonDocument formats = JsonDocument.Parse(await _http.GetStringAsync("/v1/native/mesh/formats"));

        Assert.Equal(["obj", "ply", "stl", "glb"], formats.RootElement.GetProperty("targets").EnumerateArray().Select(e => e.GetString()));
        Assert.Equal(["glb", "obj"], formats.RootElement.GetProperty("sources").EnumerateArray().Select(e => e.GetString()));
    }

    [Fact]
    public async Task ConvertReturnsTheFilesAndAnArchiveOnRequest()
    {
        HttpResponseMessage response = await _http.PostAsJsonAsync("/v1/native/mesh/convert", new { data = Glb(), format = "glb", target = "obj", name = "bust", archive = true });

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        using JsonDocument body = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
        Assert.Equal("bust.obj", body.RootElement.GetProperty("files")[0].GetProperty("name").GetString());
        Assert.Equal(1, body.RootElement.GetProperty("triangles").GetInt32());
        Assert.False(body.RootElement.GetProperty("staticOnly").GetBoolean());
        Assert.NotEqual(JsonValueKind.Null, body.RootElement.GetProperty("archive").ValueKind);
    }

    [Theory]
    [InlineData("not base64!!", "glb", "obj")]
    [InlineData("", "glb", "obj")]
    [InlineData("AAAA", "glb", "fbx")]
    [InlineData("AAAA", "glb", "obj")]
    public async Task ABadRequestIsA4xxWithAMessage(string data, string format, string target)
    {
        HttpResponseMessage response = await _http.PostAsJsonAsync("/v1/native/mesh/convert", new { data, format, target });

        Assert.InRange((int)response.StatusCode, 400, 499);
        Assert.False(string.IsNullOrWhiteSpace(await response.Content.ReadAsStringAsync()));
    }
}
