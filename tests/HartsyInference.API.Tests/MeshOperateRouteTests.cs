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

/// <summary>The mesh operation routes over HTTP: the operation listing, an edit with its step reports, and the 400s.</summary>
public sealed class MeshOperateRouteTests : IAsyncLifetime
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

    private static string Glb() => Convert.ToBase64String(GlbWriter.Write(new Mesh { Vertices = [10, 10, 10, 12, 10, 10, 10, 12, 10], Indices = [0, 1, 2] }));

    [Fact]
    public async Task OperationsListsEveryOperationWithItsParameters()
    {
        using JsonDocument all = JsonDocument.Parse(await _http.GetStringAsync("/v1/native/mesh/operations"));

        string[] ids = [.. all.RootElement.EnumerateArray().Select(e => e.GetProperty("id").GetString()!)];
        Assert.Equal(["remove-small-components", "fill-holes", "decimate", "smooth", "recompute-normals", "recenter", "rescale"], ids);
        JsonElement smooth = all.RootElement.EnumerateArray().Single(e => e.GetProperty("id").GetString() == "smooth");
        JsonElement iterations = smooth.GetProperty("parameters").EnumerateArray().Single(p => p.GetProperty("name").GetString() == "iterations");
        Assert.Equal((5, 1, 100, true), (iterations.GetProperty("default").GetInt32(), iterations.GetProperty("min").GetInt32(), iterations.GetProperty("max").GetInt32(), iterations.GetProperty("integer").GetBoolean()));
    }

    [Fact]
    public async Task OperateReturnsTheEditedFileAndAReportPerStep()
    {
        HttpResponseMessage response = await _http.PostAsJsonAsync("/v1/native/mesh/operate", new
        {
            data = Glb(),
            format = "glb",
            operations = new object[] { new { operation = "recenter" }, new { operation = "rescale", parameters = new { longestSide = 1.0 } } },
            name = "tri",
        });

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        using JsonDocument body = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
        Assert.Equal("glb", body.RootElement.GetProperty("target").GetString());
        Assert.Equal("tri.glb", body.RootElement.GetProperty("files")[0].GetProperty("name").GetString());
        byte[] glb = Convert.FromBase64String(body.RootElement.GetProperty("files")[0].GetProperty("data").GetString()!);
        Mesh mesh = GlbReader.Parse(glb);
        Assert.Equal(1.0, Math.Max(mesh.Vertices[3] - mesh.Vertices[0], mesh.Vertices[7] - mesh.Vertices[1]), 3);
        JsonElement steps = body.RootElement.GetProperty("steps");
        Assert.Equal(["recenter", "rescale"], steps.EnumerateArray().Select(s => s.GetProperty("operation").GetString()));
        Assert.Equal(1, steps[0].GetProperty("trianglesBefore").GetInt32());
    }

    [Theory]
    [InlineData("explode", null)]
    [InlineData("smooth", 0.0)]
    public async Task ABadOperationOrNumberIsA400WithAMessage(string operation, double? iterations)
    {
        HttpResponseMessage response = await _http.PostAsJsonAsync("/v1/native/mesh/operate", new
        {
            data = Glb(),
            format = "glb",
            operations = new object[] { new { operation, parameters = iterations is null ? null : new { iterations } } },
        });

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        Assert.False(string.IsNullOrWhiteSpace(await response.Content.ReadAsStringAsync()));
    }

    [Theory]
    [InlineData("")]
    [InlineData("AAAA")]
    public async Task AMissingOperationListOrAnUnreadableFileIsA400(string data)
    {
        HttpResponseMessage noOperations = await _http.PostAsJsonAsync("/v1/native/mesh/operate", new { data = Glb(), format = "glb", operations = Array.Empty<object>() });
        HttpResponseMessage unreadable = await _http.PostAsJsonAsync("/v1/native/mesh/operate", new { data, format = "glb", operations = new object[] { new { operation = "recenter" } } });

        Assert.Equal(HttpStatusCode.BadRequest, noOperations.StatusCode);
        Assert.Equal(HttpStatusCode.BadRequest, unreadable.StatusCode);
    }
}
