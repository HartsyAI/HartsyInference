using System.Net;
using System.Net.Http.Json;
using Microsoft.AspNetCore.Mvc.Testing;
using Xunit;

namespace HartsyInference.API.Tests;

/// <summary>In-process tests for <c>/v1/native/{vision,mesh}</c>. Runs on the CPU backend with no model ever
/// loaded. Each "unresolvable model" expectation below was confirmed by reading the actual service source first —
/// this repo's services are NOT consistent about which exception type signals "no checkpoint", so guessing a
/// single status code across modalities would be wrong (Vision's Embed path hits an unmapped
/// InvalidOperationException → 500; Mesh hits FileNotFoundException → 400, same as images).</summary>
public sealed class VisionMeshEndpointsTests : IClassFixture<WebApplicationFactory<Program>>
{
    private readonly WebApplicationFactory<Program> _factory;

    public VisionMeshEndpointsTests(WebApplicationFactory<Program> factory)
    {
        _factory = factory.WithWebHostBuilder(builder => builder.UseSetting("HartsyInference:Backend", "cpu"));
    }

    [Fact]
    public async Task Vision_Embed_UnresolvableModel_Returns500()
    {
        // VisionService.Embed throws InvalidOperationException for a null LocalPath, which GenerationErrors
        // doesn't special-case (unlike FileNotFoundException/HartsyInferenceException) — documenting the real
        // current behavior, not an aspirational 400.
        using HttpClient client = _factory.CreateClient();
        HttpResponseMessage resp = await client.PostAsJsonAsync("/v1/native/vision", new
        {
            model = "not-a-real-vision-model",
            request = new { image = new { rgb = Convert.ToBase64String([0, 0, 0]), width = 1, height = 1 }, mode = "Embed" },
        });
        Assert.Equal(HttpStatusCode.InternalServerError, resp.StatusCode);
    }

    [Fact]
    public async Task Vision_InvalidMode_Returns400FromJsonBinding()
    {
        // VisionMode is a closed enum bound via JsonStringEnumConverter — an unrecognized string fails body
        // binding (BadHttpRequestException → 400) before the handler ever runs, not VisionService's own
        // "unknown mode" NotSupportedException fallback (which is dead code from the HTTP surface's perspective).
        using HttpClient client = _factory.CreateClient();
        HttpResponseMessage resp = await client.PostAsJsonAsync("/v1/native/vision", new
        {
            model = "whatever",
            request = new { image = new { rgb = Convert.ToBase64String([0, 0, 0]), width = 1, height = 1 }, mode = "NotARealMode" },
        });
        Assert.Equal(HttpStatusCode.BadRequest, resp.StatusCode);
    }

    [Fact]
    public async Task Mesh_UnresolvableModel_Returns400()
    {
        using HttpClient client = _factory.CreateClient();
        HttpResponseMessage resp = await client.PostAsJsonAsync("/v1/native/mesh", new
        {
            model = "not-a-real-3d-model",
            request = new { image = new { rgb = Convert.ToBase64String([0, 0, 0]), width = 1, height = 1 } },
        });
        Assert.Equal(HttpStatusCode.BadRequest, resp.StatusCode);
    }

    [Fact]
    public async Task Mesh_TextTo3D_Returns501NotWiredYet()
    {
        // MeshRequest.Image is null (pure text-to-3D) — MeshService rejects this by name before even looking at
        // the model, per its own class doc: "text-to-3D has no wired pipeline yet".
        using HttpClient client = _factory.CreateClient();
        HttpResponseMessage resp = await client.PostAsJsonAsync("/v1/native/mesh", new
        {
            model = "whatever",
            request = new { prompt = "a small dragon" },
        });
        Assert.Equal(HttpStatusCode.NotImplemented, resp.StatusCode);
    }

    [Fact]
    public async Task MeshStream_UnresolvableModel_ReportsErrorEventNotHang()
    {
        using HttpClient client = _factory.CreateClient();
        using HttpRequestMessage req = new HttpRequestMessage(HttpMethod.Post, "/v1/native/mesh/stream")
        {
            Content = JsonContent.Create(new
            {
                model = "not-a-real-3d-model",
                request = new { image = new { rgb = Convert.ToBase64String([0, 0, 0]), width = 1, height = 1 } },
            }),
        };

        using HttpResponseMessage resp = await client.SendAsync(req, HttpCompletionOption.ResponseHeadersRead);
        Assert.Equal(HttpStatusCode.OK, resp.StatusCode);
        string body = await resp.Content.ReadAsStringAsync();
        Assert.Contains("event: error", body);
    }
}
