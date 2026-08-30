using System.Net;
using System.Net.Http.Json;
using Microsoft.AspNetCore.Mvc.Testing;
using Xunit;

namespace HartsyInference.API.Tests;

/// <summary>In-process tests for <c>/v1/native/video/stream</c> and <c>/v1/native/world/sessions*</c>. Runs on the
/// CPU backend with no model ever loaded, so a session can never actually be opened here — these cover the
/// routing/validation/404 surface only, same scope as every other Phase 1-4 test file. The registry's own
/// mechanics (register/get/close/idle-eviction) are covered directly in
/// <see cref="WorldSessionRegistryTests"/>.</summary>
public sealed class VideoWorldEndpointsTests : IClassFixture<WebApplicationFactory<Program>>
{
    private readonly WebApplicationFactory<Program> _factory;

    public VideoWorldEndpointsTests(WebApplicationFactory<Program> factory)
    {
        _factory = factory.WithWebHostBuilder(builder => builder.UseSetting("HartsyInference:Backend", "cpu"));
    }

    [Fact]
    public async Task VideoStream_UnresolvableModel_ReturnsTyped422BeforeStreaming()
    {
        // Preflight runs before SSE commits its response. A model that cannot resolve therefore remains an
        // ordinary typed HTTP error instead of producing an error event after a misleading 200 response.
        using HttpClient client = _factory.CreateClient();
        using HttpRequestMessage req = new HttpRequestMessage(HttpMethod.Post, "/v1/native/video/stream")
        {
            Content = JsonContent.Create(new
            {
                model = "not-a-real-video-model",
                request = new { prompt = "a dog running" },
            }),
        };

        using HttpResponseMessage resp = await client.SendAsync(req, HttpCompletionOption.ResponseHeadersRead);
        Assert.Equal(HttpStatusCode.UnprocessableEntity, resp.StatusCode);
        Assert.DoesNotContain("text/event-stream", resp.Content.Headers.ContentType?.MediaType ?? "");
        string body = await resp.Content.ReadAsStringAsync();
        Assert.Contains("video.plan.model_unresolvable", body, StringComparison.Ordinal);
        Assert.Contains("model", body, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task WorldOpen_UnresolvableModel_Returns400()
    {
        // WorldService.Open requires InitImage before it even looks at the model (else ArgumentException/400,
        // a different case) — supplying one here to reach the FileNotFoundException/400 path being tested.
        using HttpClient client = _factory.CreateClient();
        HttpResponseMessage resp = await client.PostAsJsonAsync("/v1/native/world/sessions", new
        {
            model = "not-a-real-world-model",
            request = new { initImage = new { rgb = Convert.ToBase64String([0, 0, 0]), width = 1, height = 1 } },
        });
        Assert.Equal(HttpStatusCode.BadRequest, resp.StatusCode);
    }

    [Fact]
    public async Task WorldOpen_MissingInitImage_Returns400()
    {
        // WorldService validates InitImage with ArgumentException, which GenerationErrors maps to a caller error.
        using HttpClient client = _factory.CreateClient();
        HttpResponseMessage resp = await client.PostAsJsonAsync("/v1/native/world/sessions", new
        {
            model = "whatever",
            request = new { prompt = "a small room" },
        });
        Assert.Equal(HttpStatusCode.BadRequest, resp.StatusCode);
    }

    [Fact]
    public async Task WorldAction_UnknownSession_Returns404()
    {
        using HttpClient client = _factory.CreateClient();
        HttpResponseMessage resp = await client.PostAsJsonAsync("/v1/native/world/sessions/not-a-real-session/action", new { action = "forward" });
        Assert.Equal(HttpStatusCode.NotFound, resp.StatusCode);
    }

    [Fact]
    public async Task WorldStream_UnknownSession_Returns404()
    {
        using HttpClient client = _factory.CreateClient();
        HttpResponseMessage resp = await client.GetAsync("/v1/native/world/sessions/not-a-real-session/stream");
        Assert.Equal(HttpStatusCode.NotFound, resp.StatusCode);
    }

    [Fact]
    public async Task WorldClose_UnknownSession_Returns404()
    {
        using HttpClient client = _factory.CreateClient();
        HttpResponseMessage resp = await client.DeleteAsync("/v1/native/world/sessions/not-a-real-session");
        Assert.Equal(HttpStatusCode.NotFound, resp.StatusCode);
    }

    [Fact]
    public async Task AdminQueue_ReportsBothFastAndLongRunning()
    {
        using HttpClient client = _factory.CreateClient();
        System.Text.Json.JsonElement body = await client.GetFromJsonAsync<System.Text.Json.JsonElement>("/admin/queue");
        Assert.Equal(0, body.GetProperty("fast").GetProperty("pending").GetInt32());
        Assert.Equal(1, body.GetProperty("fast").GetProperty("maxConcurrency").GetInt32());
        Assert.Equal(16, body.GetProperty("fast").GetProperty("maxQueueDepth").GetInt32());
        Assert.Equal(0, body.GetProperty("longRunning").GetProperty("pending").GetInt32());
        Assert.Equal(1, body.GetProperty("longRunning").GetProperty("maxConcurrency").GetInt32());
        Assert.Equal(4, body.GetProperty("longRunning").GetProperty("maxQueueDepth").GetInt32());
    }
}
