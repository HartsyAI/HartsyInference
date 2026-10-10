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
    public async Task WorldAction_UnknownSession_Returns404()
    {
        using HttpClient client = _factory.CreateClient();
        HttpResponseMessage resp = await client.PostAsJsonAsync("/v1/native/world/sessions/not-a-real-session/action", new { action = "forward" });
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
