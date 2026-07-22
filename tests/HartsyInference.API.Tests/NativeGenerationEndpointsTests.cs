using System.Net;
using System.Net.Http.Json;
using System.Text.Json;
using Microsoft.AspNetCore.Mvc.Testing;
using Xunit;

namespace HartsyInference.API.Tests;

/// <summary>In-process tests for the native <c>/v1/native/*</c> generation routes' request handling — model
/// resolution, error mapping, and the queue-gated SSE path. Runs on the CPU backend with no model ever loaded, so
/// every case here hits <c>FileNotFoundException</c> (no checkpoint resolved) before any real inference would
/// start; it validates routing/validation/error-mapping only. Real end-to-end generation is verified separately
/// through Swarm, not a standalone server process, per this repo's GPU-sharing discipline.</summary>
public sealed class NativeGenerationEndpointsTests : IClassFixture<WebApplicationFactory<Program>>
{
    private readonly WebApplicationFactory<Program> _factory;

    public NativeGenerationEndpointsTests(WebApplicationFactory<Program> factory)
    {
        _factory = factory.WithWebHostBuilder(builder => builder.UseSetting("HartsyInference:Backend", "cpu"));
    }

    [Fact]
    public async Task Images_UnresolvableModel_Returns400()
    {
        using HttpClient client = _factory.CreateClient();
        HttpResponseMessage resp = await client.PostAsJsonAsync("/v1/native/images", new
        {
            model = "not-a-real-model-id",
            request = new { prompt = "a cat" },
        });
        Assert.Equal(HttpStatusCode.BadRequest, resp.StatusCode);
        JsonElement body = await resp.Content.ReadFromJsonAsync<JsonElement>();
        Assert.Contains("checkpoint", body.GetProperty("error").GetProperty("message").GetString(), StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task Images_MissingPrompt_Returns400()
    {
        using HttpClient client = _factory.CreateClient();
        HttpResponseMessage resp = await client.PostAsJsonAsync("/v1/native/images", new
        {
            model = "sdxl",
            request = new { },
        });
        Assert.Equal(HttpStatusCode.BadRequest, resp.StatusCode);
    }

    [Fact]
    public async Task Text_UnresolvableModel_Returns400()
    {
        using HttpClient client = _factory.CreateClient();
        HttpResponseMessage resp = await client.PostAsJsonAsync("/v1/native/text", new
        {
            model = "not-a-real-model-id",
            request = new { messages = new[] { new { role = "User", content = "hi" } } },
        });
        Assert.Equal(HttpStatusCode.BadRequest, resp.StatusCode);
    }

    [Fact]
    public async Task Text_CountTokens_UnresolvableModel_FallsBackToHeuristicInsteadOfFailing()
    {
        // ITextService.CountTokens deliberately never loads a model just to count (TextService.CountTokens):
        // with no matching tokenizer loaded it falls back to a cheap chars/4 heuristic rather than throwing —
        // so an unresolvable model is NOT an error here, unlike every other native route.
        using HttpClient client = _factory.CreateClient();
        HttpResponseMessage resp = await client.PostAsJsonAsync("/v1/native/text/count-tokens", new
        {
            model = "not-a-real-model-id",
            text = "hello world",
        });
        Assert.Equal(HttpStatusCode.OK, resp.StatusCode);
        JsonElement body = await resp.Content.ReadFromJsonAsync<JsonElement>();
        Assert.True(body.GetProperty("tokens").GetInt32() > 0);
    }

    [Fact]
    public async Task ImagesStream_UnresolvableModel_ReportsErrorEventNotHang()
    {
        using HttpClient client = _factory.CreateClient();
        using HttpRequestMessage req = new HttpRequestMessage(HttpMethod.Post, "/v1/native/images/stream")
        {
            Content = JsonContent.Create(new
            {
                model = "not-a-real-model-id",
                request = new { prompt = "a cat" },
            }),
        };

        using HttpResponseMessage resp = await client.SendAsync(req, HttpCompletionOption.ResponseHeadersRead);
        Assert.Equal(HttpStatusCode.OK, resp.StatusCode); // SSE headers already sent by the time the producer fails
        string body = await resp.Content.ReadAsStringAsync();
        Assert.Contains("event: error", body);
        Assert.Contains("checkpoint", body, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task TextStream_UnresolvableModel_EmitsErrorStopReasonChunkNotHang()
    {
        // ITextService.StreamAsync never lets a load failure escape as an exception (TextService.StreamAsync
        // catches internally) — it emits a normal "chunk" event carrying Kind=StopReason/Stop=Error instead of
        // SseHelpers' generic "event: error" fallback, which only fires for failures outside the text service's
        // own handling.
        using HttpClient client = _factory.CreateClient();
        using HttpRequestMessage req = new HttpRequestMessage(HttpMethod.Post, "/v1/native/text/stream")
        {
            Content = JsonContent.Create(new
            {
                model = "not-a-real-model-id",
                request = new { messages = new[] { new { role = "User", content = "hi" } } },
            }),
        };

        using HttpResponseMessage resp = await client.SendAsync(req, HttpCompletionOption.ResponseHeadersRead);
        Assert.Equal(HttpStatusCode.OK, resp.StatusCode);
        string body = await resp.Content.ReadAsStringAsync();
        Assert.Contains("event: chunk", body);
        Assert.Contains("\"kind\":\"StopReason\"", body);
        Assert.Contains("\"stop\":\"Error\"", body);
    }
}
