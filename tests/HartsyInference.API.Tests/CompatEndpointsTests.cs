using System.Net;
using System.Net.Http.Json;
using Microsoft.AspNetCore.Mvc.Testing;
using Xunit;

namespace HartsyInference.API.Tests;

/// <summary>In-process tests for the OpenAI-compat routes — these are thin wrappers over the same native handlers
/// <see cref="NativeGenerationEndpointsTests"/> exercises, so this file focuses on the DTO-mapping/validation
/// surface specific to the compat layer rather than re-proving the underlying generation path.</summary>
public sealed class CompatEndpointsTests : IClassFixture<WebApplicationFactory<Program>>
{
    private readonly WebApplicationFactory<Program> _factory;

    public CompatEndpointsTests(WebApplicationFactory<Program> factory)
    {
        _factory = factory.WithWebHostBuilder(builder => builder.UseSetting("HartsyInference:Backend", "cpu"));
    }

    [Fact]
    public async Task Chat_MissingModel_Returns400()
    {
        using HttpClient client = _factory.CreateClient();
        HttpResponseMessage resp = await client.PostAsJsonAsync("/v1/chat/completions", new
        {
            messages = new[] { new { role = "user", content = "hi" } },
        });
        Assert.Equal(HttpStatusCode.BadRequest, resp.StatusCode);
    }

    [Fact]
    public async Task Chat_EmptyMessages_Returns400()
    {
        using HttpClient client = _factory.CreateClient();
        HttpResponseMessage resp = await client.PostAsJsonAsync("/v1/chat/completions", new
        {
            model = "whatever",
            messages = Array.Empty<object>(),
        });
        Assert.Equal(HttpStatusCode.BadRequest, resp.StatusCode);
    }

    [Theory]
    [InlineData("json_object")]
    [InlineData("json_schema")]
    public async Task Chat_NonTextResponseFormat_Returns400(string type)
    {
        // The native TextRequest contract has no JSON-mode field yet (see CompatEndpoints.MapCompatEndpoints) —
        // both json_object and json_schema are rejected rather than silently generating unconstrained text.
        using HttpClient client = _factory.CreateClient();
        HttpResponseMessage resp = await client.PostAsJsonAsync("/v1/chat/completions", new
        {
            model = "whatever",
            messages = new[] { new { role = "user", content = "hi" } },
            response_format = new { type },
        });
        Assert.Equal(HttpStatusCode.BadRequest, resp.StatusCode);
    }

    [Fact]
    public async Task Chat_UnresolvableModel_Returns400()
    {
        using HttpClient client = _factory.CreateClient();
        HttpResponseMessage resp = await client.PostAsJsonAsync("/v1/chat/completions", new
        {
            model = "not-a-real-model-id",
            messages = new[] { new { role = "user", content = "hi" } },
        });
        Assert.Equal(HttpStatusCode.BadRequest, resp.StatusCode);
    }

    [Fact]
    public async Task Chat_Streaming_UnresolvableModel_EndsWithDoneNotHang()
    {
        // TextService.StreamAsync never lets a load failure escape as an exception — it surfaces as a normal
        // StopReason.Error chunk, which ToFinishReason maps to "stop" (OpenAI's vocabulary has no error slot).
        // The point of this test is that the stream still terminates cleanly with [DONE] rather than hanging.
        using HttpClient client = _factory.CreateClient();
        using HttpRequestMessage req = new HttpRequestMessage(HttpMethod.Post, "/v1/chat/completions")
        {
            Content = JsonContent.Create(new
            {
                model = "not-a-real-model-id",
                messages = new[] { new { role = "user", content = "hi" } },
                stream = true,
            }),
        };

        using HttpResponseMessage resp = await client.SendAsync(req, HttpCompletionOption.ResponseHeadersRead);
        Assert.Equal(HttpStatusCode.OK, resp.StatusCode);
        string body = await resp.Content.ReadAsStringAsync();
        Assert.Contains("data: [DONE]", body);
        Assert.DoesNotContain("event:", body); // real OpenAI SSE framing has no named events, unlike /v1/native/*
    }

    [Fact]
    public async Task Images_MissingModel_Returns400()
    {
        using HttpClient client = _factory.CreateClient();
        HttpResponseMessage resp = await client.PostAsJsonAsync("/v1/images/generations", new { prompt = "a cat" });
        Assert.Equal(HttpStatusCode.BadRequest, resp.StatusCode);
    }

    [Fact]
    public async Task Images_UnresolvableModel_Returns400()
    {
        using HttpClient client = _factory.CreateClient();
        HttpResponseMessage resp = await client.PostAsJsonAsync("/v1/images/generations", new
        {
            model = "not-a-real-model-id",
            prompt = "a cat",
        });
        Assert.Equal(HttpStatusCode.BadRequest, resp.StatusCode);
    }

    [Fact]
    public async Task ImagesStream_MissingModel_Returns400BeforeSseHeaders()
    {
        using HttpClient client = _factory.CreateClient();
        HttpResponseMessage resp = await client.PostAsJsonAsync("/v1/images/generations/stream", new { prompt = "a cat" });
        Assert.Equal(HttpStatusCode.BadRequest, resp.StatusCode);
    }
}
