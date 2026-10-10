using System.Net;
using System.Net.Http.Json;
using System.Text;
using System.Text.Json;
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

    [Theory]
    [InlineData("json_object")]
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

    [Theory]
    [InlineData("""[{"type":"text"}]""")]
    [InlineData("""[{"type":"text","text":5}]""")]
    public async Task Chat_TextPartWithoutStringText_Returns400(string content)
    {
        using HttpClient client = _factory.CreateClient();
        using StringContent body = new($$"""{"model":"whatever","messages":[{"role":"user","content":{{content}}}]}""", Encoding.UTF8, "application/json");

        HttpResponseMessage resp = await client.PostAsync("/v1/chat/completions", body);

        Assert.Equal(HttpStatusCode.BadRequest, resp.StatusCode);
        using JsonDocument error = JsonDocument.Parse(await resp.Content.ReadAsStringAsync());
        Assert.Equal("A 'text' content part needs a string 'text'.", error.RootElement.GetProperty("error").GetProperty("message").GetString());
    }

    [Theory]
    [InlineData("data:image/png;base64,@@@@", "Image data is not valid base64.")]
    [InlineData("data:image/jpeg;base64,/9j/4AAQSkZJRgAB", "Image data could not be decoded: Unsupported image format (leading bytes JPEG).")]
    public async Task Chat_ImageThatCannotBeDecoded_Returns400NotA500(string url, string messageStart)
    {
        using HttpClient client = _factory.CreateClient();
        HttpResponseMessage resp = await client.PostAsJsonAsync("/v1/chat/completions", new
        {
            model = "whatever",
            messages = new[] { new { role = "user", content = new object[] { new { type = "image_url", image_url = new { url } } } } },
        });

        Assert.Equal(HttpStatusCode.BadRequest, resp.StatusCode);
        using JsonDocument error = JsonDocument.Parse(await resp.Content.ReadAsStringAsync());
        Assert.StartsWith(messageStart, error.RootElement.GetProperty("error").GetProperty("message").GetString(), StringComparison.Ordinal);
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
    public async Task Models_List_ReturnsNonEmptyCatalogInOpenAiShape()
    {
        using HttpClient client = _factory.CreateClient();
        JsonElement body = await client.GetFromJsonAsync<JsonElement>("/v1/models");
        Assert.Equal("list", body.GetProperty("object").GetString());
        JsonElement data = body.GetProperty("data");
        Assert.True(data.GetArrayLength() > 0);
        JsonElement first = data[0];
        Assert.Equal("model", first.GetProperty("object").GetString());
        Assert.True(first.GetProperty("created").GetInt64() > 0);
        Assert.Equal("hartsyinference", first.GetProperty("owned_by").GetString());
    }

    [Fact]
    public async Task Models_Retrieve_UnknownId_Returns404()
    {
        using HttpClient client = _factory.CreateClient();
        HttpResponseMessage resp = await client.GetAsync("/v1/models/not-a-real-model-id");
        Assert.Equal(HttpStatusCode.NotFound, resp.StatusCode);
    }

    [Fact]
    public async Task Speech_UnsupportedResponseFormat_Returns400()
    {
        using HttpClient client = _factory.CreateClient();
        HttpResponseMessage resp = await client.PostAsJsonAsync("/v1/audio/speech", new
        {
            model = "whatever",
            input = "hello",
            response_format = "mp3",
        });
        Assert.Equal(HttpStatusCode.BadRequest, resp.StatusCode);
    }

    [Fact]
    public async Task Speech_UnresolvableModel_Returns501()
    {
        // Speech resolves its catalog id through XCatalog.Resolve(id), which throws NotSupportedException for
        // an unknown id -> 501, same as the native /v1/native/speech route (see AudioEndpointsTests) -- not the
        // 400 that Image/Text give for an unresolvable model.
        using HttpClient client = _factory.CreateClient();
        HttpResponseMessage resp = await client.PostAsJsonAsync("/v1/audio/speech", new
        {
            model = "not-a-real-model-id",
            input = "hello",
        });
        Assert.Equal(HttpStatusCode.NotImplemented, resp.StatusCode);
    }
}
