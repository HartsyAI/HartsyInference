using System.Net;
using System.Net.Http.Json;
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
    public async Task Models_Retrieve_KnownId_ReturnsEntry()
    {
        using HttpClient client = _factory.CreateClient();
        JsonElement catalog = await client.GetFromJsonAsync<JsonElement>("/v1/models");
        string knownId = catalog.GetProperty("data")[0].GetProperty("id").GetString()!;

        JsonElement entry = await client.GetFromJsonAsync<JsonElement>($"/v1/models/{knownId}");
        Assert.Equal(knownId, entry.GetProperty("id").GetString());
    }

    [Fact]
    public async Task Models_Retrieve_UnknownId_Returns404()
    {
        using HttpClient client = _factory.CreateClient();
        HttpResponseMessage resp = await client.GetAsync("/v1/models/not-a-real-model-id");
        Assert.Equal(HttpStatusCode.NotFound, resp.StatusCode);
    }

    [Fact]
    public async Task Chat_WithTools_UnresolvableModel_StillReturns400NotA500()
    {
        // Confirms the tools/tool_choice mapping in ToTextRequest doesn't itself blow up before model
        // resolution -- same "unresolvable model" 400 as the no-tools case, not an unrelated server error.
        using HttpClient client = _factory.CreateClient();
        HttpResponseMessage resp = await client.PostAsJsonAsync("/v1/chat/completions", new
        {
            model = "not-a-real-model-id",
            messages = new[] { new { role = "user", content = "what's the weather in Paris?" } },
            tools = new[]
            {
                new
                {
                    type = "function",
                    function = new { name = "get_weather", description = "Get the weather", parameters = new { type = "object", properties = new { } } },
                },
            },
            tool_choice = "auto",
        });
        Assert.Equal(HttpStatusCode.BadRequest, resp.StatusCode);
    }

    [Fact]
    public async Task Chat_ForcedToolChoice_UnresolvableModel_StillReturns400NotA500()
    {
        using HttpClient client = _factory.CreateClient();
        HttpResponseMessage resp = await client.PostAsJsonAsync("/v1/chat/completions", new
        {
            model = "not-a-real-model-id",
            messages = new[] { new { role = "user", content = "hi" } },
            tools = new[]
            {
                new { type = "function", function = new { name = "get_weather", parameters = new { } } },
            },
            tool_choice = new { type = "function", function = new { name = "get_weather" } },
        });
        Assert.Equal(HttpStatusCode.BadRequest, resp.StatusCode);
    }

    [Fact]
    public async Task Speech_MissingModel_Returns400()
    {
        using HttpClient client = _factory.CreateClient();
        HttpResponseMessage resp = await client.PostAsJsonAsync("/v1/audio/speech", new { input = "hello" });
        Assert.Equal(HttpStatusCode.BadRequest, resp.StatusCode);
    }

    [Fact]
    public async Task Speech_MissingInput_Returns400()
    {
        using HttpClient client = _factory.CreateClient();
        HttpResponseMessage resp = await client.PostAsJsonAsync("/v1/audio/speech", new { model = "whatever" });
        Assert.Equal(HttpStatusCode.BadRequest, resp.StatusCode);
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

    [Fact]
    public async Task Transcriptions_NonMultipartBody_Returns400()
    {
        using HttpClient client = _factory.CreateClient();
        HttpResponseMessage resp = await client.PostAsJsonAsync("/v1/audio/transcriptions", new { model = "whatever" });
        Assert.Equal(HttpStatusCode.BadRequest, resp.StatusCode);
    }

    [Fact]
    public async Task Transcriptions_MissingFile_Returns400()
    {
        using HttpClient client = _factory.CreateClient();
        using MultipartFormDataContent form = new MultipartFormDataContent { { new StringContent("whatever"), "model" } };
        HttpResponseMessage resp = await client.PostAsync("/v1/audio/transcriptions", form);
        Assert.Equal(HttpStatusCode.BadRequest, resp.StatusCode);
    }

    [Fact]
    public async Task Transcriptions_MissingModel_Returns400()
    {
        using HttpClient client = _factory.CreateClient();
        using MultipartFormDataContent form = new MultipartFormDataContent
        {
            { new ByteArrayContent([0x52, 0x49, 0x46, 0x46]), "file", "clip.wav" },
        };
        HttpResponseMessage resp = await client.PostAsync("/v1/audio/transcriptions", form);
        Assert.Equal(HttpStatusCode.BadRequest, resp.StatusCode);
    }

    [Fact]
    public async Task Transcriptions_UnsupportedResponseFormat_Returns400()
    {
        using HttpClient client = _factory.CreateClient();
        using MultipartFormDataContent form = new MultipartFormDataContent
        {
            { new ByteArrayContent([0x52, 0x49, 0x46, 0x46]), "file", "clip.wav" },
            { new StringContent("whatever"), "model" },
            { new StringContent("srt"), "response_format" },
        };
        HttpResponseMessage resp = await client.PostAsync("/v1/audio/transcriptions", form);
        Assert.Equal(HttpStatusCode.BadRequest, resp.StatusCode);
    }

    [Fact]
    public async Task Transcriptions_UnresolvableModel_Returns501()
    {
        // Same NotSupportedException -> 501 catalog-resolution behavior as Speech/the native /v1/native/transcribe
        // route (see AudioEndpointsTests) -- confirmed empirically, not assumed as 400 like Image/Text.
        using HttpClient client = _factory.CreateClient();
        using MultipartFormDataContent form = new MultipartFormDataContent
        {
            { new ByteArrayContent([0x52, 0x49, 0x46, 0x46]), "file", "clip.wav" },
            { new StringContent("not-a-real-model-id"), "model" },
        };
        HttpResponseMessage resp = await client.PostAsync("/v1/audio/transcriptions", form);
        Assert.Equal(HttpStatusCode.NotImplemented, resp.StatusCode);
    }

    [Fact]
    public async Task Embeddings_MissingModel_Returns400()
    {
        using HttpClient client = _factory.CreateClient();
        HttpResponseMessage resp = await client.PostAsJsonAsync("/v1/embeddings", new { input = "hello" });
        Assert.Equal(HttpStatusCode.BadRequest, resp.StatusCode);
    }

    [Fact]
    public async Task Embeddings_UnsupportedEncodingFormat_Returns400()
    {
        using HttpClient client = _factory.CreateClient();
        HttpResponseMessage resp = await client.PostAsJsonAsync("/v1/embeddings", new
        {
            model = "whatever",
            input = "hello",
            encoding_format = "base64",
        });
        Assert.Equal(HttpStatusCode.BadRequest, resp.StatusCode);
    }

    [Fact]
    public async Task Embeddings_InputWrongType_Returns400()
    {
        using HttpClient client = _factory.CreateClient();
        HttpResponseMessage resp = await client.PostAsJsonAsync("/v1/embeddings", new
        {
            model = "whatever",
            input = 12345,
        });
        Assert.Equal(HttpStatusCode.BadRequest, resp.StatusCode);
    }

    [Fact]
    public async Task Embeddings_EmptyInputArray_Returns400()
    {
        using HttpClient client = _factory.CreateClient();
        HttpResponseMessage resp = await client.PostAsJsonAsync("/v1/embeddings", new
        {
            model = "whatever",
            input = Array.Empty<string>(),
        });
        Assert.Equal(HttpStatusCode.BadRequest, resp.StatusCode);
    }

    [Fact]
    public async Task Embeddings_AcceptsStringOrArrayInput_BothReachModelResolution()
    {
        // Confirms both OpenAI-accepted `input` shapes (bare string, string array) parse past validation and reach
        // model resolution (same 400 either way here, since the model is unresolvable in this hermetic run) --
        // not that one shape 400s at parsing while the other doesn't.
        using HttpClient client = _factory.CreateClient();
        HttpResponseMessage stringResp = await client.PostAsJsonAsync("/v1/embeddings", new { model = "not-a-real-model-id", input = "hello" });
        HttpResponseMessage arrayResp = await client.PostAsJsonAsync("/v1/embeddings", new { model = "not-a-real-model-id", input = new[] { "hello", "world" } });
        Assert.Equal(HttpStatusCode.BadRequest, stringResp.StatusCode);
        Assert.Equal(HttpStatusCode.BadRequest, arrayResp.StatusCode);
    }
}
