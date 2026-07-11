using System.Net;
using System.Net.Http.Json;
using Microsoft.AspNetCore.Mvc.Testing;
using Xunit;

namespace HartsyInference.Server.Tests;

/// <summary>In-process HTTP integration tests for the chat-completions route and its request validation —
/// exercises the real ASP.NET Core pipeline (routing, DI, DTO binding) via <see cref="WebApplicationFactory{T}"/>,
/// not just the unit-level pieces <c>ServerTests</c> covers. Runs on the default CPU backend with no model
/// loaded, so these never touch the GPU or download anything — they validate the HTTP layer's own logic
/// (missing/invalid fields, unsupported response_format, routing to <see cref="ModelManager"/>), which
/// previously had zero automated coverage beyond manual curl checks during development.</summary>
public sealed class ChatCompletionsIntegrationTests : IClassFixture<WebApplicationFactory<Program>>
{
    private readonly WebApplicationFactory<Program> _factory;

    public ChatCompletionsIntegrationTests(WebApplicationFactory<Program> factory)
    {
        // Force CPU backend explicitly (the default anyway) so this never tries to touch a GPU/PTX
        // directory regardless of what an ambient appsettings.json/env var might say on the box.
        _factory = factory.WithWebHostBuilder(builder =>
        {
            builder.UseSetting("HartsyInference:Backend", "Cpu");
        });
    }

    [Fact]
    public async Task Health_Returns200()
    {
        using HttpClient client = _factory.CreateClient();
        HttpResponseMessage resp = await client.GetAsync("/health");
        Assert.Equal(HttpStatusCode.OK, resp.StatusCode);
    }

    [Fact]
    public async Task Models_InitiallyEmpty()
    {
        using HttpClient client = _factory.CreateClient();
        ModelListResponse? body = await client.GetFromJsonAsync<ModelListResponse>("/v1/models");
        Assert.NotNull(body);
        Assert.Empty(body!.Data);
    }

    [Fact]
    public async Task Chat_MissingModel_Returns400WithClearError()
    {
        using HttpClient client = _factory.CreateClient();
        HttpResponseMessage resp = await client.PostAsJsonAsync("/v1/chat/completions", new
        {
            messages = new[] { new { role = "user", content = "hi" } },
        });
        Assert.Equal(HttpStatusCode.BadRequest, resp.StatusCode);
        OpenAiError? err = await resp.Content.ReadFromJsonAsync<OpenAiError>();
        Assert.Contains("model", err!.Error.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task Chat_UnloadedModel_Returns400()
    {
        using HttpClient client = _factory.CreateClient();
        HttpResponseMessage resp = await client.PostAsJsonAsync("/v1/chat/completions", new
        {
            model = "not-a-loaded-model",
            messages = new[] { new { role = "user", content = "hi" } },
        });
        Assert.Equal(HttpStatusCode.BadRequest, resp.StatusCode);
        OpenAiError? err = await resp.Content.ReadFromJsonAsync<OpenAiError>();
        Assert.Contains("loaded", err!.Error.Message, StringComparison.OrdinalIgnoreCase);
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

    [Fact]
    public async Task Chat_UnsupportedJsonSchemaResponseFormat_Returns400NotSilentlyIgnored()
    {
        using HttpClient client = _factory.CreateClient();
        HttpResponseMessage resp = await client.PostAsJsonAsync("/v1/chat/completions", new
        {
            model = "whatever",
            messages = new[] { new { role = "user", content = "hi" } },
            response_format = new { type = "json_schema" },
        });
        Assert.Equal(HttpStatusCode.BadRequest, resp.StatusCode);
        OpenAiError? err = await resp.Content.ReadFromJsonAsync<OpenAiError>();
        Assert.Contains("json_schema", err!.Error.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task ModelsLoad_EmptyModelField_Returns400WithoutAttemptingAnyLoad()
    {
        // Deliberately NOT testing a nonexistent path/repo id here: LoadAsync falls through to
        // ModelRegistry.LoadAsync (a HuggingFace network call) for anything that isn't an existing local
        // path, which would make this test network-dependent and flaky in CI. The empty-field case is
        // rejected before any load attempt, so it's safe to test in-process with no external dependency.
        using HttpClient client = _factory.CreateClient();
        HttpResponseMessage resp = await client.PostAsJsonAsync("/v1/models/load", new { model = "" });
        Assert.Equal(HttpStatusCode.BadRequest, resp.StatusCode);
    }
}
