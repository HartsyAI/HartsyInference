using System.Net;
using System.Net.Http.Json;
using System.Text.Json;
using HartsyInference.Engine;
using Microsoft.AspNetCore.Mvc.Testing;
using Xunit;

namespace HartsyInference.API.Tests;

/// <summary>In-process HTTP integration tests for the Phase 1 skeleton (health/ready/version/settings/admin)
/// wired onto <see cref="IInferenceEngine"/>. Runs on the CPU backend with a scratch model cache directory, so
/// these never touch a GPU or the developer's real <c>~/.hartsyinference/models</c> cache.</summary>
public sealed class SkeletonEndpointsTests : IClassFixture<WebApplicationFactory<Program>>, IDisposable
{
    private readonly WebApplicationFactory<Program> _factory;
    private readonly string _scratchCacheDir = Path.Combine(Path.GetTempPath(), "hartsy-api-tests-" + Path.GetRandomFileName());

    public SkeletonEndpointsTests(WebApplicationFactory<Program> factory)
    {
        _factory = factory.WithWebHostBuilder(builder =>
        {
            builder.UseSetting("HartsyInference:Backend", "cpu");
            builder.UseSetting("HartsyInference:ModelCacheDirectory", _scratchCacheDir);
        });
    }

    public void Dispose()
    {
        if (Directory.Exists(_scratchCacheDir))
            Directory.Delete(_scratchCacheDir, recursive: true);
    }

    [Fact]
    public async Task Ready_ReportsBackend()
    {
        using HttpClient client = _factory.CreateClient();
        HttpResponseMessage resp = await client.GetAsync("/ready");
        Assert.Equal(HttpStatusCode.OK, resp.StatusCode);
        JsonElement body = await resp.Content.ReadFromJsonAsync<JsonElement>();
        Assert.Equal("ready", body.GetProperty("status").GetString());
        Assert.False(string.IsNullOrEmpty(body.GetProperty("backend").GetString()));
    }

    [Fact]
    public async Task Settings_RedactsApiKeyValue_ButReportsWhetherOneIsConfigured()
    {
        using WebApplicationFactory<Program> factory = _factory.WithWebHostBuilder(builder =>
            builder.UseSetting("HartsyInference:ApiKey", "super-secret-key"));
        using HttpClient client = factory.CreateClient();
        client.DefaultRequestHeaders.Add("x-api-key", "super-secret-key");

        HttpResponseMessage resp = await client.GetAsync("/settings");
        Assert.Equal(HttpStatusCode.OK, resp.StatusCode);
        string body = await resp.Content.ReadAsStringAsync();
        Assert.DoesNotContain("super-secret-key", body);
        Assert.Contains("\"apiKeyConfigured\":true", body);
    }

    [Fact]
    public async Task Admin_WithoutApiKey_WhenConfigured_Returns401()
    {
        using WebApplicationFactory<Program> factory = _factory.WithWebHostBuilder(builder =>
            builder.UseSetting("HartsyInference:ApiKey", "super-secret-key"));
        using HttpClient client = factory.CreateClient();

        HttpResponseMessage resp = await client.GetAsync("/admin/catalog");
        Assert.Equal(HttpStatusCode.Unauthorized, resp.StatusCode);
    }

    [Fact]
    public async Task AdminCatalog_ReturnsNonEmptyCatalog()
    {
        using HttpClient client = _factory.CreateClient();
        JsonElement body = await client.GetFromJsonAsync<JsonElement>("/admin/catalog");
        Assert.True(body.GetArrayLength() > 0);
    }

    [Fact]
    public async Task AdminCatalog_UnknownModality_Returns400()
    {
        using HttpClient client = _factory.CreateClient();
        HttpResponseMessage resp = await client.GetAsync("/admin/catalog?modality=not-a-modality");
        Assert.Equal(HttpStatusCode.BadRequest, resp.StatusCode);
    }

    /// <summary>The quantize endpoint validates before it touches the filesystem, so a bad request is a 400 rather
    /// than a partially-written multi-GB file.</summary>
    [Theory]
    [InlineData("{\"modelPath\":\"\",\"out\":\"/tmp/x.gguf\"}")]
    [InlineData("{\"modelPath\":\"/tmp/in.safetensors\",\"out\":\"/tmp/x.gguf\",\"format\":\"not-a-format\"}")]
    public async Task AdminModelsQuantize_BadRequest_Returns400(string body)
    {
        using HttpClient client = _factory.CreateClient();
        using StringContent content = new(body, System.Text.Encoding.UTF8, "application/json");
        HttpResponseMessage resp = await client.PostAsync("/admin/models/quantize", content);
        Assert.Equal(HttpStatusCode.BadRequest, resp.StatusCode);
    }

    [Fact]
    public async Task AdminMemoryFree_Hard_RecreatesBackendAndStaysUsable()
    {
        // Real behavior, not just routing: {hard:true} calls SetBackend (dispose+recreate) rather than the soft
        // evict/trim path -- verified here by driving it on the CPU backend (cheap, no GPU needed to construct)
        // and confirming the engine is still healthy/usable immediately afterward.
        using HttpClient client = _factory.CreateClient();
        HttpResponseMessage resp = await client.PostAsJsonAsync("/admin/memory/free", new { hard = true });
        Assert.Equal(HttpStatusCode.OK, resp.StatusCode);
        JsonElement body = await resp.Content.ReadFromJsonAsync<JsonElement>();
        Assert.True(body.GetProperty("freed").GetBoolean());
        Assert.True(body.GetProperty("hard").GetBoolean());

        HttpResponseMessage ready = await client.GetAsync("/ready");
        Assert.Equal(HttpStatusCode.OK, ready.StatusCode);
    }

    // AdminQueue coverage moved to VideoWorldEndpointsTests.AdminQueue_ReportsBothFastAndLongRunning — Phase 5
    // changed /admin/queue's response shape from a flat {pending,maxConcurrency,maxQueueDepth} to
    // {fast:{...},longRunning:{...}} when the long-running queue was split out.
}
