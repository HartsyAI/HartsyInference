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
    public async Task Health_Returns200()
    {
        using HttpClient client = _factory.CreateClient();
        HttpResponseMessage resp = await client.GetAsync("/health");
        Assert.Equal(HttpStatusCode.OK, resp.StatusCode);
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
    public async Task Version_ReturnsBackendSelector()
    {
        using HttpClient client = _factory.CreateClient();
        JsonElement body = await client.GetFromJsonAsync<JsonElement>("/version");
        Assert.Equal("cpu", body.GetProperty("backendSelector").GetString());
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
    public async Task Health_IsReachable_EvenWhenApiKeyConfigured()
    {
        using WebApplicationFactory<Program> factory = _factory.WithWebHostBuilder(builder =>
            builder.UseSetting("HartsyInference:ApiKey", "super-secret-key"));
        using HttpClient client = factory.CreateClient();

        HttpResponseMessage resp = await client.GetAsync("/health");
        Assert.Equal(HttpStatusCode.OK, resp.StatusCode);
    }

    [Fact]
    public async Task AdminCatalog_ReturnsNonEmptyCatalog()
    {
        using HttpClient client = _factory.CreateClient();
        JsonElement body = await client.GetFromJsonAsync<JsonElement>("/admin/catalog");
        Assert.True(body.GetArrayLength() > 0);
    }

    [Fact]
    public async Task AdminCatalog_FiltersByModality()
    {
        using HttpClient client = _factory.CreateClient();
        JsonElement body = await client.GetFromJsonAsync<JsonElement>("/admin/catalog?modality=text");
        Assert.True(body.GetArrayLength() > 0);
        foreach (JsonElement entry in body.EnumerateArray())
            Assert.Equal("Text", entry.GetProperty("modality").GetString());
    }

    [Fact]
    public async Task AdminCatalog_UnknownModality_Returns400()
    {
        using HttpClient client = _factory.CreateClient();
        HttpResponseMessage resp = await client.GetAsync("/admin/catalog?modality=not-a-modality");
        Assert.Equal(HttpStatusCode.BadRequest, resp.StatusCode);
    }

    [Fact]
    public async Task AdminModels_InitiallyEmpty()
    {
        using HttpClient client = _factory.CreateClient();
        JsonElement body = await client.GetFromJsonAsync<JsonElement>("/admin/models");
        Assert.Equal(0, body.GetProperty("loaded").GetArrayLength());
    }

    [Fact]
    public async Task AdminCache_ReturnsScratchDirectoryAndEmptyModels()
    {
        using HttpClient client = _factory.CreateClient();
        JsonElement body = await client.GetFromJsonAsync<JsonElement>("/admin/cache");
        Assert.Equal(_scratchCacheDir, body.GetProperty("directory").GetString());
        Assert.Equal(0, body.GetProperty("models").GetArrayLength());
    }

    [Fact]
    public async Task AdminCacheDelete_UnknownId_Returns404()
    {
        using HttpClient client = _factory.CreateClient();
        HttpResponseMessage resp = await client.DeleteAsync("/admin/cache/not-cached");
        Assert.Equal(HttpStatusCode.NotFound, resp.StatusCode);
    }

    [Fact]
    public async Task AdminModelsPull_UnknownCatalogId_Returns404()
    {
        using HttpClient client = _factory.CreateClient();
        HttpResponseMessage resp = await client.PostAsJsonAsync("/admin/models/pull", new { model = "not-a-catalog-id" });
        Assert.Equal(HttpStatusCode.NotFound, resp.StatusCode);
    }

    [Fact]
    public async Task AdminMemoryFree_NoBody_DefaultsToSoftFree()
    {
        using HttpClient client = _factory.CreateClient();
        HttpResponseMessage resp = await client.PostAsync("/admin/memory/free", content: null);
        Assert.Equal(HttpStatusCode.OK, resp.StatusCode);
        JsonElement body = await resp.Content.ReadFromJsonAsync<JsonElement>();
        Assert.True(body.GetProperty("freed").GetBoolean());
        Assert.False(body.GetProperty("hard").GetBoolean());
    }

    [Fact]
    public async Task AdminMemoryFree_EmptyBody_DefaultsToSoftFree()
    {
        using HttpClient client = _factory.CreateClient();
        HttpResponseMessage resp = await client.PostAsJsonAsync("/admin/memory/free", new { });
        Assert.Equal(HttpStatusCode.OK, resp.StatusCode);
        JsonElement body = await resp.Content.ReadFromJsonAsync<JsonElement>();
        Assert.False(body.GetProperty("hard").GetBoolean());
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

    [Fact]
    public async Task AdminBackend_InvalidSelector_Returns400()
    {
        using HttpClient client = _factory.CreateClient();
        HttpResponseMessage resp = await client.PostAsJsonAsync("/admin/backend", new { backend = "quantum" });
        Assert.Equal(HttpStatusCode.BadRequest, resp.StatusCode);
    }

    [Fact]
    public async Task AdminBackend_ValidSelector_Switches()
    {
        using HttpClient client = _factory.CreateClient();
        HttpResponseMessage resp = await client.PostAsJsonAsync("/admin/backend", new { backend = "cpu" });
        Assert.Equal(HttpStatusCode.OK, resp.StatusCode);
        JsonElement body = await resp.Content.ReadFromJsonAsync<JsonElement>();
        Assert.Equal("cpu", body.GetProperty("backend").GetString());
    }

    // AdminQueue coverage moved to VideoWorldEndpointsTests.AdminQueue_ReportsBothFastAndLongRunning — Phase 5
    // changed /admin/queue's response shape from a flat {pending,maxConcurrency,maxQueueDepth} to
    // {fast:{...},longRunning:{...}} when the long-running queue was split out.
}
