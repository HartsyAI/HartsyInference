using System.Net;
using System.Net.Http.Json;
using System.Text.Json;
using HartsyInference.Core.Configuration;
using Microsoft.AspNetCore.Mvc.Testing;
using Xunit;

namespace HartsyInference.API.Tests;

/// <summary>In-process tests for the settings routes. <c>PUT</c> writes the real settings file, so the whole class
/// is pointed at a temporary one through <see cref="KnobFile.ExplicitPath"/> — without that these tests would edit
/// the machine's actual configuration, which is the kind of test that is discovered by someone else's surprise.</summary>
/// <remarks>Not parallelised with the other API tests: <see cref="KnobFile.ExplicitPath"/> and the override store
/// are process-wide, so a concurrent class reading a setting would see this one's writes.</remarks>
[Collection("settings-file")]
public sealed class SettingsEndpointsTests : IClassFixture<WebApplicationFactory<Program>>, IDisposable
{
    private readonly WebApplicationFactory<Program> _factory;
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "hartsy-api-settings-" + Guid.NewGuid().ToString("N"));
    private readonly string? _previous = KnobFile.ExplicitPath;

    public SettingsEndpointsTests(WebApplicationFactory<Program> factory)
    {
        Directory.CreateDirectory(_dir);
        KnobFile.ExplicitPath = Path.Combine(_dir, "settings.json");
        _factory = factory.WithWebHostBuilder(builder => builder.UseSetting("HartsyInference:Backend", "cpu"));
    }

    public void Dispose()
    {
        KnobFile.ExplicitPath = _previous;
        KnobStore.ResetOverrides();
        try
        {
            Directory.Delete(_dir, recursive: true);
        }
        catch (IOException)
        {
            // A leftover temp directory is not worth failing a test over.
        }
    }

    /// <summary>The route is named /settings, so it has to describe the engine and not only the ASP.NET host.</summary>
    [Fact]
    public async Task Get_ReportsBothServerAndEngine()
    {
        using HttpClient client = _factory.CreateClient();
        JsonElement body = await client.GetFromJsonAsync<JsonElement>("/settings");

        Assert.True(body.TryGetProperty("server", out JsonElement server));
        Assert.True(server.TryGetProperty("backend", out _));

        Assert.True(body.TryGetProperty("engine", out JsonElement engine));
        Assert.Equal(KnobFile.Path, engine.GetProperty("file").GetString());
        Assert.NotEmpty(engine.GetProperty("settings").EnumerateArray());
    }

    /// <summary>Every setting carries where its value came from, which is the whole reason the endpoint is useful.</summary>
    [Fact]
    public async Task GetOne_CarriesValueSourceAndScope()
    {
        using HttpClient client = _factory.CreateClient();
        JsonElement body = await client.GetFromJsonAsync<JsonElement>("/settings/engine/vram.keepModels");

        Assert.Equal("vram.keepModels", body.GetProperty("id").GetString());
        Assert.Equal("Boolean", body.GetProperty("type").GetString());
        Assert.False(string.IsNullOrEmpty(body.GetProperty("source").GetString()));
        Assert.Equal("next run", body.GetProperty("appliesAt").GetString());
    }

    /// <summary>A Construction-scoped setting says it needs a restart rather than implying it took effect.</summary>
    [Fact]
    public async Task GetOne_ConstructionScopeSaysRestart()
    {
        using HttpClient client = _factory.CreateClient();
        JsonElement body = await client.GetFromJsonAsync<JsonElement>("/settings/engine/vram.lowvramQuant");
        Assert.Equal("restart", body.GetProperty("appliesAt").GetString());
    }

    /// <summary>The point of PUT: the value is written to the file, not just held for this process.</summary>
    [Fact]
    public async Task Put_PersistsToTheFile()
    {
        using HttpClient client = _factory.CreateClient();
        HttpResponseMessage resp = await client.PutAsJsonAsync("/settings/engine/paths.modelsRoot", new { value = "/mnt/put-test" });
        Assert.Equal(HttpStatusCode.OK, resp.StatusCode);

        JsonElement body = await resp.Content.ReadFromJsonAsync<JsonElement>();
        Assert.Equal("/mnt/put-test", body.GetProperty("value").GetString());
        Assert.Equal("settings file", body.GetProperty("source").GetString());

        Assert.Contains("/mnt/put-test", await File.ReadAllTextAsync(KnobFile.Path), StringComparison.Ordinal);
    }

    /// <summary>A clamped setting stores what the engine will honour, so the file cannot disagree with the engine.</summary>
    [Fact]
    public async Task Put_StoresTheCoercedValue()
    {
        using HttpClient client = _factory.CreateClient();
        HttpResponseMessage resp = await client.PutAsJsonAsync("/settings/engine/numerics.gemvWpb", new { value = "999" });
        Assert.Equal(HttpStatusCode.OK, resp.StatusCode);
        JsonElement body = await resp.Content.ReadFromJsonAsync<JsonElement>();
        Assert.Equal(16, body.GetProperty("value").GetInt32());
    }

    [Fact]
    public async Task GetOne_UnknownIdReturns404()
    {
        using HttpClient client = _factory.CreateClient();
        HttpResponseMessage resp = await client.GetAsync("/settings/engine/paths.noSuchSetting");
        Assert.Equal(HttpStatusCode.NotFound, resp.StatusCode);
    }

    [Fact]
    public async Task Put_UnknownIdReturns400()
    {
        using HttpClient client = _factory.CreateClient();
        HttpResponseMessage resp = await client.PutAsJsonAsync("/settings/engine/paths.noSuchSetting", new { value = "x" });
        Assert.Equal(HttpStatusCode.BadRequest, resp.StatusCode);
    }

    /// <summary>A value the setting's type cannot hold is refused when it is sent, not at the next startup.</summary>
    [Fact]
    public async Task Put_WrongTypeReturns400()
    {
        using HttpClient client = _factory.CreateClient();
        HttpResponseMessage resp = await client.PutAsJsonAsync("/settings/engine/vram.keepModels", new { value = "banana" });
        Assert.Equal(HttpStatusCode.BadRequest, resp.StatusCode);
    }

    [Fact]
    public async Task Put_MissingValueReturns400()
    {
        using HttpClient client = _factory.CreateClient();
        HttpResponseMessage resp = await client.PutAsJsonAsync("/settings/engine/vram.keepModels", new { });
        Assert.Equal(HttpStatusCode.BadRequest, resp.StatusCode);
    }
}

/// <summary>Serialises the settings-file tests; <see cref="KnobFile.ExplicitPath"/> is process-wide.</summary>
[CollectionDefinition("settings-file", DisableParallelization = true)]
public sealed class SettingsFileCollection
{
}
