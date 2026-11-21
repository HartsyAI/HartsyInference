using System.Text.Json;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Hosting.Server;
using Microsoft.AspNetCore.Hosting.Server.Features;
using Microsoft.Extensions.DependencyInjection;

namespace HartsyInference.API.Tests;

/// <summary>The recipe routes carry each family's prompt block in the shape a client reads: camelCase, string values from the
/// closed vocabulary, and no key for a field the guide did not establish.</summary>
public sealed class RecipePromptRouteTests : IAsyncLifetime
{
    private WebApplication? _app;
    private HttpClient _http = null!;   // set in InitializeAsync

    public async Task InitializeAsync()
    {
        WebApplicationBuilder builder = WebApplication.CreateBuilder();
        builder.WebHost.UseUrls("http://127.0.0.1:0");
        builder.Services.AddHartsyInference();
        _app = builder.Build();
        _app.MapHartsyInferenceEndpoints();
        await _app.StartAsync();
        string address = _app.Services.GetRequiredService<IServer>().Features.Get<IServerAddressesFeature>()!.Addresses.First();
        _http = new HttpClient { BaseAddress = new Uri(address), Timeout = TimeSpan.FromSeconds(30) };
    }

    public async Task DisposeAsync()
    {
        _http?.Dispose();
        if (_app is not null) await _app.DisposeAsync();
    }

    [Fact]
    public async Task TheFamilyRouteAnswersWithThePromptBlock()
    {
        using JsonDocument body = JsonDocument.Parse(await _http.GetStringAsync("/v1/native/recipes/image/ideogram4"));
        JsonElement prompt = body.RootElement.GetProperty("prompt");

        Assert.Equal("image", prompt.GetProperty("kind").GetString());
        Assert.Equal("json", prompt.GetProperty("format").GetString());
        Assert.Equal("yxyx-1000", prompt.GetProperty("boxes").GetString());
        Assert.Equal(["boxes", "text-rendering"], prompt.GetProperty("supports").EnumerateArray().Select(e => e.GetString()));
        Assert.False(prompt.TryGetProperty("negative", out _));
    }

    [Fact]
    public async Task AVariantRouteAnswersForThatVariant()
    {
        using JsonDocument turbo = JsonDocument.Parse(await _http.GetStringAsync("/v1/native/recipes/image/zimage?variant=turbo"));
        using JsonDocument baseBuild = JsonDocument.Parse(await _http.GetStringAsync("/v1/native/recipes/image/zimage?variant=base"));

        Assert.Equal("none", turbo.RootElement.GetProperty("prompt").GetProperty("negative").GetString());
        Assert.Equal("separate", baseBuild.RootElement.GetProperty("prompt").GetProperty("negative").GetString());
    }

    [Fact]
    public async Task TheListCarriesAProfileForEveryImageAndVideoRecipe()
    {
        using JsonDocument all = JsonDocument.Parse(await _http.GetStringAsync("/v1/native/recipes"));

        JsonElement[] descriptors = [.. all.RootElement.EnumerateArray()];
        Assert.NotEmpty(descriptors);
        foreach (JsonElement descriptor in descriptors)
        {
            string modality = descriptor.GetProperty("modality").GetString()!.ToLowerInvariant();
            Assert.Equal(modality, descriptor.GetProperty("prompt").GetProperty("kind").GetString());
        }
    }
}
