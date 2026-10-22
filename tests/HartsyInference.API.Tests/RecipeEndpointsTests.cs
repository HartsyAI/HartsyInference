using System.Net;
using System.Net.Http.Json;
using System.Text.Json;
using HartsyInference.Engine;
using HartsyInference.Engine.Recipes;
using Microsoft.AspNetCore.Mvc.Testing;
using Xunit;

namespace HartsyInference.API.Tests;

/// <summary>HTTP contract coverage for the native recipe catalog: listing, family lookup, describe and typed errors.</summary>
public sealed class RecipeEndpointsTests : IClassFixture<WebApplicationFactory<Program>>
{
    private readonly WebApplicationFactory<Program> _factory;

    public RecipeEndpointsTests(WebApplicationFactory<Program> factory)
    {
        _factory = factory.WithWebHostBuilder(builder => builder.UseSetting("HartsyInference:Backend", "cpu"));
    }

    [Fact]
    public async Task List_ReturnsEveryRegisteredFamilyInRegistryOrder()
    {
        using HttpClient client = _factory.CreateClient();
        JsonElement list = await client.GetFromJsonAsync<JsonElement>("/v1/native/recipes");

        string[] families = list.EnumerateArray()
            .Select(item => item.GetProperty("recipe").GetProperty("familyId").GetString()!)
            .ToArray();
        string[] expected = [.. RecipeRegistry.DefaultNames, .. VideoRecipeRegistry.DefaultNames];
        Assert.Equal(expected, families);
    }

    [Fact]
    public async Task List_FiltersByModality()
    {
        using HttpClient client = _factory.CreateClient();
        JsonElement video = await client.GetFromJsonAsync<JsonElement>("/v1/native/recipes?modality=video");

        Assert.Equal(VideoRecipeRegistry.DefaultNames.Count, video.GetArrayLength());
        Assert.All(video.EnumerateArray(), item => Assert.Equal("video", item.GetProperty("$type").GetString()));
    }

    [Fact]
    public async Task Family_UnknownFamilyIsNotFound()
    {
        using HttpClient client = _factory.CreateClient();
        using HttpResponseMessage response = await client.GetAsync("/v1/native/recipes/image/no-such-family");

        Assert.Equal(HttpStatusCode.NotFound, response.StatusCode);
    }

    [Theory]
    [InlineData("/v1/native/recipes?modality=text")]
    [InlineData("/v1/native/recipes/text/sdxl")]
    [InlineData("/v1/native/recipes/image/qwen-image?variant=no-such-variant")]
    public async Task BadModalityOrVariant_IsBadRequest(string route)
    {
        using HttpClient client = _factory.CreateClient();
        using HttpResponseMessage response = await client.GetAsync(route);

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
    }

    [Fact]
    public async Task UnknownVariant_NamesEveryValidVariantId()
    {
        using HttpClient client = _factory.CreateClient();
        JsonElement family = await client.GetFromJsonAsync<JsonElement>("/v1/native/recipes/image/qwen-image");
        string[] valid = family.GetProperty("variants").EnumerateArray().Select(id => id.GetString()!).ToArray();
        Assert.NotEmpty(valid);

        using HttpResponseMessage response = await client.GetAsync("/v1/native/recipes/image/qwen-image?variant=no-such-variant");
        string json = await response.Content.ReadAsStringAsync();
        foreach (string id in valid)
        {
            Assert.Contains(id, json, StringComparison.Ordinal);
        }
    }

    [Fact]
    public async Task Family_CarriesTheProfileBindingOnlyForMiniMaxH3()
    {
        using HttpClient client = _factory.CreateClient();
        JsonElement h3 = await client.GetFromJsonAsync<JsonElement>("/v1/native/recipes/video/minimax-h3");
        JsonElement wan = await client.GetFromJsonAsync<JsonElement>("/v1/native/recipes/video/wan");

        Assert.True(h3.GetProperty("profileBound").GetBoolean());
        Assert.False(wan.GetProperty("profileBound").GetBoolean());
    }

    [Fact]
    public async Task Describe_ReturnsTheVideoDescriptorForAWanSpec()
    {
        using HttpClient client = _factory.CreateClient();
        using HttpResponseMessage response = await client.PostAsJsonAsync("/v1/native/recipes/describe", new
        {
            modality = "video",
            model = "wan",
        });

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        JsonElement body = await response.Content.ReadFromJsonAsync<JsonElement>();
        Assert.Equal("video", body.GetProperty("$type").GetString());
        Assert.Equal("wan", body.GetProperty("recipe").GetProperty("familyId").GetString());
        Assert.True(body.GetProperty("defaults").GetProperty("steps").GetInt32() > 0);
        Assert.True(body.GetProperty("constraints").GetProperty("frames").TryGetProperty("step", out _));
    }

    [Fact]
    public async Task Describe_UnregisteredFamilyIs422WithTheTypedIssue()
    {
        using HttpClient client = _factory.CreateClient();
        using HttpResponseMessage response = await client.PostAsJsonAsync("/v1/native/recipes/describe", new
        {
            modality = "video",
            model = "sdxl",
        });

        Assert.Equal(HttpStatusCode.UnprocessableEntity, response.StatusCode);
        JsonElement body = await response.Content.ReadFromJsonAsync<JsonElement>();
        JsonElement issue = body.GetProperty("issues").EnumerateArray()
            .Single(item => item.GetProperty("code").GetString() == "recipe.family.unregistered");
        Assert.Equal("Error", issue.GetProperty("severity").GetString());
    }

    [Fact]
    public async Task Describe_BodyNeverNamesAFilesystemPathOrHash()
    {
        string missing = Path.Combine(Path.GetTempPath(), $"hartsy-recipe-{Guid.NewGuid():N}", "model.safetensors");
        using HttpClient client = _factory.CreateClient();

        using HttpResponseMessage response = await client.PostAsJsonAsync("/v1/native/recipes/describe", new
        {
            modality = "video",
            model = "minimax-h3",
            modelPath = missing,
        });

        string json = await response.Content.ReadAsStringAsync();
        Assert.DoesNotContain(missing, json, StringComparison.Ordinal);
        Assert.DoesNotContain(Path.GetTempPath(), json, StringComparison.Ordinal);
        Assert.DoesNotContain("localPath", json, StringComparison.Ordinal);
        Assert.DoesNotMatch("[0-9a-fA-F]{64}", json);
    }
}
