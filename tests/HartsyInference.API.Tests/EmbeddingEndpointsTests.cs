using System.Net;
using System.Net.Http.Json;
using System.Text.Json;
using HartsyInference.Engine;
using Microsoft.AspNetCore.Mvc.Testing;
using Xunit;

namespace HartsyInference.API.Tests;

/// <summary>In-process tests for the native <c>/v1/native/embeddings</c> route. Runs on the CPU backend with no
/// model ever loaded for the validation-only tests. One test opportunistically exercises the real
/// "qwen3-embedding" catalog entry end-to-end through the full HTTP stack when its checkpoint happens to already
/// be present under the default models root (downloaded by the correctness pass this route's Engine layer went
/// through) — skips cleanly otherwise, since this test tier doesn't sandbox the models root the way
/// <c>ModelResolverTests</c> does, so a real download can't be relied on to be present for every run.</summary>
public sealed class EmbeddingEndpointsTests : IClassFixture<WebApplicationFactory<Program>>
{
    private readonly WebApplicationFactory<Program> _factory;

    public EmbeddingEndpointsTests(WebApplicationFactory<Program> factory)
    {
        _factory = factory.WithWebHostBuilder(builder => builder.UseSetting("HartsyInference:Backend", "cpu"));
    }

    [Fact]
    public async Task Embeddings_UnresolvableModel_Returns400()
    {
        // EmbeddingService throws HartsyInferenceException for a null LocalPath (mirroring TextService.LoadInto's
        // own "no local path" check) -> 400, same as Image/Text's unresolvable-model behavior, unlike the
        // NotSupportedException -> 501 that Speech/Transcribe's catalog-lookup-first services give.
        using HttpClient client = _factory.CreateClient();
        HttpResponseMessage resp = await client.PostAsJsonAsync("/v1/native/embeddings", new
        {
            model = "not-a-real-embedding-model",
            request = new { input = new[] { "hello" } },
        });
        Assert.Equal(HttpStatusCode.BadRequest, resp.StatusCode);
    }

    [Fact]
    public async Task Embeddings_RealCheckpointIfPresent_ProducesRealVector()
    {
        string expectedPath = Path.Combine(RepoPaths.ModelsRoot(), "Embedding", "qwen3-embedding", "Qwen3-Embedding-0.6B-f16.gguf");
        if (!File.Exists(expectedPath))
            return; // not downloaded in this environment -- skip rather than fail or fabricate a fixture.

        using HttpClient client = _factory.CreateClient();
        HttpResponseMessage resp = await client.PostAsJsonAsync("/v1/native/embeddings", new
        {
            model = "qwen3-embedding",
            request = new { input = new[] { "hello world" } },
        });
        Assert.Equal(HttpStatusCode.OK, resp.StatusCode);
        JsonElement body = await resp.Content.ReadFromJsonAsync<JsonElement>();
        JsonElement vector = body.GetProperty("vectors")[0];
        Assert.Equal(1024, vector.GetArrayLength());
        Assert.Equal(1024, body.GetProperty("dimensions").GetInt32());
        Assert.True(body.GetProperty("totalTokens").GetInt32() > 0);
    }
}
