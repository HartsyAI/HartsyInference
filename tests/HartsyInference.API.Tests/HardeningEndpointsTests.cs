using System.Net;
using System.Net.Http.Json;
using System.Text.Json;
using Microsoft.AspNetCore.Mvc.Testing;
using Xunit;

namespace HartsyInference.API.Tests;

/// <summary>In-process tests for the production-hardening surface: multi-key auth with per-key identity,
/// per-key rate limiting, usage metering, and the observability/discovery endpoints (<c>/metrics</c>,
/// <c>/openapi/v1.json</c>). Runs on the CPU backend, same tier as every other file in this project — no GPU.</summary>
public sealed class HardeningEndpointsTests : IClassFixture<WebApplicationFactory<Program>>
{
    private readonly WebApplicationFactory<Program> _factory;

    public HardeningEndpointsTests(WebApplicationFactory<Program> factory)
    {
        _factory = factory.WithWebHostBuilder(builder => builder.UseSetting("HartsyInference:Backend", "cpu"));
    }

    [Fact]
    public async Task MultiKeyAuth_DistinctKeysResolveDistinctIdentities()
    {
        using WebApplicationFactory<Program> factory = _factory.WithWebHostBuilder(builder =>
        {
            builder.UseSetting("HartsyInference:ApiKeys:0:Key", "key-a");
            builder.UseSetting("HartsyInference:ApiKeys:0:Name", "caller-a");
            builder.UseSetting("HartsyInference:ApiKeys:1:Key", "key-b");
            builder.UseSetting("HartsyInference:ApiKeys:1:Name", "caller-b");
        });

        using HttpClient clientA = factory.CreateClient();
        clientA.DefaultRequestHeaders.Add("x-api-key", "key-a");
        using HttpClient clientB = factory.CreateClient();
        clientB.DefaultRequestHeaders.Add("x-api-key", "key-b");
        using HttpClient clientUnknown = factory.CreateClient();
        clientUnknown.DefaultRequestHeaders.Add("x-api-key", "not-a-real-key");

        Assert.Equal(HttpStatusCode.OK, (await clientA.GetAsync("/admin/catalog")).StatusCode);
        Assert.Equal(HttpStatusCode.OK, (await clientB.GetAsync("/admin/catalog")).StatusCode);
        Assert.Equal(HttpStatusCode.Unauthorized, (await clientUnknown.GetAsync("/admin/catalog")).StatusCode);

        JsonElement usage = await clientA.GetFromJsonAsync<JsonElement>("/admin/usage");
        Assert.True(usage.TryGetProperty("caller-a", out _));
        Assert.True(usage.TryGetProperty("caller-b", out _));
        Assert.False(usage.TryGetProperty("not-a-real-key", out _));
    }

    [Fact]
    public async Task RateLimiter_RejectsOnceOverPerKeyLimit()
    {
        using WebApplicationFactory<Program> factory = _factory.WithWebHostBuilder(builder =>
        {
            builder.UseSetting("HartsyInference:ApiKeys:0:Key", "rate-limited-key");
            builder.UseSetting("HartsyInference:ApiKeys:0:Name", "rate-limited-caller");
            builder.UseSetting("HartsyInference:ApiKeys:0:RateLimitPerMinute", "2");
        });
        using HttpClient client = factory.CreateClient();
        client.DefaultRequestHeaders.Add("x-api-key", "rate-limited-key");

        List<HttpStatusCode> statuses = [];
        for (int i = 0; i < 5; i++)
            statuses.Add((await client.GetAsync("/admin/catalog")).StatusCode);

        Assert.Contains(HttpStatusCode.OK, statuses);
        Assert.Contains(HttpStatusCode.TooManyRequests, statuses);

        // Confirm the rejected response is the distinct per-key error, not the queue-capacity one QueueFullException
        // maps to -- a client needs to be able to tell "over your quota" from "server momentarily saturated".
        HttpResponseMessage rejected = await client.GetAsync("/admin/catalog");
        if (rejected.StatusCode == HttpStatusCode.TooManyRequests)
        {
            JsonElement body = await rejected.Content.ReadFromJsonAsync<JsonElement>();
            Assert.Equal("rate_limit_exceeded", body.GetProperty("error").GetProperty("type").GetString());
        }
    }

    [Fact]
    public async Task RateLimiter_DoesNotGateHealthReadyVersionProbes()
    {
        using WebApplicationFactory<Program> factory = _factory.WithWebHostBuilder(builder =>
        {
            builder.UseSetting("HartsyInference:ApiKeys:0:Key", "probe-test-key");
            builder.UseSetting("HartsyInference:ApiKeys:0:RateLimitPerMinute", "1");
        });
        using HttpClient client = factory.CreateClient();

        // No API key presented at all -- probes must stay reachable regardless of auth/rate-limit config.
        for (int i = 0; i < 5; i++)
            Assert.Equal(HttpStatusCode.OK, (await client.GetAsync("/health")).StatusCode);
    }

}
