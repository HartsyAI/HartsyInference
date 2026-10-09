using System.Net.Http.Json;
using System.Text.Json;
using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;
using Xunit;

namespace HartsyInference.API.Tests;

/// <summary>The status codes and bodies of the deployment admin routes, against a text service whose answers the test scripts. DELETE unloads by deployment: it never
/// frees whatever the device holds now.</summary>
public sealed class DeploymentEndpointsTests : IClassFixture<WebApplicationFactory<Program>>
{
    private readonly WebApplicationFactory<Program> _factory;

    public DeploymentEndpointsTests(WebApplicationFactory<Program> factory)
    {
        _factory = factory.WithWebHostBuilder(builder => builder.UseSetting("HartsyInference:Backend", "cpu"));
    }

    private WebApplicationFactory<Program> WithText(ScriptedDeploymentsText text) =>
        _factory.WithWebHostBuilder(builder => builder.ConfigureServices(services =>
        {
            services.RemoveAll<IInferenceEngine>();
            services.AddSingleton<IInferenceEngine>(new TextOnlyEngine(text));
        }));

    [Theory]
    [InlineData(DeploymentUnloadOutcome.Unloaded, 200)]
    [InlineData(DeploymentUnloadOutcome.AlreadyGone, 200)]
    [InlineData(DeploymentUnloadOutcome.TimedOut, 409)]
    [InlineData(DeploymentUnloadOutcome.NotFound, 404)]
    public async Task Delete_Answers_Each_Unload_Outcome_With_Its_Status(DeploymentUnloadOutcome outcome, int status)
    {
        ScriptedDeploymentsText text = new() { UnloadOutcome = outcome };
        using WebApplicationFactory<Program> app = WithText(text);
        using HttpClient client = app.CreateClient();

        using HttpResponseMessage response = await client.DeleteAsync("/admin/deployments/a");

        Assert.Equal(status, (int)response.StatusCode);
        Assert.Equal(["a"], text.Unloaded);
        Assert.Equal(0, text.DeviceUnloads);
        JsonElement body = await response.Content.ReadFromJsonAsync<JsonElement>();
        if (status == 200)
        {
            Assert.Equal(outcome == DeploymentUnloadOutcome.Unloaded, body.GetProperty("unloaded").GetBoolean());
            bool explained = body.TryGetProperty("reason", out JsonElement reason) && reason.ValueKind == JsonValueKind.String;
            Assert.Equal(outcome == DeploymentUnloadOutcome.AlreadyGone, explained);
        }
        else
        {
            Assert.False(string.IsNullOrEmpty(body.GetProperty("error").GetProperty("message").GetString()));
        }
    }

    [Fact]
    public async Task Get_Lists_The_Deployments_With_Lowercase_States()
    {
        ScriptedDeploymentsText text = new();
        text.Listed.Add(new DeploymentStatus("chat", "llm-a", "cpu", DeploymentState.Ready, null));
        text.Listed.Add(new DeploymentStatus("old", "llm-b", "cpu", DeploymentState.Failed, "no checkpoint"));
        using WebApplicationFactory<Program> app = WithText(text);
        using HttpClient client = app.CreateClient();

        using HttpResponseMessage response = await client.GetAsync("/admin/deployments");

        Assert.Equal(200, (int)response.StatusCode);
        JsonElement[] listed = [.. (await response.Content.ReadFromJsonAsync<JsonElement>()).GetProperty("deployments").EnumerateArray()];
        Assert.Equal(["chat", "old"], listed.Select(d => d.GetProperty("deployment_id").GetString()));
        Assert.Equal(["ready", "failed"], listed.Select(d => d.GetProperty("state").GetString()));
        Assert.Equal("no checkpoint", listed[1].GetProperty("problem").GetString());
    }

    [Fact]
    public async Task Post_Answers_400_Without_Its_Fields_And_200_With_The_Status_The_Load_Ended_In()
    {
        ScriptedDeploymentsText text = new()
        {
            Deploy = request => new DeploymentStatus(request.DeploymentId, request.Model.Requested, request.Device, DeploymentState.Failed, "the load failed"),
        };
        using WebApplicationFactory<Program> app = WithText(text);
        using HttpClient client = app.CreateClient();

        using HttpResponseMessage missing = await client.PostAsJsonAsync("/admin/deployments", new { model = "llm-a" });
        Assert.Equal(400, (int)missing.StatusCode);

        // A failed load is reported in the status, not as an error.
        using HttpResponseMessage failed = await client.PostAsJsonAsync("/admin/deployments", new { deployment_id = "chat", model = "llm-a", device = "cpu" });
        Assert.Equal(200, (int)failed.StatusCode);
        JsonElement body = await failed.Content.ReadFromJsonAsync<JsonElement>();
        Assert.Equal("failed", body.GetProperty("state").GetString());
        Assert.Equal("the load failed", body.GetProperty("problem").GetString());
    }

    [Theory]
    [InlineData("argument", 400)]
    [InlineData("engine", 400)]
    [InlineData("unsupported", 501)]
    public async Task Post_Maps_A_Thrown_Error_Through_GenerationErrors(string error, int status)
    {
        ScriptedDeploymentsText text = new()
        {
            Deploy = _ => throw (error switch
            {
                "argument" => new ArgumentException("bad deployment request"),
                "engine" => new HartsyInference.Core.Exceptions.HartsyInferenceException("bad model"),
                _ => (Exception)new NotSupportedException("no deployments here"),
            }),
        };
        using WebApplicationFactory<Program> app = WithText(text);
        using HttpClient client = app.CreateClient();

        using HttpResponseMessage response = await client.PostAsJsonAsync("/admin/deployments", new { deployment_id = "chat", model = "llm-a" });

        Assert.Equal(status, (int)response.StatusCode);
    }

    [Fact]
    public async Task Capacity_Reports_Each_Deployment_In_Pages()
    {
        ScriptedDeploymentsText text = new();
        text.Listed.Add(new DeploymentStatus("chat", "llm-a", "cuda:0", DeploymentState.Ready, null));
        text.Listed.Add(new DeploymentStatus("host", "dsv41", "cpu", DeploymentState.Ready, null));
        text.Capacities["chat"] = new DeploymentCapacity(DeploymentState.Ready, Active: 2, Queued: 1, MaxConcurrent: 64, KvPagesFree: 10, KvPagesTotal: 32);
        text.Capacities["host"] = new DeploymentCapacity(DeploymentState.Ready, Active: 0, Queued: 0, MaxConcurrent: 4, KvPagesFree: null, KvPagesTotal: null);
        using WebApplicationFactory<Program> app = WithText(text);
        using HttpClient client = app.CreateClient();

        using HttpResponseMessage response = await client.GetAsync("/admin/capacity");

        Assert.Equal(200, (int)response.StatusCode);
        JsonElement[] rows = [.. (await response.Content.ReadFromJsonAsync<JsonElement>()).GetProperty("deployments").EnumerateArray()];
        Assert.Equal(2, rows.Length);
        Assert.Equal("chat", rows[0].GetProperty("deployment_id").GetString());
        Assert.Equal(2, rows[0].GetProperty("active").GetInt32());
        Assert.Equal(1, rows[0].GetProperty("queued").GetInt32());
        Assert.Equal(64, rows[0].GetProperty("max_concurrent").GetInt32());
        Assert.Equal(10, rows[0].GetProperty("kv_pages_free").GetInt32());
        Assert.Equal(32, rows[0].GetProperty("kv_pages_total").GetInt32());
        Assert.Equal(JsonValueKind.Null, rows[1].GetProperty("kv_pages_total").ValueKind);
    }

    /// <summary>What the real engine answers for a model that resolves to no checkpoint: <c>ModelResolver.Resolve</c> does not throw, and <c>DeployAsync</c> records the
    /// load failure in the deployment, so the route answers 200 with state <c>failed</c> rather than an error status.</summary>
    [Fact]
    public async Task Post_Of_An_Unresolvable_Model_On_The_Real_Engine_Is_200_With_State_Failed()
    {
        using HttpClient client = _factory.CreateClient();

        using HttpResponseMessage response = await client.PostAsJsonAsync("/admin/deployments",
            new { deployment_id = "missing", model = "hartsy-no-such-model-0f3a" });

        Assert.Equal(200, (int)response.StatusCode);
        JsonElement body = await response.Content.ReadFromJsonAsync<JsonElement>();
        Assert.Equal("failed", body.GetProperty("state").GetString());
        Assert.Contains("No checkpoint found", body.GetProperty("problem").GetString());
    }

    /// <summary>A text service that answers the deployment calls from a script and records what the routes asked of it.</summary>
    private sealed class ScriptedDeploymentsText : ITextService
    {
        public List<DeploymentStatus> Listed { get; } = [];
        public Dictionary<string, DeploymentCapacity> Capacities { get; } = [];
        public DeploymentUnloadOutcome UnloadOutcome { get; init; } = DeploymentUnloadOutcome.Unloaded;
        public Func<DeploymentRequest, DeploymentStatus> Deploy { get; init; } = request =>
            new DeploymentStatus(request.DeploymentId, request.Model.Requested, request.Device, DeploymentState.Ready, null);

        public List<string> Unloaded { get; } = [];
        public int DeviceUnloads { get; private set; }

        public IReadOnlyList<DeploymentStatus> Deployments => Listed;
        public DeploymentCapacity? Capacity(string deploymentId) => Capacities.GetValueOrDefault(deploymentId);
        public Task<DeploymentStatus> DeployAsync(DeploymentRequest request, CancellationToken cancel = default) => Task.FromResult(Deploy(request));

        public DeploymentUnloadOutcome UnloadDeployment(string deploymentId)
        {
            Unloaded.Add(deploymentId);
            return UnloadOutcome;
        }

        public bool Unload(string? device = null)
        {
            DeviceUnloads++;
            return true;
        }

        public Task<TextResult> GenerateAsync(ModelSpec spec, TextRequest request, CancellationToken cancel = default) => throw new NotSupportedException();
        public IAsyncEnumerable<TextChunk> StreamAsync(ModelSpec spec, TextRequest request, CancellationToken cancel = default) => throw new NotSupportedException();
        public int CountTokens(ModelSpec spec, string text) => 0;
    }

    /// <summary>An engine whose only working service is the scripted text service.</summary>
    private sealed class TextOnlyEngine(ITextService text) : IInferenceEngine
    {
        public string BackendSelector => "fake";
        public string BackendDescription => "fake text backend";
        public IReadOnlyCollection<string> LoadedPipelineKeys => [];
        public ITextService Text => text;
        public IImagesService Images => throw new NotSupportedException();
        public IVideoService Video => throw new NotSupportedException();
        public IMusicService Music => throw new NotSupportedException();
        public ISpeechService Speech => throw new NotSupportedException();
        public ITranscribeService Transcribe => throw new NotSupportedException();
        public IVoiceConversionService VoiceConversion => throw new NotSupportedException();
        public IFxService Fx => throw new NotSupportedException();
        public IModelPrefetchService ModelPrefetch => throw new NotSupportedException();
        public IVisionService Vision => throw new NotSupportedException();
        public IRestoreService Restore => throw new NotSupportedException();
        public IMeshService Mesh => throw new NotSupportedException();
        public IWorldService World => throw new NotSupportedException();
        public IEmbeddingService Embeddings => throw new NotSupportedException();
        public IDecisionService Decisions => throw new NotSupportedException();

        public bool IsSupported(Modality modality) => modality == Modality.Text;
        public void SetBackend(string selector) => throw new NotSupportedException();
        public void FreeMemory() { }
        public void Dispose() { }
    }
}
