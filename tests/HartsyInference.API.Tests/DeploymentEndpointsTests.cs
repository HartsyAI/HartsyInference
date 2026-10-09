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
