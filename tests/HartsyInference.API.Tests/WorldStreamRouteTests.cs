using System.Net;
using System.Net.Http.Json;
using HartsyInference.Engine;
using HartsyInference.Engine.Requests;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Hosting.Server;
using Microsoft.AspNetCore.Hosting.Server.Features;
using Microsoft.Extensions.DependencyInjection;

namespace HartsyInference.API.Tests;

/// <summary>The world stream route over HTTP, against scripted sessions registered straight into the registry: a session whose stream does
/// the whole job waits for a long-running slot, an interactive one does not, and a failing or disposed session ends cleanly.</summary>
public sealed class WorldStreamRouteTests : IAsyncLifetime
{
    private WebApplication? _app;
    private HttpClient _http = null!;   // set in InitializeAsync
    private WorldSessionRegistry _registry = null!;
    private InferenceQueue _queue = null!;

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
        _registry = _app.Services.GetRequiredService<WorldSessionRegistry>();
        _queue = _app.Services.GetRequiredKeyedService<InferenceQueue>(QueueKeys.LongRunning);
    }

    public async Task DisposeAsync()
    {
        _http?.Dispose();
        if (_app is not null) await _app.DisposeAsync();
    }

    [Fact]
    public async Task ASessionThatDefersItsDeviceWork_WaitsForALongRunningSlot()
    {
        ScriptedSession session = new(defersDeviceWork: true);
        string id = _registry.Register(session, fps: 16);
        TaskCompletionSource release = new(TaskCreationOptions.RunContinuationsAsynchronously);
        TaskCompletionSource held = new(TaskCreationOptions.RunContinuationsAsynchronously);
        Task slot = _queue.EnqueueAsync(async () => { held.SetResult(); await release.Task; return true; }, CancellationToken.None);
        await held.Task;   // the only long-running slot is now taken

        Task<string> body = _http.GetStringAsync($"/v1/native/world/sessions/{id}/stream");
        await Task.Delay(500);
        Assert.False(session.Started, "The stream started while another job held the long-running slot.");

        release.SetResult();
        await slot;
        string stream = await body;
        Assert.True(session.Started);
        Assert.Contains("event: frame", stream);
    }

    [Fact]
    public async Task AnInteractiveSession_StreamsWithoutTakingASlot()
    {
        ScriptedSession session = new(defersDeviceWork: false);
        string id = _registry.Register(session, fps: 16);
        TaskCompletionSource release = new(TaskCreationOptions.RunContinuationsAsynchronously);
        TaskCompletionSource held = new(TaskCreationOptions.RunContinuationsAsynchronously);
        Task slot = _queue.EnqueueAsync(async () => { held.SetResult(); await release.Task; return true; }, CancellationToken.None);
        await held.Task;

        string stream = await _http.GetStringAsync($"/v1/native/world/sessions/{id}/stream");

        Assert.Contains("event: frame", stream);
        release.SetResult();
        await slot;
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public async Task AFailingStream_EndsWithAnErrorEvent(bool defersDeviceWork)
    {
        ScriptedSession session = new(defersDeviceWork) { Failure = new InvalidOperationException("the clip failed") };
        string id = _registry.Register(session, fps: 16);

        string stream = await _http.GetStringAsync($"/v1/native/world/sessions/{id}/stream");

        Assert.Contains("event: error", stream);
        Assert.Contains("the clip failed", stream);
    }

    [Fact]
    public async Task AnActionOnASessionDisposedMeanwhile_IsNotFound_NotAServerError()
    {
        ScriptedSession session = new(defersDeviceWork: false);
        string id = _registry.Register(session, fps: 16);
        session.Dispose();

        HttpResponseMessage response = await _http.PostAsJsonAsync($"/v1/native/world/sessions/{id}/action", new { action = "w" });

        Assert.Equal(HttpStatusCode.NotFound, response.StatusCode);
    }

    private sealed class ScriptedSession(bool defersDeviceWork) : IWorldSession
    {
        private bool _disposed;

        public bool DefersDeviceWork { get; } = defersDeviceWork;

        public bool Started { get; private set; }

        public Exception? Failure { get; init; }

        public void SendAction(string action) => ObjectDisposedException.ThrowIf(_disposed, this);

        public async IAsyncEnumerable<VideoFrame> StreamAsync([EnumeratorCancellation] CancellationToken cancel)
        {
            Started = true;
            await Task.Yield();
            if (Failure is not null)
                throw Failure;
            yield return new VideoFrame { Index = 0, Width = 2, Height = 2, Rgb = new byte[2 * 2 * 3] };
        }

        public void Dispose() => _disposed = true;
    }
}
