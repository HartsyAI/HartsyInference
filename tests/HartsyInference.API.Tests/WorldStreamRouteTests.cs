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

    [Fact]
    public async Task ASessionThatDefersItsDeviceWork_IsRefusedWith429_WhenTheQueueIsFull()
    {
        ScriptedSession session = new(defersDeviceWork: true);
        string id = _registry.Register(session, fps: 16);
        TaskCompletionSource release = new(TaskCreationOptions.RunContinuationsAsynchronously);
        TaskCompletionSource held = new(TaskCreationOptions.RunContinuationsAsynchronously);
        List<Task> jobs = [_queue.EnqueueAsync(async () => { held.SetResult(); await release.Task; return true; }, CancellationToken.None)];
        await held.Task;
        // One running job plus the configured depth of waiting ones fills the queue.
        for (int i = 0; i < 4; i++)
            jobs.Add(_queue.EnqueueAsync(async () => { await release.Task; return true; }, CancellationToken.None));

        HttpResponseMessage response = await _http.GetAsync($"/v1/native/world/sessions/{id}/stream");

        Assert.Equal(HttpStatusCode.TooManyRequests, response.StatusCode);
        Assert.False(session.Started);
        release.SetResult();
        await Task.WhenAll(jobs);
    }

    [Fact]
    public async Task AFrameIsNotPulledFromTheSession_UntilThePreviousOneIsWritten()
    {
        // The session yields frames as fast as it is asked; the client reads none until it is told to. The response write of a frame
        // must finish before the next is pulled, so the session can run only a bounded number of frames ahead of the socket buffers.
        ScriptedSession session = new(defersDeviceWork: true) { FrameCount = 300, FrameSize = 256 };
        string id = _registry.Register(session, fps: 16);

        using HttpResponseMessage response = await _http.GetAsync($"/v1/native/world/sessions/{id}/stream", HttpCompletionOption.ResponseHeadersRead);
        await Task.Delay(1500);   // the client reads nothing meanwhile

        Assert.True(session.Pulled < 100, $"The session was pulled {session.Pulled} frames ahead of a client that read none.");
        string body = await response.Content.ReadAsStringAsync();
        Assert.Equal(300, session.Pulled);
        Assert.Contains("event: frame", body);
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
        private int _pulled;

        public bool DefersDeviceWork { get; } = defersDeviceWork;

        public bool Started { get; private set; }

        public Exception? Failure { get; init; }

        public int FrameCount { get; init; } = 1;

        /// <summary>Frames are FrameSize x FrameSize noise, which a PNG cannot shrink, so each is a real quarter-megabyte write.</summary>
        public int FrameSize { get; init; } = 2;

        public int Pulled => Volatile.Read(ref _pulled);

        public void SendAction(string action) => ObjectDisposedException.ThrowIf(_disposed, this);

        public async IAsyncEnumerable<VideoFrame> StreamAsync([EnumeratorCancellation] CancellationToken cancel)
        {
            Started = true;
            await Task.Yield();
            if (Failure is not null)
                throw Failure;
            for (int i = 0; i < FrameCount; i++)
            {
                Interlocked.Increment(ref _pulled);
                byte[] rgb = new byte[FrameSize * FrameSize * 3];
                Random.Shared.NextBytes(rgb);
                yield return new VideoFrame { Index = i, Width = FrameSize, Height = FrameSize, Rgb = rgb };
            }
        }

        public void Dispose() => _disposed = true;
    }
}
