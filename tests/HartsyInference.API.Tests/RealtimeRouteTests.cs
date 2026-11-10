using System.Net;
using System.Net.Http.Json;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Threading.Channels;
using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Hosting.Server;
using Microsoft.AspNetCore.Hosting.Server.Features;
using Microsoft.Extensions.DependencyInjection;

namespace HartsyInference.API.Tests;

/// <summary>The live-session routes end to end over HTTP, against a scripted engine: open, push frames and prompts, read the SSE stream,
/// read stats, close.</summary>
public sealed class RealtimeRouteTests : IAsyncLifetime
{
    private readonly FakeText _text = new();
    private WebApplication? _app;
    private HttpClient _http = null!;   // set in InitializeAsync
    private string _modelPath = "";

    public async Task InitializeAsync()
    {
        _modelPath = Path.Combine(Path.GetTempPath(), $"hartsy-moss-{Guid.NewGuid():N}.gguf");
        await File.WriteAllBytesAsync(_modelPath, [0]);
        WebApplicationBuilder builder = WebApplication.CreateBuilder();
        builder.WebHost.UseUrls("http://127.0.0.1:0");
        builder.Services.AddHartsyInference();
        builder.Services.AddSingleton(FakeEngine.Create(_text));   // replaces the real engine
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
        File.Delete(_modelPath);
    }

    [Fact]
    public async Task ASessionTakesFramesAndPromptsAndStreamsItsEvents()
    {
        HttpResponseMessage opened = await _http!.PostAsJsonAsync("/v1/native/realtime/sessions",
            new { model = "moss-vl-realtime", modelPath = _modelPath, request = new { initialPrompt = "watch", frameQueueSize = 8 } });
        Assert.Equal(HttpStatusCode.OK, opened.StatusCode);
        string id = (await opened.Content.ReadFromJsonAsync<JsonElement>()).GetProperty("sessionId").GetString()!;
        FakeSession session = _text.Sessions.Single();
        Assert.Equal(("watch", 8), (session.Request.InitialPrompt, session.Request.FrameQueueSize));

        string png = Convert.ToBase64String(PngEncoder.Encode(new byte[2 * 2 * 3], 2, 2));
        HttpResponseMessage frame = await _http.PostAsJsonAsync($"/v1/native/realtime/sessions/{id}/frame", new { image = png, timestamp = 1.5 });
        Assert.Equal(HttpStatusCode.OK, frame.StatusCode);
        Assert.Equal((2, 2, 1.5), (session.Frames[0].Image.Width, session.Frames[0].Image.Height, session.Frames[0].Timestamp));

        HttpResponseMessage backwards = await _http.PostAsJsonAsync($"/v1/native/realtime/sessions/{id}/frame",
            new { image = "data:image/png;base64," + png, timestamp = 1.0 });
        Assert.Equal(HttpStatusCode.BadRequest, backwards.StatusCode);

        Assert.Equal(HttpStatusCode.OK, (await _http.PostAsJsonAsync($"/v1/native/realtime/sessions/{id}/prompt", new { prompt = "what now?" })).StatusCode);
        Assert.Equal(["what now?"], session.Prompts);

        session.Emit(new RealtimeVlmChunk(RealtimeVlmChunkKind.RoundStart, "what now?"));
        session.Emit(new RealtimeVlmChunk(RealtimeVlmChunkKind.Control, "<|response|>"));
        session.Emit(new RealtimeVlmChunk(RealtimeVlmChunkKind.Text, "A red disc."));
        session.Emit(new RealtimeVlmChunk(RealtimeVlmChunkKind.Silence, "<|silence|>"));
        session.Emit(new RealtimeVlmChunk(RealtimeVlmChunkKind.FramesDropped, Count: 2));
        session.End();
        string stream = await _http.GetStringAsync($"/v1/native/realtime/sessions/{id}/stream");
        string[] names = [.. stream.Split('\n').Where(l => l.StartsWith("event: ", StringComparison.Ordinal)).Select(l => l[7..])];
        Assert.Equal(["round_start", "control", "text", "silence", "frames_dropped"], names);
        Assert.Contains("\"text\":\"A red disc.\"", stream);
        Assert.Contains("\"count\":2", stream);

        JsonElement stats = await _http.GetFromJsonAsync<JsonElement>($"/v1/native/realtime/sessions/{id}");
        Assert.Equal(7, stats.GetProperty("generatedTokens").GetInt64());

        Assert.Equal(HttpStatusCode.OK, (await _http.DeleteAsync($"/v1/native/realtime/sessions/{id}")).StatusCode);
        Assert.True(session.Disposed);
        Assert.Equal(HttpStatusCode.NotFound, (await _http.PostAsJsonAsync($"/v1/native/realtime/sessions/{id}/prompt", new { prompt = "late" })).StatusCode);
        Assert.Equal(HttpStatusCode.NotFound, (await _http.DeleteAsync($"/v1/native/realtime/sessions/{id}")).StatusCode);
    }

    private async Task<string> OpenSession()
    {
        HttpResponseMessage opened = await _http.PostAsJsonAsync("/v1/native/realtime/sessions", new { model = "moss-vl-realtime", modelPath = _modelPath });
        Assert.Equal(HttpStatusCode.OK, opened.StatusCode);
        return (await opened.Content.ReadFromJsonAsync<JsonElement>()).GetProperty("sessionId").GetString()!;
    }

    [Fact]
    public async Task ASecondStreamOnTheSameSessionIsRefused()
    {
        string id = await OpenSession();
        FakeSession session = _text.Sessions.Single();
        session.Emit(new RealtimeVlmChunk(RealtimeVlmChunkKind.Text, "hello"));

        using HttpResponseMessage first = await _http.GetAsync($"/v1/native/realtime/sessions/{id}/stream", HttpCompletionOption.ResponseHeadersRead);
        Assert.Equal(HttpStatusCode.OK, first.StatusCode);   // headers go out with the first event, so the first stream is registered
        HttpResponseMessage second = await _http.GetAsync($"/v1/native/realtime/sessions/{id}/stream");
        Assert.Equal(HttpStatusCode.Conflict, second.StatusCode);

        session.End();   // the first stream drains and finishes
        string body = await first.Content.ReadAsStringAsync();
        Assert.Contains("event: text", body);
    }

    [Fact]
    public async Task AnErrorEventClosesTheSessionAndFreesItsSlot()
    {
        string id = await OpenSession();
        FakeSession session = _text.Sessions.Single();
        session.Emit(new RealtimeVlmChunk(RealtimeVlmChunkKind.Error, "the vision cap was reached"));

        string stream = await _http.GetStringAsync($"/v1/native/realtime/sessions/{id}/stream");
        Assert.Contains("event: error", stream);
        Assert.True(session.Disposed);
        Assert.Equal(HttpStatusCode.NotFound, (await _http.PostAsJsonAsync($"/v1/native/realtime/sessions/{id}/prompt", new { prompt = "more" })).StatusCode);
    }

    [Fact]
    public async Task ALargeChatBodyIsNotCutOffByKestrelsDefaultLimit()
    {
        // 35 MB is past Kestrel's 30,000,000-byte default; a video_url part is allowed far more (the option's 256 MiB).
        string content = new('A', 35_000_000);
        using StringContent body = new($$"""{"model":"qwen3","messages":[{"role":"user","content":"{{content}}"}]}""", System.Text.Encoding.UTF8, "application/json");
        HttpResponseMessage response = await _http.PostAsync("/v1/chat/completions", body);
        Assert.NotEqual(HttpStatusCode.RequestEntityTooLarge, response.StatusCode);
    }

    [Fact]
    public async Task AModelThatCannotStreamIsRefusedWithAProblem()
    {
        _text.Refuse = true;
        HttpResponseMessage opened = await _http!.PostAsJsonAsync("/v1/native/realtime/sessions", new { model = "qwen3", modelPath = _modelPath });
        Assert.False(opened.IsSuccessStatusCode);
        Assert.Contains("streaming vision-language", await opened.Content.ReadAsStringAsync());
    }

    [Fact]
    public async Task AnUnknownSessionIsNotFound()
    {
        Assert.Equal(HttpStatusCode.NotFound, (await _http!.GetAsync("/v1/native/realtime/sessions/nope/stream")).StatusCode);
        Assert.Equal(HttpStatusCode.NotFound, (await _http.GetAsync("/v1/native/realtime/sessions/nope")).StatusCode);
    }

    private sealed class FakeSession(RealtimeVlmRequest request) : IRealtimeVlmSession
    {
        private readonly Channel<RealtimeVlmChunk> _events = Channel.CreateUnbounded<RealtimeVlmChunk>();
        private double _last = double.NegativeInfinity;

        public RealtimeVlmRequest Request { get; } = request;
        public List<(Engine.Requests.ImageData Image, double Timestamp)> Frames { get; } = [];
        public List<string> Prompts { get; } = [];
        public bool Disposed { get; private set; }
        public RealtimeVlmSessionStats Stats => new(10, 20, 7, Frames.Count, 0);

        public bool PushFrame(Engine.Requests.ImageData frame, double timestampSeconds)
        {
            ObjectDisposedException.ThrowIf(Disposed, this);
            if (timestampSeconds < _last) throw new ArgumentException("Frame timestamps must not go backwards.");
            _last = timestampSeconds;
            Frames.Add((frame, timestampSeconds));
            return false;
        }

        public void PushPrompt(string prompt)
        {
            ObjectDisposedException.ThrowIf(Disposed, this);
            Prompts.Add(prompt);
        }

        public void Emit(RealtimeVlmChunk chunk) => _events.Writer.TryWrite(chunk);

        public void End() => _events.Writer.TryComplete();

        public async IAsyncEnumerable<RealtimeVlmChunk> StreamAsync([EnumeratorCancellation] CancellationToken cancel = default)
        {
            await foreach (RealtimeVlmChunk chunk in _events.Reader.ReadAllAsync(cancel)) yield return chunk;
        }

        public ValueTask DisposeAsync()
        {
            Disposed = true;
            _events.Writer.TryComplete();
            return ValueTask.CompletedTask;
        }
    }

    private sealed class FakeText : ITextService
    {
        public List<FakeSession> Sessions { get; } = [];
        public bool Refuse { get; set; }

        public Task<IRealtimeVlmSession> OpenRealtimeAsync(ModelSpec spec, RealtimeVlmRequest request, CancellationToken cancel = default)
        {
            if (Refuse)
                throw new NotSupportedException($"Live vision sessions are served for MOSS-VL-Realtime; '{spec.Requested}' is not a streaming vision-language model.");
            FakeSession session = new(request);
            Sessions.Add(session);
            return Task.FromResult<IRealtimeVlmSession>(session);
        }

        public Task<TextResult> GenerateAsync(ModelSpec spec, TextRequest request, CancellationToken cancel = default) => throw new NotSupportedException();

        public IAsyncEnumerable<TextChunk> StreamAsync(ModelSpec spec, TextRequest request, CancellationToken cancel = default) => throw new NotSupportedException();

        public int CountTokens(ModelSpec spec, string text) => 0;

        public bool Unload(string? device = null) => false;
    }

    /// <summary>An engine whose text service is <see cref="FakeText"/>; every other member refuses.</summary>
    public class FakeEngine : DispatchProxy
    {
        private ITextService? _text;

        internal static IInferenceEngine Create(ITextService text)
        {
            IInferenceEngine engine = Create<IInferenceEngine, FakeEngine>();
            ((FakeEngine)(object)engine)._text = text;
            return engine;
        }

        protected override object? Invoke(MethodInfo? method, object?[]? args) => method?.Name switch
        {
            "get_Text" => _text,
            "Dispose" => null,
            "get_BackendSelector" or "get_BackendDescription" => "fake",
            _ => throw new NotSupportedException($"The fake engine does not serve {method?.Name}."),
        };
    }
}
