using System.Buffers.Binary;
using System.Net;
using System.Net.WebSockets;
using System.Reflection;
using System.Text;
using System.Text.Json;
using HartsyInference.Engine;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Hosting.Server;
using Microsoft.AspNetCore.Hosting.Server.Features;
using Microsoft.Extensions.DependencyInjection;

namespace HartsyInference.API.Tests;

/// <summary>The live conversation route over a real WebSocket, against a scripted full-duplex model with 20 ms frames: open, ready, audio up, frames
/// and events down, close.</summary>
public sealed class DuplexRouteTests : IAsyncLifetime
{
    private const int Frame = 480;
    private readonly FakeDuplex _duplex = new();
    private WebApplication? _app;
    private Uri _address = null!;
    private string _modelPath = "";

    public async Task InitializeAsync()
    {
        _modelPath = Path.Combine(Path.GetTempPath(), $"hartsy-duplex-{Guid.NewGuid():N}.gguf");
        await File.WriteAllBytesAsync(_modelPath, [0]);
        WebApplicationBuilder builder = WebApplication.CreateBuilder();
        builder.WebHost.UseUrls("http://127.0.0.1:0");
        builder.Services.AddHartsyInference();
        ISpeechService speech = DispatchProxy.Create<ISpeechService, FakeSpeech>();
        ((FakeSpeech)(object)speech).Owner = _duplex;
        builder.Services.AddSingleton(FakeEngine.Create(speech));
        _app = builder.Build();
        _app.MapHartsyInferenceEndpoints();
        await _app.StartAsync();
        _address = new Uri(_app.Services.GetRequiredService<IServer>().Features.Get<IServerAddressesFeature>()!.Addresses.First());
    }

    public async Task DisposeAsync()
    {
        if (_app is not null) await _app.DisposeAsync();
        File.Delete(_modelPath);
    }

    private async Task<ClientWebSocket> ConnectAsync()
    {
        ClientWebSocket socket = new();
        await socket.ConnectAsync(new Uri($"ws://{_address.Authority}/v1/native/duplex/ws"), new CancellationTokenSource(TimeSpan.FromSeconds(10)).Token);
        return socket;
    }

    private static async Task SendTextAsync(ClientWebSocket socket, object message) =>
        await socket.SendAsync(JsonSerializer.SerializeToUtf8Bytes(message, new JsonSerializerOptions(JsonSerializerDefaults.Web)),
            WebSocketMessageType.Text, true, CancellationToken.None);

    private static async Task<(WebSocketMessageType Type, byte[] Data)> ReceiveAsync(ClientWebSocket socket)
    {
        using CancellationTokenSource cts = new(TimeSpan.FromSeconds(15));
        using MemoryStream message = new();
        byte[] buffer = new byte[8192];
        WebSocketReceiveResult result;
        do
        {
            result = await socket.ReceiveAsync(buffer, cts.Token);
            message.Write(buffer, 0, result.Count);
        }
        while (!result.EndOfMessage);
        return (result.MessageType, message.ToArray());
    }

    private static JsonElement Json(byte[] data) => JsonDocument.Parse(data).RootElement.Clone();

    [Fact]
    public async Task APlainHttpRequestIsToldItNeedsAWebSocket()
    {
        using HttpClient http = new() { BaseAddress = _address };
        HttpResponseMessage response = await http.GetAsync("/v1/native/duplex/ws");
        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
    }

    [Fact]
    public async Task AConversation_OpensSendsAudioReceivesFramesAndTextAndCloses()
    {
        using ClientWebSocket socket = await ConnectAsync();
        await SendTextAsync(socket, new { model = "haru", modelPath = _modelPath, leadFrames = 8, request = new { card = "hello", seed = 3 } });

        (WebSocketMessageType type, byte[] data) = await ReceiveAsync(socket);
        Assert.Equal(WebSocketMessageType.Text, type);
        JsonElement ready = Json(data);
        Assert.Equal("ready", ready.GetProperty("type").GetString());
        Assert.Equal((24000, Frame, 8, "pcm16"), (ready.GetProperty("sampleRate").GetInt32(), ready.GetProperty("frameSamples").GetInt32(),
            ready.GetProperty("leadFrames").GetInt32(), ready.GetProperty("format").GetString()));
        Assert.Equal(("hello", 3), (_duplex.Request!.Card, _duplex.Request.Seed));

        // Two frames of a constant at half scale, sent as one message and a half: the odd byte must be carried.
        byte[] audio = new byte[Frame * 2 * 2];
        for (int i = 0; i < Frame * 2; i++)
        {
            BinaryPrimitives.WriteInt16LittleEndian(audio.AsSpan(i * 2), 16384);
        }
        await socket.SendAsync(audio.AsMemory(0, 301), WebSocketMessageType.Binary, true, CancellationToken.None);
        await socket.SendAsync(audio.AsMemory(301), WebSocketMessageType.Binary, true, CancellationToken.None);

        int binary = 0;
        bool sawText = false;
        while (binary < 6 || !sawText)
        {
            (type, data) = await ReceiveAsync(socket);
            if (type == WebSocketMessageType.Binary)
            {
                Assert.Equal(Frame * 2, data.Length);
                binary++;
            }
            else if (Json(data) is { } message && message.GetProperty("type").GetString() == "text")
            {
                Assert.Equal("three", message.GetProperty("text").GetString());
                sawText = true;
            }
        }
        // The clock does not wait for the client: the audio lands wherever the ticks fall, so count samples rather than frames.
        float[] all = [.. _duplex.HeardSnapshot().SelectMany(f => f)];
        Assert.Equal(Frame * 2, all.Count(v => v == 0.5f));
        Assert.All(all, v => Assert.True(v is 0f or 0.5f, $"unexpected sample {v}"));

        await SendTextAsync(socket, new { type = "close" });
        JsonElement last = default;
        while (true)
        {
            (type, data) = await ReceiveAsync(socket);
            if (type == WebSocketMessageType.Close)
            {
                break;
            }
            if (type == WebSocketMessageType.Text)
            {
                last = Json(data);
            }
        }
        Assert.Equal("stats", last.GetProperty("type").GetString());
        Assert.True(last.GetProperty("final").GetBoolean());
        Assert.True(last.GetProperty("stats").GetProperty("frames").GetInt64() >= 1);
        await Task.Delay(100);
        Assert.True(_duplex.Disposed);
    }

    [Fact]
    public async Task AnUnknownFormat_IsReportedAsAnErrorAndTheSocketCloses()
    {
        using ClientWebSocket socket = await ConnectAsync();
        await SendTextAsync(socket, new { model = "haru", modelPath = _modelPath, format = "mp3" });
        (WebSocketMessageType type, byte[] data) = await ReceiveAsync(socket);
        Assert.Equal(WebSocketMessageType.Text, type);
        JsonElement error = Json(data);
        Assert.Equal("error", error.GetProperty("type").GetString());
        Assert.Contains("pcm16", error.GetProperty("message").GetString());
        (type, _) = await ReceiveAsync(socket);
        Assert.Equal(WebSocketMessageType.Close, type);
    }

    [Fact]
    public async Task Float32AudioIsAccepted_AndSentBack()
    {
        using ClientWebSocket socket = await ConnectAsync();
        await SendTextAsync(socket, new { model = "haru", modelPath = _modelPath, leadFrames = 8, format = "f32" });
        (_, byte[] ready) = await ReceiveAsync(socket);
        Assert.Equal("f32", Json(ready).GetProperty("format").GetString());
        byte[] audio = new byte[Frame * 4];
        for (int i = 0; i < Frame; i++)
        {
            BinaryPrimitives.WriteSingleLittleEndian(audio.AsSpan(i * 4), 0.25f);
        }
        await socket.SendAsync(audio, WebSocketMessageType.Binary, true, CancellationToken.None);
        int frames = 0;
        while (frames < 5)
        {
            (WebSocketMessageType type, byte[] data) = await ReceiveAsync(socket);
            if (type == WebSocketMessageType.Binary)
            {
                Assert.Equal(Frame * 4, data.Length);
                frames++;
            }
        }
        Assert.Equal(Frame, _duplex.HeardSnapshot().SelectMany(f => f).Count(v => v == 0.25f));
    }

    [Theory]
    [InlineData(11025)]
    [InlineData(-16000)]
    public async Task ABadInputRate_IsRefusedBeforeTheModelIsOpened_SoTheNextConversationStillOpens(int rate)
    {
        using (ClientWebSocket bad = await ConnectAsync())
        {
            await SendTextAsync(bad, new { model = "haru", modelPath = _modelPath, inputSampleRate = rate });
            (WebSocketMessageType type, byte[] data) = await ReceiveAsync(bad);
            Assert.Equal(WebSocketMessageType.Text, type);
            Assert.Contains("inputSampleRate", Json(data).GetProperty("message").GetString());
        }
        Assert.Equal(0, _duplex.Opens);

        using ClientWebSocket good = await ConnectAsync();
        await SendTextAsync(good, new { model = "haru", modelPath = _modelPath, inputSampleRate = 16000 });
        (_, byte[] ready) = await ReceiveAsync(good);
        Assert.Equal("ready", Json(ready).GetProperty("type").GetString());
        Assert.Equal(1, _duplex.Opens);
    }

    [Fact]
    public async Task WhenTheModelEndsTheConversation_TheClientGetsTheFinalStatsAndACleanClose()
    {
        _duplex.DoneAfter = 3;
        using ClientWebSocket socket = await ConnectAsync();
        await SendTextAsync(socket, new { model = "haru", modelPath = _modelPath, leadFrames = 8 });
        JsonElement last = default;
        WebSocketReceiveResult? closed = null;
        using CancellationTokenSource cts = new(TimeSpan.FromSeconds(15));
        byte[] buffer = new byte[8192];
        while (closed is null)
        {
            using MemoryStream message = new();
            WebSocketReceiveResult result;
            do
            {
                result = await socket.ReceiveAsync(buffer, cts.Token);
                message.Write(buffer, 0, result.Count);
            }
            while (!result.EndOfMessage);
            if (result.MessageType == WebSocketMessageType.Close)
            {
                closed = result;
            }
            else if (result.MessageType == WebSocketMessageType.Text)
            {
                last = Json(message.ToArray());
            }
        }
        Assert.Equal(WebSocketCloseStatus.NormalClosure, closed.CloseStatus);
        Assert.Equal("stats", last.GetProperty("type").GetString());
        Assert.True(last.GetProperty("final").GetBoolean());
        Assert.Equal(3, last.GetProperty("stats").GetProperty("frames").GetInt64());
        await socket.CloseAsync(WebSocketCloseStatus.NormalClosure, "bye", cts.Token);
        await Task.Delay(100);
        Assert.True(_duplex.Disposed);
    }

    [Fact]
    public async Task WhenTheModelFails_TheClientGetsAnErrorEventAndAServerErrorClose()
    {
        _duplex.ThrowAtStep = 2;
        using ClientWebSocket socket = await ConnectAsync();
        await SendTextAsync(socket, new { model = "haru", modelPath = _modelPath, leadFrames = 8 });
        string? error = null;
        WebSocketReceiveResult? closed = null;
        using CancellationTokenSource cts = new(TimeSpan.FromSeconds(15));
        byte[] buffer = new byte[8192];
        while (closed is null)
        {
            using MemoryStream message = new();
            WebSocketReceiveResult result;
            do
            {
                result = await socket.ReceiveAsync(buffer, cts.Token);
                message.Write(buffer, 0, result.Count);
            }
            while (!result.EndOfMessage);
            if (result.MessageType == WebSocketMessageType.Close)
            {
                closed = result;
            }
            else if (result.MessageType == WebSocketMessageType.Text && Json(message.ToArray()) is { } json && json.GetProperty("type").GetString() == "error")
            {
                error = json.GetProperty("message").GetString();
            }
        }
        Assert.Equal("the model broke", error);
        Assert.Equal(WebSocketCloseStatus.InternalServerError, closed.CloseStatus);
        await socket.CloseAsync(WebSocketCloseStatus.NormalClosure, "bye", cts.Token);
        await Task.Delay(100);
        Assert.True(_duplex.Disposed);
    }

    internal sealed class FakeDuplex
    {
        public DuplexSpeechRequest? Request { get; set; }
        public bool Disposed { get; set; }
        public int Opens { get; set; }
        public bool Open { get; set; }
        public int DoneAfter { get; set; } = int.MaxValue;
        public int ThrowAtStep { get; set; }
        public List<float[]> Heard { get; } = [];

        public float[][] HeardSnapshot()
        {
            lock (Heard)
            {
                return [.. Heard];
            }
        }
    }

    private sealed class FakeSession(FakeDuplex owner) : IDuplexSpeechSession
    {
        private int _steps;
        public int SampleRate => 24000;
        public int FrameSamples => Frame;
        public int RemainingFrames => int.MaxValue;

        public DuplexFrame Start() => new(new float[Frame], "", 0, 0, false);

        public DuplexFrame Step(ReadOnlySpan<float> userFrame)
        {
            lock (owner.Heard)
            {
                owner.Heard.Add(userFrame.ToArray());
            }
            _steps++;
            if (_steps == owner.ThrowAtStep)
            {
                throw new InvalidOperationException("the model broke");
            }
            float[] reply = new float[Frame];
            Array.Fill(reply, 0.1f * _steps);
            return new DuplexFrame(reply, _steps == 3 ? "three" : "", _steps, 0, _steps >= owner.DoneAfter);
        }

        public void Dispose()
        {
            owner.Disposed = true;
            owner.Open = false;
        }
    }

    public class FakeSpeech : DispatchProxy
    {
        internal FakeDuplex Owner { get; set; } = null!;

        protected override object? Invoke(MethodInfo? method, object?[]? args)
        {
            if (method?.Name == "OpenDuplexAsync")
            {
                // The real model refuses a second conversation while one is open.
                if (Owner.Open)
                {
                    throw new InvalidOperationException("A conversation is already open on this model.");
                }
                Owner.Open = true;
                Owner.Opens++;
                Owner.Request = (DuplexSpeechRequest)args![1]!;
                return Task.FromResult<IDuplexSpeechSession>(new FakeSession(Owner));
            }
            throw new NotSupportedException($"The fake speech service does not serve {method?.Name}.");
        }
    }

    public class FakeEngine : DispatchProxy
    {
        private ISpeechService? _speech;

        internal static IInferenceEngine Create(ISpeechService speech)
        {
            IInferenceEngine engine = Create<IInferenceEngine, FakeEngine>();
            ((FakeEngine)(object)engine)._speech = speech;
            return engine;
        }

        protected override object? Invoke(MethodInfo? method, object?[]? args) => method?.Name switch
        {
            "get_Speech" => _speech,
            "Dispose" => null,
            "get_BackendSelector" or "get_BackendDescription" => "fake",
            _ => throw new NotSupportedException($"The fake engine does not serve {method?.Name}."),
        };
    }
}
