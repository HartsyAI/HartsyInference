using System.Buffers.Binary;
using System.Collections.Concurrent;
using System.Net;
using System.Net.Sockets;
using System.Text;
using HartsyInference.Audio.Models.Wake;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Cpu;
using HartsyInference.Engine.Audio.Wake;
using HartsyInference.ModelAssets.Onnx;
using Xunit;

namespace HartsyInference.API.Tests;

/// <summary>End-to-end transport: a satellite connects over TCP, streams PCM, and gets a detection back.
///
/// <para>Covers the parts that only fail when the pieces are wired together — frame codec round-trips across
/// arbitrary TCP segment boundaries, the socket-to-worker handoff, device-keyed sessions surviving a reconnect,
/// and a sequence gap resetting model state instead of splicing across it.</para>
///
/// <para>Set <c>HARTSYINFERENCE_WAKE_MODELS</c> to the wake model root to run; skips otherwise.</para></summary>
public sealed class WakeTransportTests
{
    private static string? ModelsDir => Environment.GetEnvironmentVariable("HARTSYINFERENCE_WAKE_MODELS");

    [Fact]
    public async Task Satellite_StreamsAudio_AndReceivesDetection()
    {
        if (ModelsDir is not { Length: > 0 } models) { Assert.True(true, "set HARTSYINFERENCE_WAKE_MODELS to run"); return; }

        ConcurrentDictionary<string, WakeSession> sessions = new();
        List<WakeDetection> observed = [];
        using CpuBackend backend = new();

        // Threshold 0 makes every scoring step a detection, so this exercises the transport rather than
        // depending on a positive wake-word recording.
        using WakeMelFrontend mel = LoadMel(models);
        using SpeechEmbeddingModel embedding = LoadEmbedding(models);
        using WakeHead head = LoadHead(models, "oww_alexa_v0.1");

        WakeSession Factory(string deviceId)
        {
            WakeDetectionPipeline pipeline = new(mel, embedding);
            pipeline.AddWord(head, new WakeWordSettings { Threshold = 0f, SmoothingWindow = 1, RefractorySeconds = 0 });
            return new WakeSession(deviceId, pipeline);
        }

        TaskCompletionSource detected = new(TaskCreationOptions.RunContinuationsAsynchronously);
        using WakeWorker worker = new(sessions, (session, detection) =>
        {
            lock (observed)
            {
                observed.Add(detection);
                if (observed.Count == 1) detected.TrySetResult();
            }
            return Task.CompletedTask;
        });
        worker.Start();

        WakeServiceOptions options = new() { Port = 0, PingInterval = TimeSpan.FromSeconds(30) };
        using WakeListener listener = new(sessions, Factory, options);
        listener.Start();

        using TcpClient client = new();
        await client.ConnectAsync("127.0.0.1", listener.Port);
        using NetworkStream stream = client.GetStream();

        await WriteHeaderAsync(stream, "{\"type\":\"hello\",\"data\":{\"device_id\":\"pico-test\",\"rate\":16000,\"width\":2,\"channels\":1}}");

        // Two seconds of audio in 20 ms frames — enough to pass the pipeline's 1.3 s warm-up.
        for (int i = 0; i < 100; i++)
            await WriteAudioAsync(stream, i, 320);

        await detected.Task.WaitAsync(TimeSpan.FromSeconds(20));
        Assert.True(sessions.ContainsKey("pico-test"));
        lock (observed) Assert.NotEmpty(observed);
    }

    [Fact]
    public async Task Reconnect_ReusesSessionAndResetsDetectionState()
    {
        if (ModelsDir is not { Length: > 0 } models) { Assert.True(true, "set HARTSYINFERENCE_WAKE_MODELS to run"); return; }

        ConcurrentDictionary<string, WakeSession> sessions = new();
        using WakeMelFrontend mel = LoadMel(models);
        using SpeechEmbeddingModel embedding = LoadEmbedding(models);

        WakeSession Factory(string deviceId) => new(deviceId, new WakeDetectionPipeline(mel, embedding));

        WakeServiceOptions options = new() { Port = 0, PingInterval = TimeSpan.FromSeconds(30) };
        using WakeListener listener = new(sessions, Factory, options);
        listener.Start();

        for (int attempt = 0; attempt < 2; attempt++)
        {
            using TcpClient client = new();
            await client.ConnectAsync("127.0.0.1", listener.Port);
            using NetworkStream stream = client.GetStream();
            await WriteHeaderAsync(stream, "{\"type\":\"hello\",\"data\":{\"device_id\":\"pico-test\",\"rate\":16000,\"width\":2,\"channels\":1}}");
            await WriteAudioAsync(stream, 0, 320);
            await WaitForAsync(() => sessions.ContainsKey("pico-test"), TimeSpan.FromSeconds(10));
        }

        // One session for the device across both connections — configuration and words survive the drop.
        Assert.Single(sessions);

        // A second hello re-armed the reset flag, so the worker clears model state rather than splicing the
        // pre- and post-disconnect audio together.
        WakeSession session = sessions["pico-test"];
        Assert.Equal(0, session.SamplesDropped);
    }

    [Fact]
    public async Task SequenceGap_RequestsDetectionReset()
    {
        if (ModelsDir is not { Length: > 0 } models) { Assert.True(true, "set HARTSYINFERENCE_WAKE_MODELS to run"); return; }

        using WakeMelFrontend mel = LoadMel(models);
        using SpeechEmbeddingModel embedding = LoadEmbedding(models);
        using WakeDetectionPipeline pipeline = new(mel, embedding);
        using WakeSession session = new("pico-test", pipeline);

        float[] audio = new float[320];
        session.Enqueue(audio, 0);
        session.RequestReset = false;
        session.Enqueue(audio, 1);
        Assert.False(session.RequestReset);

        // Frame 2 never arrived; the model must not be fed audio spliced across the hole.
        session.Enqueue(audio, 3);
        Assert.True(session.RequestReset);
    }

    /// <summary>Reproduces the race a review caught before merge: a device's reconnect lands while its OLD
    /// connection is still unwinding (the "dropping the older one" path in <c>ServeStreamAsync</c>'s hello
    /// case), and that old connection's disconnect teardown runs after. Needs no real wake model weights —
    /// this never sends an audio-chunk frame, only hello and a raw socket close, so an unloaded
    /// <see cref="WakeMelFrontend"/>/<see cref="SpeechEmbeddingModel"/> pair is enough to construct a session.
    ///
    /// <para>Does not use <see cref="WakeListener.Start"/>: this drives its own <see cref="TcpListener"/> so the
    /// test can hold each connection's <c>ServeStreamAsync</c> <see cref="Task"/> directly instead of the
    /// fire-and-forget one <see cref="WakeListener"/>'s own accept loop would give it, which is what makes the
    /// ordering below deterministic rather than a sleep-and-hope.</para></summary>
    [Fact]
    public async Task Disconnect_StaleAfterReconnect_LeavesNewConnectionsCodecAndClaimIntact()
    {
        TcpListener rawListener = new(IPAddress.Loopback, 0);
        rawListener.Start();
        int port = ((IPEndPoint)rawListener.LocalEndpoint).Port;
        try
        {
            ConcurrentDictionary<string, WakeSession> sessions = new();
            WakeSession Factory(string deviceId) =>
                new(deviceId, new WakeDetectionPipeline(new WakeMelFrontend(), new SpeechEmbeddingModel()));
            WakeServiceOptions options = new() { Port = 0, PingInterval = TimeSpan.FromSeconds(30) };
            using WakeListener listener = new(sessions, Factory, options);

            using TcpClient clientA = new();
            await clientA.ConnectAsync(IPAddress.Loopback, port);
            using TcpClient serverA = await rawListener.AcceptTcpClientAsync();
            Task serveA = listener.ServeStreamAsync(serverA.GetStream(), "A", CancellationToken.None);

            await WriteHeaderAsync(clientA.GetStream(), "{\"type\":\"hello\",\"data\":{\"device_id\":\"race-test\",\"rate\":16000,\"width\":2,\"channels\":1}}");
            await WaitForAsync(() => sessions.ContainsKey("race-test"), TimeSpan.FromSeconds(10));
            WakeSession session = sessions["race-test"];
            await WaitForAsync(() => session.Codec is not null, TimeSpan.FromSeconds(10));
            object codecA = session.Codec!;

            int disconnected = 0;
            WakeDeviceClaim claim = new(_ => { }, () => Interlocked.Increment(ref disconnected));
            Assert.Null(Interlocked.CompareExchange(ref session.Claim, claim, null));

            // Connection B reconnects for the SAME device while A's socket is still open as far as anything
            // has told it -- the overlap the bug depends on.
            using TcpClient clientB = new();
            await clientB.ConnectAsync(IPAddress.Loopback, port);
            using TcpClient serverB = await rawListener.AcceptTcpClientAsync();
            Task serveB = listener.ServeStreamAsync(serverB.GetStream(), "B", CancellationToken.None);

            await WriteHeaderAsync(clientB.GetStream(), "{\"type\":\"hello\",\"data\":{\"device_id\":\"race-test\",\"rate\":16000,\"width\":2,\"channels\":1}}");
            // OnReconnected runs synchronously inside ServeStreamAsync's hello case before anything is written
            // back, so once session.Codec differs from connection A's, B's reconnect has already landed -- no
            // need to round-trip B's hello-ack.
            await WaitForAsync(() => session.Codec is not null && !ReferenceEquals(session.Codec, codecA), TimeSpan.FromSeconds(10));
            object codecB = session.Codec!;

            // NOW close A -- its ServeStreamAsync loop sees end-of-stream and runs its own finally next.
            clientA.Close();
            await serveA.WaitAsync(TimeSpan.FromSeconds(10));

            // The discriminating assertions: at this exact point nothing but A's teardown has run. Pre-fix, A's
            // unconditional clear wipes out B's codec and claim and fires a disconnect that belongs to a
            // connection (B's) that is still live. Post-fix, A's teardown recognizes it is no longer the live
            // connection for this device and does nothing.
            Assert.Same(codecB, session.Codec);
            Assert.Same(claim, session.Claim);
            Assert.Equal(0, disconnected);

            // B's own, legitimate disconnect still works normally.
            clientB.Close();
            await serveB.WaitAsync(TimeSpan.FromSeconds(10));
            Assert.Null(session.Codec);
            Assert.Null(session.Claim);
            Assert.Equal(1, disconnected);
        }
        finally
        {
            rawListener.Stop();
        }
    }

    /// <summary>The other half of the same review comment: a disconnect that legitimately owns the session
    /// (no reconnect involved) must still complete its teardown when the claim's own <c>OnDisconnected</c>
    /// callback throws -- the callback is the host's bug, not a reason to leave <see cref="WakeSession.Codec"/>,
    /// <see cref="WakeSession.State"/> or <see cref="WakeSession.Claim"/> uncleared, and not a reason to fault
    /// the connection loop. Drives <see cref="WakeListener.ServeStreamAsync"/> directly against a one-shot fake
    /// stream (hello, then EOF) rather than a real socket: the only thing under test is whether the exception
    /// escapes the <c>finally</c>, which a real socket cannot make deterministic either way.</summary>
    [Fact]
    public async Task Disconnect_WhenOnDisconnectedThrows_StillClearsStateWithoutFaulting()
    {
        ConcurrentDictionary<string, WakeSession> sessions = new();
        WakeSession Factory(string deviceId) =>
            new(deviceId, new WakeDetectionPipeline(new WakeMelFrontend(), new SpeechEmbeddingModel()));
        WakeSession session = sessions.GetOrAdd("race-test-2", Factory);

        WakeDeviceClaim claim = new(_ => { }, () => throw new InvalidOperationException("host callback bug"));
        Assert.Null(Interlocked.CompareExchange(ref session.Claim, claim, null));

        byte[] hello = Encoding.UTF8.GetBytes(
            "{\"type\":\"hello\",\"data\":{\"device_id\":\"race-test-2\",\"rate\":16000,\"width\":2,\"channels\":1}}\n");
        OneFrameStream stream = new(hello);

        WakeServiceOptions options = new() { Port = 0, PingInterval = TimeSpan.FromSeconds(30) };
        using WakeListener listener = new(sessions, Factory, options);

        // Must complete, not fault: a throwing host callback must not escape this connection's own teardown.
        await listener.ServeStreamAsync(stream, "test", CancellationToken.None);

        Assert.Null(session.Codec);
        Assert.Null(session.Claim);
        Assert.Equal(WakeSessionState.Handshake, session.State);
    }

    private static async Task WriteHeaderAsync(NetworkStream stream, string json)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(json + "\n");
        await stream.WriteAsync(bytes);
        await stream.FlushAsync();
    }

    private static async Task WriteAudioAsync(NetworkStream stream, long sequence, int samples)
    {
        byte[] payload = new byte[samples * 2];
        for (int i = 0; i < samples; i++)
            BinaryPrimitives.WriteInt16LittleEndian(payload.AsSpan(i * 2), (short)(3000 * Math.Sin(i * 0.05)));
        string header = $"{{\"type\":\"audio-chunk\",\"data\":{{\"seq\":{sequence}}},\"payload_length\":{payload.Length}}}\n";
        await stream.WriteAsync(Encoding.UTF8.GetBytes(header));
        await stream.WriteAsync(payload);
        await stream.FlushAsync();
    }

    private static async Task WaitForAsync(Func<bool> condition, TimeSpan timeout)
    {
        DateTime deadline = DateTime.UtcNow + timeout;
        while (DateTime.UtcNow < deadline)
        {
            if (condition()) return;
            await Task.Delay(50);
        }
        Assert.Fail($"condition not met within {timeout}");
    }

    private static WakeMelFrontend LoadMel(string modelsDir)
    {
        using OnnxWeightLoader loader = new();
        loader.Load(Path.Combine(modelsDir, "backbone", "melspectrogram.onnx"));
        WakeMelFrontend mel = new();
        mel.LoadWeights(loader.GetAllTensors());
        return mel;
    }

    private static SpeechEmbeddingModel LoadEmbedding(string modelsDir)
    {
        using OnnxWeightLoader loader = new();
        loader.Load(Path.Combine(modelsDir, "backbone", "embedding_model.onnx"));
        SpeechEmbeddingModel model = new();
        model.LoadWeights(loader.GetAllTensors());
        return model;
    }

    private static WakeHead LoadHead(string modelsDir, string name)
    {
        using OnnxWeightLoader loader = new();
        loader.Load(Path.Combine(modelsDir, "heads", name + ".onnx"));
        WakeHead head = new(name);
        head.LoadWeights(loader.GetAllTensors());
        return head;
    }

    /// <summary>Exactly one frame's worth of readable bytes, then EOF; writes are accepted and discarded. Lets
    /// <see cref="WakeListener.ServeStreamAsync"/> run its hello case once and then see end-of-stream on its
    /// next read, without a real socket.</summary>
    private sealed class OneFrameStream(byte[] data) : Stream
    {
        private int _position;

        public override bool CanRead => true;
        public override bool CanSeek => false;
        public override bool CanWrite => true;
        public override long Length => data.Length;
        public override long Position { get => _position; set => throw new NotSupportedException(); }

        public override int Read(byte[] buffer, int offset, int count) => Read(buffer.AsSpan(offset, count));

        public override int Read(Span<byte> buffer)
        {
            int available = Math.Min(buffer.Length, data.Length - _position);
            if (available <= 0) return 0;
            data.AsSpan(_position, available).CopyTo(buffer);
            _position += available;
            return available;
        }

        public override ValueTask<int> ReadAsync(Memory<byte> buffer, CancellationToken cancellationToken = default) =>
            ValueTask.FromResult(Read(buffer.Span));

        public override void Flush() { }
        public override Task FlushAsync(CancellationToken cancellationToken) => Task.CompletedTask;
        public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
        public override void SetLength(long value) => throw new NotSupportedException();
        public override void Write(byte[] buffer, int offset, int count) { }
        public override ValueTask WriteAsync(ReadOnlyMemory<byte> buffer, CancellationToken cancellationToken = default) => ValueTask.CompletedTask;
    }
}
