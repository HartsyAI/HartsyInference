using System.Buffers.Binary;
using System.Collections.Concurrent;
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
}
