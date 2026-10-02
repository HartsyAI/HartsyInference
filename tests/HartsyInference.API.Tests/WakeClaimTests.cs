using System.Collections.Concurrent;
using HartsyInference.Audio.Models.Wake;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Engine.Audio.Wake;
using HartsyInference.ModelAssets.Onnx;
using Xunit;

namespace HartsyInference.API.Tests;

/// <summary>Unit tests for <see cref="WakeService.Claim"/>/<see cref="WakeService.Release"/>'s effect on the
/// wake worker's per-device routing decision: scoring versus a claimed sink.
///
/// <para>Drives <see cref="WakeSession"/>/<see cref="WakeWorker"/> directly, the same way
/// <c>WakeTransportTests</c> does, rather than through a full <see cref="WakeService"/> -- nothing in this repo
/// constructs one in a test, since doing so needs an <c>IInferenceEngine</c> that none of this exercises.
/// <see cref="WakeDeviceClaim"/>'s own public constructor exists for exactly this: a test can set
/// <see cref="WakeSession.Claim"/> directly and observe the worker's reaction, which is where the behavior this
/// PR adds actually lives. <see cref="WakeService.Claim"/>/<see cref="WakeService.Release"/> themselves are a
/// thin lookup-plus-CompareExchange over that same field.</para>
///
/// <para>Threshold 0 on the loaded word makes every scoring step a detection, so "no detection fired" is a
/// direct, real signal that scoring did not run -- not an artifact of a wake phrase never being said. Set
/// <c>HARTSYINFERENCE_WAKE_MODELS</c> to the wake model root to run; skips otherwise, same convention as
/// <c>WakeTransportTests</c>.</para></summary>
public sealed class WakeClaimTests
{
    private static string? ModelsDir => Environment.GetEnvironmentVariable("HARTSYINFERENCE_WAKE_MODELS");

    [Fact]
    public async Task Claim_DeliversFramesToTheSinkInOrder_AndSuspendsScoring()
    {
        if (ModelsDir is not { Length: > 0 } models) { Assert.True(true, "set HARTSYINFERENCE_WAKE_MODELS to run"); return; }

        using WakeMelFrontend mel = LoadMel(models);
        using SpeechEmbeddingModel embedding = LoadEmbedding(models);
        using WakeHead head = LoadHead(models, "oww_alexa_v0.1");
        WakeDetectionPipeline pipeline = new(mel, embedding);
        pipeline.AddWord(head, new WakeWordSettings { Threshold = 0f, SmoothingWindow = 1, RefractorySeconds = 0 });
        using WakeSession session = new("sat-claim-1", pipeline) { State = WakeSessionState.Listening };

        ConcurrentDictionary<string, WakeSession> sessions = new() { ["sat-claim-1"] = session };
        List<WakeDetection> detections = [];
        using WakeWorker worker = new(sessions, (_, detection) =>
        {
            lock (detections) detections.Add(detection);
            return Task.CompletedTask;
        });

        List<float[]> received = [];
        session.Claim = new WakeDeviceClaim(samples => { lock (received) received.Add(samples.ToArray()); });

        worker.Start();
        // Two full seconds, well past the pipeline's ~1.3 s mel-frontend warm-up (WakeTransportTests' own
        // "100 frames" convention) -- enough that an unclaimed session would certainly have detected by now at
        // Threshold 0, so "nothing detected" below is a real claim about suspension, not a coincidence of too
        // little audio either way.
        float[] expected = EnqueueFrames(session, frameCount: 100);

        await WaitForAsync(() =>
        {
            lock (received) return received.Sum(f => f.Length) >= expected.Length;
        }, TimeSpan.FromSeconds(10));

        float[] all;
        lock (received) all = [.. received.SelectMany(f => f)];
        Assert.Equal(expected.Length, all.Length);
        // Order preserved, and correctly normalized from the wake path's int16 scale to [-1, 1].
        for (int i = 0; i < expected.Length; i++) Assert.Equal(expected[i] / 32768f, all[i], 0.0001);

        // Threshold 0 means scoring would have produced a detection on essentially any two seconds of audio;
        // none did, and StepsProcessed (only ever incremented on the scoring branch) stayed at zero.
        lock (detections) Assert.Empty(detections);
        Assert.Equal(0, worker.StepsProcessed);
    }

    [Fact]
    public async Task Release_ResumesNormalDetection()
    {
        if (ModelsDir is not { Length: > 0 } models) { Assert.True(true, "set HARTSYINFERENCE_WAKE_MODELS to run"); return; }

        using WakeMelFrontend mel = LoadMel(models);
        using SpeechEmbeddingModel embedding = LoadEmbedding(models);
        using WakeHead head = LoadHead(models, "oww_alexa_v0.1");
        WakeDetectionPipeline pipeline = new(mel, embedding);
        pipeline.AddWord(head, new WakeWordSettings { Threshold = 0f, SmoothingWindow = 1, RefractorySeconds = 0 });
        using WakeSession session = new("sat-claim-2", pipeline) { State = WakeSessionState.Listening };

        ConcurrentDictionary<string, WakeSession> sessions = new() { ["sat-claim-2"] = session };
        List<WakeDetection> detections = [];
        using WakeWorker worker = new(sessions, (_, detection) =>
        {
            lock (detections) detections.Add(detection);
            return Task.CompletedTask;
        });

        List<float[]> received = [];
        WakeDeviceClaim claim = new(samples => { lock (received) received.Add(samples.ToArray()); });
        session.Claim = claim;

        worker.Start();
        session.Enqueue(MakeTone(320, phase: 0), 0);
        await WaitForAsync(() => { lock (received) return received.Count > 0; }, TimeSpan.FromSeconds(10));
        lock (detections) Assert.Empty(detections);

        // The exact CompareExchange-then-reset WakeService.Release performs, exercised directly here for the
        // same reason the claim above is set directly (see the class remarks).
        Assert.Same(claim, Interlocked.CompareExchange(ref session.Claim, null, claim));
        session.RequestReset = true;

        // The reset the worker applies on its next iteration clears the pipeline's own buffered state, so
        // detection needs the same ~1.3 s mel-frontend warm-up all over again -- same 100-frame convention as
        // every other real-pipeline test here, starting the sequence numbering where the claimed phase left off.
        EnqueueFrames(session, frameCount: 100, startSequence: 1);
        await WaitForAsync(() => { lock (detections) return detections.Count > 0; }, TimeSpan.FromSeconds(10));
        lock (detections) Assert.NotEmpty(detections);
    }

    [Fact]
    public async Task TwoDevices_OneClaimedOneNot_EachBehavesIndependently()
    {
        if (ModelsDir is not { Length: > 0 } models) { Assert.True(true, "set HARTSYINFERENCE_WAKE_MODELS to run"); return; }

        using WakeMelFrontend mel = LoadMel(models);
        using SpeechEmbeddingModel embedding = LoadEmbedding(models);
        using WakeHead head = LoadHead(models, "oww_alexa_v0.1");

        WakeSession MakeSession(string deviceId)
        {
            WakeDetectionPipeline pipeline = new(mel, embedding);
            pipeline.AddWord(head, new WakeWordSettings { Threshold = 0f, SmoothingWindow = 1, RefractorySeconds = 0 });
            return new WakeSession(deviceId, pipeline) { State = WakeSessionState.Listening };
        }

        using WakeSession claimed = MakeSession("sat-claimed");
        using WakeSession unclaimed = MakeSession("sat-unclaimed");
        ConcurrentDictionary<string, WakeSession> sessions = new()
        {
            [claimed.DeviceId] = claimed,
            [unclaimed.DeviceId] = unclaimed,
        };

        ConcurrentDictionary<string, List<WakeDetection>> detectionsByDevice = new()
        {
            [claimed.DeviceId] = [],
            [unclaimed.DeviceId] = [],
        };
        using WakeWorker worker = new(sessions, (session, detection) =>
        {
            List<WakeDetection> list = detectionsByDevice[session.DeviceId];
            lock (list) list.Add(detection);
            return Task.CompletedTask;
        });

        List<float[]> claimedReceived = [];
        claimed.Claim = new WakeDeviceClaim(samples => { lock (claimedReceived) claimedReceived.Add(samples.ToArray()); });

        worker.Start();
        // 100 frames (2 s) on both, the same real-pipeline warm-up margin every other test here uses, so the
        // unclaimed device detecting is a real result and not a lucky race against too little audio.
        EnqueueFrames(claimed, frameCount: 100);
        EnqueueFrames(unclaimed, frameCount: 100);

        List<WakeDetection> unclaimedDetections = detectionsByDevice[unclaimed.DeviceId];
        await WaitForAsync(() =>
        {
            lock (claimedReceived)
            lock (unclaimedDetections)
                return claimedReceived.Count > 0 && unclaimedDetections.Count > 0;
        }, TimeSpan.FromSeconds(10));

        Assert.NotEmpty(claimedReceived);
        List<WakeDetection> claimedDetections = detectionsByDevice[claimed.DeviceId];
        lock (claimedDetections) Assert.Empty(claimedDetections);
        lock (unclaimedDetections) Assert.NotEmpty(unclaimedDetections);
    }

    [Fact]
    public async Task Disconnect_WhileClaimed_NotifiesTheHost_AndLeavesNothingClaimed()
    {
        if (ModelsDir is not { Length: > 0 } models) { Assert.True(true, "set HARTSYINFERENCE_WAKE_MODELS to run"); return; }

        using WakeMelFrontend mel = LoadMel(models);
        using SpeechEmbeddingModel embedding = LoadEmbedding(models);
        using WakeDetectionPipeline pipeline = new(mel, embedding);
        using WakeSession session = new("sat-disconnect", pipeline);

        ConcurrentDictionary<string, WakeSession> sessions = new();
        WakeSession Factory(string deviceId) => session;
        WakeServiceOptions options = new() { Port = 0, PingInterval = TimeSpan.FromSeconds(30) };
        using WakeListener listener = new(sessions, Factory, options);
        listener.Start();

        TaskCompletionSource disconnected = new(TaskCreationOptions.RunContinuationsAsynchronously);
        WakeDeviceClaim claim = new(_ => { }, onDisconnected: () => disconnected.TrySetResult());

        using System.Net.Sockets.TcpClient client = new();
        await client.ConnectAsync("127.0.0.1", listener.Port);
        using System.Net.Sockets.NetworkStream stream = client.GetStream();
        await WriteHeaderAsync(stream, "{\"type\":\"hello\",\"data\":{\"device_id\":\"sat-disconnect\",\"rate\":16000,\"width\":2,\"channels\":1}}");
        await WaitForAsync(() => session.Codec is not null, TimeSpan.FromSeconds(10));

        session.Claim = claim;
        client.Close();

        await disconnected.Task.WaitAsync(TimeSpan.FromSeconds(10));
        Assert.Null(session.Claim);
    }

    private static float[] MakeTone(int samples, int phase)
    {
        float[] result = new float[samples];
        for (int i = 0; i < samples; i++) result[i] = (float)(3000 * Math.Sin((phase + i) * 0.05));
        return result;
    }

    /// <summary>Enqueues <paramref name="frameCount"/> 20 ms (320-sample) frames of a simple tone and returns
    /// the concatenated expected samples, for an order/content check on whatever received them.</summary>
    private static float[] EnqueueFrames(WakeSession session, int frameCount, long startSequence = 0)
    {
        float[] all = new float[frameCount * 320];
        for (int f = 0; f < frameCount; f++)
        {
            float[] frame = MakeTone(320, phase: f * 320);
            frame.CopyTo(all, f * 320);
            session.Enqueue(frame, startSequence + f);
        }
        return all;
    }

    private static async Task WriteHeaderAsync(System.Net.Sockets.NetworkStream stream, string json)
    {
        byte[] bytes = System.Text.Encoding.UTF8.GetBytes(json + "\n");
        await stream.WriteAsync(bytes);
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
