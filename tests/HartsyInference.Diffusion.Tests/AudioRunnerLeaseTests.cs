using HartsyInference.Audio.Io;
using HartsyInference.Audio.Streaming;
using HartsyInference.Core.Backends;
using HartsyInference.Engine;
using HartsyInference.Engine.Audio;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Runner leases on the public speech services, driven through a CPU engine whose caches are seeded with fake
/// runners under the exact keys the services resolve, so nothing loads or downloads. Covers the pin lifetime, the
/// memory-pressure sweep, sharing with the service calls, revocation by every engine release path, waiting for a call
/// in flight, and idempotent Dispose. Every test relaxes the host floor first: a box genuinely short of RAM must not
/// evict a fake before its lease pins it, which would send the open to the real loader.</summary>
public sealed class AudioRunnerLeaseTests
{
    private const string KokoroKey = "hexgrad/Kokoro-82M";
    private const string WhisperKey = "openai/whisper-tiny";
    private const string PiperKey = "rhasspy/piper-voices|en_US-lessac-medium";

    private static readonly ModelSpec Kokoro = new() { Requested = "kokoro", Modality = Modality.Speech };
    private static readonly ModelSpec WhisperTiny = new() { Requested = "whisper:openai/whisper-tiny", Modality = Modality.Transcribe };
    private static readonly SpeechRequest TtsOptions = new() { Text = "" };
    private static readonly AudioRequest SttOptions = new() { Audio = new AudioClip { Data = [] } };

    [Fact]
    public async Task SynthesizerLease_PinsTheServiceRunner_AndDisposeReleasesIt()
    {
        using IDisposable calm = AudioEvictionPressure.Relax();
        using InferenceEngine engine = new("cpu");
        FakeTts tts = await SeedTtsAsync(engine, KokoroKey);

        ISynthesizerLease lease = await engine.Speech.OpenSynthesizerAsync(Kokoro);
        Assert.True(engine.AudioRuntime.Tts.IsPinned(KokoroKey));
        Assert.Equal(tts.SampleRate, lease.SampleRate);

        float[] samples = lease.Synthesize("Hello there.", new SpeechRequest { Text = "ignored", Voice = "af_bella", Speed = 1.25 });
        Assert.Equal(tts.Output, samples);
        Assert.Equal("Hello there.", tts.LastJob!.Text);
        Assert.Equal("af_bella", tts.LastJob.Voice);
        Assert.Equal(1.25, tts.LastJob.Speed);

        lease.Dispose();
        Assert.False(engine.AudioRuntime.Tts.IsPinned(KokoroKey));
        Assert.True(engine.AudioRuntime.Tts.IsResident(KokoroKey));
        Assert.Throws<ObjectDisposedException>(() => lease.Synthesize("Again.", TtsOptions));
        Assert.Equal(1, tts.Calls);
    }

    [Fact]
    public async Task TranscriberLease_PassesPcmStraightThrough_AndResamplesOnlyAForeignRate()
    {
        using IDisposable calm = AudioEvictionPressure.Relax();
        using InferenceEngine engine = new("cpu");
        FakeStt stt = await SeedSttAsync(engine, WhisperKey);

        using ITranscriberLease lease = await engine.Transcribe.OpenTranscriberAsync(WhisperTiny);
        Assert.True(engine.AudioRuntime.Stt.IsPinned(WhisperKey));

        float[] native = Ramp(1600);
        string heard = lease.Transcribe(native, 16_000, new AudioRequest { Audio = new AudioClip { Data = [] }, Language = "de", Translate = true });
        Assert.Equal("heard", heard);
        Assert.Equal(native, stt.LastAudio);
        Assert.NotSame(native, stt.LastAudio);
        Assert.Equal("de", stt.LastRequest!.Language);
        Assert.True(stt.LastRequest.Translate);

        float[] narrowband = Ramp(800);
        lease.Transcribe(narrowband, 8_000, SttOptions);
        Assert.Equal(Resampler.Create(8_000, 16_000).Resample(narrowband), stt.LastAudio);
    }

    [Fact]
    public async Task TranscriberLease_RefusesTimedOrEmptyRequests_WithoutTouchingTheRunner()
    {
        using IDisposable calm = AudioEvictionPressure.Relax();
        using InferenceEngine engine = new("cpu");
        FakeStt stt = await SeedSttAsync(engine, WhisperKey);
        using ITranscriberLease lease = await engine.Transcribe.OpenTranscriberAsync(WhisperTiny);
        float[] pcm = Ramp(160);

        Assert.Throws<ArgumentException>(() => lease.Transcribe(pcm, 16_000, SttOptions with { WordTimestamps = true }));
        Assert.Throws<ArgumentException>(() => lease.Transcribe(pcm, 16_000, SttOptions with { Diarization = true }));
        Assert.Throws<ArgumentException>(() => lease.Transcribe(ReadOnlySpan<float>.Empty, 16_000, SttOptions));
        Assert.Throws<ArgumentOutOfRangeException>(() => lease.Transcribe(pcm, 0, SttOptions));
        Assert.Equal(0, stt.Calls);
    }

    [Fact]
    public async Task SynthesizerLease_OnVoiceSelectedWeights_RefusesAnotherVoice()
    {
        using IDisposable calm = AudioEvictionPressure.Relax();
        using InferenceEngine engine = new("cpu");
        FakeTts piper = await SeedTtsAsync(engine, PiperKey, sampleRate: 22_050);

        using ISynthesizerLease lease = await engine.Speech.OpenSynthesizerAsync(
            new ModelSpec { Requested = "piper:en_US-lessac-medium", Modality = Modality.Speech });
        Assert.True(engine.AudioRuntime.Tts.IsPinned(PiperKey));
        lease.Synthesize("Hi.", TtsOptions);
        lease.Synthesize("Hi.", TtsOptions with { Voice = "default" });
        lease.Synthesize("Hi.", TtsOptions with { Voice = "en_US-lessac-medium" });

        Assert.Throws<ArgumentException>(() => lease.Synthesize("Hi.", TtsOptions with { Voice = "en_US-ryan-medium" }));
        Assert.Throws<ArgumentException>(() => lease.Synthesize("  ", TtsOptions));
        Assert.Equal(3, piper.Calls);
    }

    [Fact]
    public async Task PinnedLeaseRunners_SurviveTenForcedPressureSwitches()
    {
        using IDisposable calm = AudioEvictionPressure.Relax();
        using InferenceEngine engine = new("cpu");
        AudioRuntime runtime = engine.AudioRuntime;
        FakeTts tts = await SeedTtsAsync(engine, KokoroKey);
        FakeStt stt = await SeedSttAsync(engine, WhisperKey);
        FakeTts bystander = await SeedTtsAsync(engine, "bystander");
        using ISynthesizerLease synth = await engine.Speech.OpenSynthesizerAsync(Kokoro);
        using ITranscriberLease hear = await engine.Transcribe.OpenTranscriberAsync(WhisperTiny);

        using (AudioEvictionPressure.Force())
        {
            for (int turn = 0; turn < 10; turn++)
            {
                TranscriptResult transcript = await engine.Transcribe.RunAsync(WhisperTiny, Clip16k());
                Assert.Equal("heard", transcript.Text);
                Assert.Equal("heard", hear.Transcribe(Ramp(1600), 16_000, SttOptions));
                AudioResult spoken = await engine.Speech.SynthesizeAsync(Kokoro, new SpeechRequest { Text = "Next turn." });
                Assert.Equal(tts.SampleRate, spoken.SampleRate);
                Assert.Equal(tts.Output, synth.Synthesize("Next turn.", TtsOptions));
            }
        }

        Assert.False(tts.Disposed);
        Assert.False(stt.Disposed);
        Assert.True(runtime.Tts.IsResident(KokoroKey));
        Assert.True(runtime.Stt.IsResident(WhisperKey));
        Assert.Equal(20, tts.Calls);
        Assert.Equal(20, stt.Calls);
        if (AudioEvictionPressure.HostPressureObservable)
        {
            Assert.True(bystander.Disposed, "the sweep should have evicted the unpinned runner.");
            Assert.False(runtime.Tts.IsResident("bystander"));
        }
    }

    [Fact]
    public async Task ServiceCalls_RunOnTheLeasedRunner_WhileTheLeaseIsOpen()
    {
        using IDisposable calm = AudioEvictionPressure.Relax();
        using InferenceEngine engine = new("cpu");
        FakeTts tts = await SeedTtsAsync(engine, KokoroKey);
        FakeStt stt = await SeedSttAsync(engine, WhisperKey);
        using ISynthesizerLease synth = await engine.Speech.OpenSynthesizerAsync(Kokoro);
        using ITranscriberLease hear = await engine.Transcribe.OpenTranscriberAsync(WhisperTiny);

        AudioResult spoken = await engine.Speech.SynthesizeAsync(Kokoro, new SpeechRequest { Text = "From the service." });
        Assert.Equal(AudioClipCodec.EncodeWav(tts.Output, null, tts.SampleRate), spoken.Data);
        List<AudioChunk> streamed = [];
        await foreach (AudioChunk chunk in engine.Speech.SynthesizeStreamAsync(Kokoro, new SpeechRequest { Text = "Streamed." }))
        {
            streamed.Add(chunk);
        }
        Assert.Equal(tts.Output, Assert.Single(streamed).Samples);
        TranscriptResult transcript = await engine.Transcribe.RunAsync(WhisperTiny, Clip16k());
        Assert.Equal("heard", transcript.Text);

        Assert.Equal(tts.Output, synth.Synthesize("From the lease.", TtsOptions));
        Assert.Equal("heard", hear.Transcribe(Ramp(1600), 16_000, SttOptions));
        Assert.Equal(3, tts.Calls);
        Assert.Equal(2, stt.Calls);
        Assert.True(engine.AudioRuntime.Tts.IsPinned(KokoroKey));
        Assert.True(engine.AudioRuntime.Stt.IsPinned(WhisperKey));
    }

    [Fact]
    public async Task LeaseCalls_DoNotWaitForTheGenerationLock()
    {
        using IDisposable calm = AudioEvictionPressure.Relax();
        using InferenceEngine engine = new("cpu");
        FakeTts busy = await SeedTtsAsync(engine, KokoroKey, blocking: true);
        await SeedSttAsync(engine, WhisperKey);
        using ITranscriberLease hear = await engine.Transcribe.OpenTranscriberAsync(WhisperTiny);

        // Off the test thread: with a resident runner nothing in the service path yields before the synthesis.
        Task<AudioResult> generation = Task.Run(() => engine.Speech.SynthesizeAsync(Kokoro, new SpeechRequest { Text = "A long generation." }));
        Assert.True(busy.CallEntered.Wait(TimeSpan.FromSeconds(10)), "the service generation never started.");
        string heard = await Task.Run(() => hear.Transcribe(Ramp(1600), 16_000, SttOptions)).WaitAsync(TimeSpan.FromSeconds(10));
        Assert.Equal("heard", heard);
        Assert.False(generation.IsCompleted);

        busy.CallMayFinish.Set();
        await generation.WaitAsync(TimeSpan.FromSeconds(10));
    }

    [Theory]
    [InlineData("Dispose")]
    [InlineData("FreeMemory")]
    [InlineData("SetBackend")]
    public async Task EngineRelease_RevokesOpenLeases(string release)
    {
        using IDisposable calm = AudioEvictionPressure.Relax();
        using InferenceEngine engine = new("cpu");
        FakeTts tts = await SeedTtsAsync(engine, KokoroKey);
        FakeStt stt = await SeedSttAsync(engine, WhisperKey);
        ISynthesizerLease synth = await engine.Speech.OpenSynthesizerAsync(Kokoro);
        ITranscriberLease hear = await engine.Transcribe.OpenTranscriberAsync(WhisperTiny);

        switch (release)
        {
            case "Dispose":
                engine.Dispose();
                break;
            case "FreeMemory":
                engine.FreeMemory();
                break;
            default:
                engine.SetBackend("cpu");
                break;
        }

        Assert.True(tts.Disposed);
        Assert.True(stt.Disposed);
        Assert.False(engine.AudioRuntime.Tts.IsPinned(KokoroKey));
        Assert.False(engine.AudioRuntime.Stt.IsPinned(WhisperKey));
        ObjectDisposedException revoked = Assert.Throws<ObjectDisposedException>(() => synth.Synthesize("After.", TtsOptions));
        Assert.Contains("released its audio models", revoked.Message, StringComparison.Ordinal);
        Assert.Throws<ObjectDisposedException>(() => hear.Transcribe(Ramp(160), 16_000, SttOptions));
        synth.Dispose();
        hear.Dispose();
        Assert.Equal(0, tts.Calls);
        Assert.Equal(0, stt.Calls);
    }

    [Fact]
    public async Task EngineDispose_WaitsForTheLeaseCallInFlight()
    {
        using IDisposable calm = AudioEvictionPressure.Relax();
        InferenceEngine engine = new("cpu");
        FakeTts tts = await SeedTtsAsync(engine, KokoroKey, blocking: true);
        ISynthesizerLease lease = await engine.Speech.OpenSynthesizerAsync(Kokoro);

        Task<float[]> call = Task.Run(() => lease.Synthesize("A sentence in flight.", TtsOptions));
        Assert.True(tts.CallEntered.Wait(TimeSpan.FromSeconds(10)), "the lease call never started.");
        Task teardown = Task.Run(engine.Dispose);
        Assert.NotSame(teardown, await Task.WhenAny(teardown, Task.Delay(300)));
        Assert.False(tts.Disposed, "the engine disposed the runner under a running call.");

        tts.CallMayFinish.Set();
        Assert.Equal(tts.Output, await call.WaitAsync(TimeSpan.FromSeconds(10)));
        await teardown.WaitAsync(TimeSpan.FromSeconds(10));
        Assert.True(tts.Disposed);
        Assert.Throws<ObjectDisposedException>(() => lease.Synthesize("After.", TtsOptions));
    }

    [Fact]
    public async Task LeaseDispose_WaitsForItsOwnCallInFlight()
    {
        using IDisposable calm = AudioEvictionPressure.Relax();
        using InferenceEngine engine = new("cpu");
        FakeTts tts = await SeedTtsAsync(engine, KokoroKey, blocking: true);
        ISynthesizerLease lease = await engine.Speech.OpenSynthesizerAsync(Kokoro);

        Task<float[]> call = Task.Run(() => lease.Synthesize("A sentence in flight.", TtsOptions));
        Assert.True(tts.CallEntered.Wait(TimeSpan.FromSeconds(10)), "the lease call never started.");
        Task disposing = Task.Run(lease.Dispose);
        Assert.NotSame(disposing, await Task.WhenAny(disposing, Task.Delay(300)));
        Assert.True(engine.AudioRuntime.Tts.IsPinned(KokoroKey), "Dispose released the pin under a running call.");

        tts.CallMayFinish.Set();
        Assert.Equal(tts.Output, await call.WaitAsync(TimeSpan.FromSeconds(10)));
        await disposing.WaitAsync(TimeSpan.FromSeconds(10));
        Assert.False(engine.AudioRuntime.Tts.IsPinned(KokoroKey));
    }

    [Fact]
    public async Task ConcurrentDisposes_AndAnEngineDispose_ReleaseEachHoldOnce()
    {
        using IDisposable calm = AudioEvictionPressure.Relax();
        InferenceEngine engine = new("cpu");
        FakeTts tts = await SeedTtsAsync(engine, KokoroKey);
        ISynthesizerLease first = await engine.Speech.OpenSynthesizerAsync(Kokoro);
        ISynthesizerLease second = await engine.Speech.OpenSynthesizerAsync(Kokoro);

        Parallel.For(0, 16, _ => first.Dispose());
        Assert.True(engine.AudioRuntime.Tts.IsPinned(KokoroKey), "one lease's disposes released the other's hold.");

        Parallel.Invoke(second.Dispose, engine.Dispose, second.Dispose);
        Assert.False(engine.AudioRuntime.Tts.IsPinned(KokoroKey));
        Assert.True(tts.Disposed);
        Assert.Throws<ObjectDisposedException>(() => second.Synthesize("After.", TtsOptions));
    }

    [Fact]
    public async Task DoubleDispose_ReleasesOneHold()
    {
        using IDisposable calm = AudioEvictionPressure.Relax();
        using InferenceEngine engine = new("cpu");
        await SeedSttAsync(engine, WhisperKey);
        ITranscriberLease first = await engine.Transcribe.OpenTranscriberAsync(WhisperTiny);
        ITranscriberLease second = await engine.Transcribe.OpenTranscriberAsync(WhisperTiny);

        first.Dispose();
        first.Dispose();
        Assert.True(engine.AudioRuntime.Stt.IsPinned(WhisperKey));
        Assert.Equal("heard", second.Transcribe(Ramp(160), 16_000, SttOptions));

        second.Dispose();
        second.Dispose();
        Assert.False(engine.AudioRuntime.Stt.IsPinned(WhisperKey));
    }

    [Fact]
    public async Task OpenFailure_LeavesNothingPinned()
    {
        using IDisposable calm = AudioEvictionPressure.Relax();
        using InferenceEngine engine = new("cpu");
        // Whisper infers its config from the repo id before fetching, so an unknown one fails without a download.
        ModelSpec unknown = new() { Requested = "whisper:acme/not-a-whisper", Modality = Modality.Transcribe };

        await Assert.ThrowsAsync<ArgumentException>(() => engine.Transcribe.OpenTranscriberAsync(unknown));
        Assert.False(engine.AudioRuntime.Stt.IsPinned("acme/not-a-whisper"));
        Assert.Empty(engine.AudioRuntime.Stt.ResidentKeys);
    }

    [Fact]
    public async Task OpenCancelledWhileQueued_ThrowsAndLeavesNothingPinned()
    {
        using IDisposable calm = AudioEvictionPressure.Relax();
        using InferenceEngine engine = new("cpu");
        FakeTts busy = await SeedTtsAsync(engine, KokoroKey, blocking: true);
        await SeedSttAsync(engine, WhisperKey);
        Task<AudioResult> generation = Task.Run(() => engine.Speech.SynthesizeAsync(Kokoro, new SpeechRequest { Text = "Holding the lock." }));
        Assert.True(busy.CallEntered.Wait(TimeSpan.FromSeconds(10)), "the service generation never started.");

        using CancellationTokenSource cancel = new();
        Task<ITranscriberLease> opening = engine.Transcribe.OpenTranscriberAsync(WhisperTiny, cancel.Token);
        cancel.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => opening.WaitAsync(TimeSpan.FromSeconds(10)));
        Assert.False(engine.AudioRuntime.Stt.IsPinned(WhisperKey));

        busy.CallMayFinish.Set();
        await generation.WaitAsync(TimeSpan.FromSeconds(10));
    }

    [Fact]
    public async Task OpenThatStraddlesARelease_IsRefused()
    {
        using IDisposable calm = AudioEvictionPressure.Relax();
        using InferenceEngine engine = new("cpu");
        AudioRuntime runtime = engine.AudioRuntime;
        FakeTts busy = await SeedTtsAsync(engine, KokoroKey, blocking: true);
        Task<AudioResult> generation = Task.Run(() => engine.Speech.SynthesizeAsync(Kokoro, new SpeechRequest { Text = "Holding the lock." }));
        Assert.True(busy.CallEntered.Wait(TimeSpan.FromSeconds(10)), "the service generation never started.");

        Task<ITranscriberLease> opening = engine.Transcribe.OpenTranscriberAsync(WhisperTiny);
        // A release that gives up waiting on the generation lock runs while the open is still queued behind it.
        runtime.UnloadAll();
        await SeedSttAsync(engine, WhisperKey);
        busy.CallMayFinish.Set();
        await generation.WaitAsync(TimeSpan.FromSeconds(10));

        await Assert.ThrowsAsync<ObjectDisposedException>(() => opening.WaitAsync(TimeSpan.FromSeconds(10)));
        Assert.False(runtime.Stt.IsPinned(WhisperKey));
    }

    private static async Task<FakeTts> SeedTtsAsync(InferenceEngine engine, string key, int sampleRate = 24_000, bool blocking = false)
    {
        FakeTts runner = new(sampleRate, blocking);
        ITtsRunner seeded = await engine.AudioRuntime.Tts.GetOrLoadAsync(key, _ => Task.FromResult<ITtsRunner>(runner), CancellationToken.None);
        Assert.Same(runner, seeded);
        return runner;
    }

    private static async Task<FakeStt> SeedSttAsync(InferenceEngine engine, string key)
    {
        FakeStt runner = new();
        ISttRunner seeded = await engine.AudioRuntime.Stt.GetOrLoadAsync(key, _ => Task.FromResult<ISttRunner>(runner), CancellationToken.None);
        Assert.Same(runner, seeded);
        return runner;
    }

    private static AudioRequest Clip16k() =>
        new() { Audio = new AudioClip { Data = AudioClipCodec.EncodeWav(Ramp(1600), null, 16_000), Format = "wav" } };

    private static float[] Ramp(int length)
    {
        float[] samples = new float[length];
        for (int i = 0; i < length; i++)
        {
            samples[i] = (i % 64 - 32) / 64f;
        }
        return samples;
    }

    /// <summary>Returns a fixed buffer; a blocking one parks each call until released, so a test can hold a call in
    /// flight.</summary>
    private sealed class FakeTts(int sampleRate, bool blocking) : ITtsRunner
    {
        private int _calls;

        public float[] Output { get; } = [0.1f, -0.2f, 0.3f];

        public ManualResetEventSlim CallEntered { get; } = new();

        public ManualResetEventSlim CallMayFinish { get; } = new(!blocking);

        public int Calls => Volatile.Read(ref _calls);

        public TtsJob? LastJob { get; private set; }

        public bool Disposed { get; private set; }

        public int SampleRate => sampleRate;

        public float[] Synthesize(IBackend backend, TtsJob job)
        {
            Interlocked.Increment(ref _calls);
            LastJob = job;
            CallEntered.Set();
            CallMayFinish.Wait(TimeSpan.FromSeconds(30));
            return Output;
        }

        public void Dispose() => Disposed = true;
    }

    private sealed class FakeStt : ISttRunner
    {
        private int _calls;

        public int Calls => Volatile.Read(ref _calls);

        public float[]? LastAudio { get; private set; }

        public AudioRequest? LastRequest { get; private set; }

        public bool Disposed { get; private set; }

        public string Transcribe(IBackend backend, float[] audioMono, AudioRequest request)
        {
            Interlocked.Increment(ref _calls);
            LastAudio = audioMono;
            LastRequest = request;
            return "  heard  ";
        }

        public IReadOnlyList<SttSegment>? TranscribeTimed(IBackend backend, float[] audioMono, AudioRequest request) => null;

        public ScoreTranscriptResult? TranscribeScore(IBackend backend, float[] audioMono, AudioRequest request) => null;

        public void Dispose() => Disposed = true;
    }
}
