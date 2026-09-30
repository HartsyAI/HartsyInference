using HartsyInference.Core.Backends;
using HartsyInference.Core.Configuration;
using HartsyInference.Cpu;
using HartsyInference.Engine.Audio;
using HartsyInference.Engine.Requests;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The audio memory-pressure sweep must keep the model about to run and any pinned model. It used to compare the prefixed job key against the caches' bare keys, so under pressure every STT↔TTS switch evicted the incoming runner and reloaded it. Pressure is forced through <c>vram.audioEvictBelowGb</c>; the host floor only fires where <c>/proc/meminfo</c> exists, so the eviction assertions are conditional on that while the keep assertions never are.</summary>
public sealed class AudioRuntimeEvictionTests
{
    private static readonly bool HostPressureObservable = File.Exists("/proc/meminfo");

    [Fact]
    public async Task Switch_UnderHostPressure_KeepsIncomingModel()
    {
        using CpuBackend backend = new();
        AudioRuntime runtime = new();
        FakeTtsRunner tts = new();
        FakeSttRunner stt = new();
        await runtime.Tts.GetOrLoadAsync("kokoro", _ => Task.FromResult<ITtsRunner>(tts), CancellationToken.None);
        await runtime.RunAsync(backend, new AudioJob(runtime.Tts, "kokoro"), _ => Task.FromResult(0), CancellationToken.None);
        await runtime.Stt.GetOrLoadAsync("whisper", _ => Task.FromResult<ISttRunner>(stt), CancellationToken.None);

        using (ForceHostPressure())
        {
            await runtime.RunAsync(backend, new AudioJob(runtime.Stt, "whisper"), _ => Task.FromResult(0), CancellationToken.None);
        }

        Assert.False(stt.Disposed);
        Assert.True(runtime.Stt.IsResident("whisper"));
        if (HostPressureObservable)
        {
            Assert.True(tts.Disposed);
            Assert.False(runtime.Tts.IsResident("kokoro"));
        }
    }

    [Fact]
    public async Task Switch_UnderHostPressure_KeepsPinnedModel_UntilUnpinned()
    {
        using CpuBackend backend = new();
        AudioRuntime runtime = new();
        FakeTtsRunner tts = new();
        FakeSttRunner stt = new();
        await runtime.Tts.GetOrLoadAsync("kokoro", _ => Task.FromResult<ITtsRunner>(tts), CancellationToken.None);
        await runtime.Stt.GetOrLoadAsync("whisper", _ => Task.FromResult<ISttRunner>(stt), CancellationToken.None);

        IDisposable pin = runtime.Tts.Pin("kokoro");
        using (ForceHostPressure())
        {
            await runtime.RunAsync(backend, new AudioJob(runtime.Stt, "whisper"), _ => Task.FromResult(0), CancellationToken.None);
            Assert.False(tts.Disposed);
            Assert.False(stt.Disposed);

            pin.Dispose();
            await runtime.RunAsync(backend, new AudioJob(runtime.Tts, "kokoro"), _ => Task.FromResult(0), CancellationToken.None);
            await runtime.RunAsync(backend, new AudioJob(runtime.Stt, "whisper"), _ => Task.FromResult(0), CancellationToken.None);
        }

        if (HostPressureObservable)
        {
            Assert.True(tts.Disposed);
            Assert.True(stt.Disposed);
        }
    }

    [Fact]
    public async Task AlternatingSwitches_WithBothPinned_LoadOnceAndNeverDispose()
    {
        using CpuBackend backend = new();
        AudioRuntime runtime = new();
        FakeTtsRunner tts = new();
        FakeSttRunner stt = new();
        int ttsLoads = 0;
        int sttLoads = 0;
        using IDisposable ttsPin = runtime.Tts.Pin("kokoro");
        using IDisposable sttPin = runtime.Stt.Pin("whisper");

        using (ForceHostPressure())
        {
            for (int turn = 0; turn < 10; turn++)
            {
                await runtime.RunAsync(backend, new AudioJob(runtime.Tts, "kokoro"), async ct =>
                {
                    await runtime.Tts.GetOrLoadAsync("kokoro", _ => { ttsLoads++; return Task.FromResult<ITtsRunner>(tts); }, ct);
                    return 0;
                }, CancellationToken.None);
                await runtime.RunAsync(backend, new AudioJob(runtime.Stt, "whisper"), async ct =>
                {
                    await runtime.Stt.GetOrLoadAsync("whisper", _ => { sttLoads++; return Task.FromResult<ISttRunner>(stt); }, ct);
                    return 0;
                }, CancellationToken.None);
            }
        }

        Assert.Equal(1, ttsLoads);
        Assert.Equal(1, sttLoads);
        Assert.False(tts.Disposed);
        Assert.False(stt.Disposed);
    }

    [Fact]
    public void Pin_IsRefCounted_AndDisposeIsIdempotent()
    {
        AudioRunnerCache<ITtsRunner> cache = new("tts");
        IDisposable first = cache.Pin("kokoro");
        IDisposable second = cache.Pin("kokoro");
        Assert.True(cache.IsPinned("kokoro"));

        first.Dispose();
        first.Dispose();
        Assert.True(cache.IsPinned("kokoro"));

        second.Dispose();
        Assert.False(cache.IsPinned("kokoro"));
    }

    [Fact]
    public async Task UnloadAll_DropsPinnedRunners()
    {
        AudioRuntime runtime = new();
        FakeTtsRunner tts = new();
        await runtime.Tts.GetOrLoadAsync("kokoro", _ => Task.FromResult<ITtsRunner>(tts), CancellationToken.None);
        using IDisposable pin = runtime.Tts.Pin("kokoro");

        runtime.UnloadAll();

        Assert.True(tts.Disposed);
        Assert.False(runtime.Tts.IsResident("kokoro"));
    }

    [Fact]
    public void AudioJob_ModelKey_PrefixesTheCacheCategory()
    {
        AudioRuntime runtime = new();
        Assert.Equal("tts:hexgrad/Kokoro-82M", new AudioJob(runtime.Tts, "hexgrad/Kokoro-82M").ModelKey);
        Assert.Equal("fx:demucs:htdemucs", new AudioJob(runtime.Demucs, "htdemucs").ModelKey);
    }

    /// <summary>Raises the host-RAM eviction floor far above any real machine so the next model switch takes the pressure path; restores the knob on dispose.</summary>
    private static IDisposable ForceHostPressure()
    {
        bool hadOverride = KnobStore.HasOverride(EngineKnobs.AudioEvictBelowGb);
        long previous = EngineKnobs.AudioEvictBelowGb.Value;
        KnobStore.Set(EngineKnobs.AudioEvictBelowGb, 1_000_000_000L);
        return new KnobRestore(hadOverride, previous);
    }

    private sealed class KnobRestore(bool hadOverride, long previous) : IDisposable
    {
        public void Dispose()
        {
            if (hadOverride)
            {
                KnobStore.Set(EngineKnobs.AudioEvictBelowGb, previous);
            }
            else
            {
                KnobStore.Clear(EngineKnobs.AudioEvictBelowGb);
            }
        }
    }

    private sealed class FakeTtsRunner : ITtsRunner
    {
        public bool Disposed { get; private set; }

        public int SampleRate => 16_000;

        public float[] Synthesize(IBackend backend, TtsJob job) => [];

        public void Dispose() => Disposed = true;
    }

    private sealed class FakeSttRunner : ISttRunner
    {
        public bool Disposed { get; private set; }

        public string Transcribe(IBackend backend, float[] audioMono, AudioRequest request) => string.Empty;

        public IReadOnlyList<SttSegment>? TranscribeTimed(IBackend backend, float[] audioMono, AudioRequest request) => null;

        public ScoreTranscriptResult? TranscribeScore(IBackend backend, float[] audioMono, AudioRequest request) => null;

        public void Dispose() => Disposed = true;
    }
}
