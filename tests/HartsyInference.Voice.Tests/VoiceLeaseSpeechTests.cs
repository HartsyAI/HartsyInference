using HartsyInference.Cpu;
using HartsyInference.Engine.Services;
using HartsyInference.Voice.Gpu;
using HartsyInference.Voice.Tests.Fakes;
using Xunit;

namespace HartsyInference.Voice.Tests;

/// <summary>The engine leases as the voice models: an engine release revokes both leases, the GPU thread reopens them
/// once and retries the job on the new pair (even when closing a revoked lease throws), and the requests carry English
/// recognition at 16 kHz and the model id's voice.</summary>
public sealed class VoiceLeaseSpeechTests
{
    [Fact]
    public async Task RevokedLeasesAreReopenedOnceAndTheJobRetriedOnTheNewPair()
    {
        List<FakeSynthesizerLease> synthesizers = [];
        List<FakeTranscriberLease> transcribers = [];
        using VoiceLeaseSpeech speech = await VoiceLeaseSpeech.OpenAsync(
            _ =>
            {
                // The first lease fails to close once revoked, which must not fail the reopen.
                FakeSynthesizerLease lease = new(synthesizers.Count + 1, throwOnDispose: synthesizers.Count == 0);
                synthesizers.Add(lease);
                return Task.FromResult<ISynthesizerLease>(lease);
            },
            _ =>
            {
                FakeTranscriberLease lease = new(transcribers.Count + 1);
                transcribers.Add(lease);
                return Task.FromResult<ITranscriberLease>(lease);
            },
            voice: "af_heart", CancellationToken.None);
        using CpuBackend device = new();
        using VoiceGpuWorker worker = new(device, speech.Reopen);

        synthesizers[0].Revoked = true;
        transcribers[0].Revoked = true;
        float[] audio = await worker.RunAsync(VoiceGpuJobKind.Synthesize, () => speech.Synthesize("Hello there."), CancellationToken.None)
            .WaitAsync(TimeSpan.FromSeconds(10));
        string heard = await worker.RunAsync(VoiceGpuJobKind.Transcribe, () => speech.Transcribe(new float[16_000]), CancellationToken.None)
            .WaitAsync(TimeSpan.FromSeconds(10));

        Assert.Equal([2f], audio);
        Assert.Equal("generation 2", heard);
        Assert.Equal(1, worker.Reopens);
        Assert.Equal(2, synthesizers.Count);
        Assert.Equal(2, transcribers.Count);
        Assert.True(synthesizers[0].Disposed);
        Assert.True(transcribers[0].Disposed);
        Assert.Equal("af_heart", synthesizers[1].LastOptions!.Voice);
        Assert.Equal("en", transcribers[1].LastOptions!.Language);
        Assert.Equal(16_000, transcribers[1].LastSampleRate);

        speech.Dispose();
        Assert.True(synthesizers[1].Disposed);
        Assert.True(transcribers[1].Disposed);
    }

    [Fact]
    public async Task AReleasedEngineThatCannotReopenFailsTheJobAndTheThreadCarriesOn()
    {
        int opens = 0;
        FakeSynthesizerLease first = new(1);
        using VoiceLeaseSpeech speech = await VoiceLeaseSpeech.OpenAsync(
            _ => ++opens == 1 ? Task.FromResult<ISynthesizerLease>(first) : throw new ObjectDisposedException("engine"),
            _ => Task.FromResult<ITranscriberLease>(new FakeTranscriberLease(1)),
            voice: null, CancellationToken.None);
        using CpuBackend device = new();
        using VoiceGpuWorker worker = new(device, speech.Reopen);

        first.Revoked = true;
        await Assert.ThrowsAsync<ObjectDisposedException>(() =>
            worker.RunAsync(VoiceGpuJobKind.Synthesize, () => speech.Synthesize("Hello."), CancellationToken.None));

        Assert.Equal(7, await worker.RunAsync(VoiceGpuJobKind.Warm, () => 7, CancellationToken.None).WaitAsync(TimeSpan.FromSeconds(10)));
        Assert.Equal(2, opens);
    }
}
