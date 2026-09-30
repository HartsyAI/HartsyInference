using HartsyInference.Core.Logging;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.Voice.Audio;

namespace HartsyInference.Voice;

/// <summary>The engine's runner leases as the model set's speech models: 16 kHz utterances to the transcriber in
/// English, sentences to the synthesizer in the configured voice.</summary>
/// <remarks>Any engine release (dispose, free memory, backend switch) revokes the leases, and the next call throws
/// <see cref="ObjectDisposedException"/>; <see cref="Reopen"/> opens a fresh pair through the same service calls. Every
/// member runs on the model set's GPU thread, so the lease fields need no synchronization.</remarks>
internal sealed class VoiceLeaseSpeech : IVoiceSpeech
{
    private static readonly AudioClip NoClip = new() { Data = [] };

    // Reopening reloads both models from the local cache; minutes would mean the open is stuck, not slow.
    private static readonly TimeSpan ReopenTimeout = TimeSpan.FromMinutes(2);

    private readonly Func<CancellationToken, Task<ISynthesizerLease>> _openSynthesizer;
    private readonly Func<CancellationToken, Task<ITranscriberLease>> _openTranscriber;
    private readonly SpeechRequest _speech;
    // English: an English-only checkpoint drops the language token itself, while an empty language on a multilingual
    // one means no language token at all, which it answers with hallucinated loops.
    private readonly AudioRequest _recognition = new() { Audio = NoClip, Language = "en" };
    private ISynthesizerLease? _tts;
    private ITranscriberLease? _stt;
    private int _sampleRate;
    private int _disposed;

    private VoiceLeaseSpeech(Func<CancellationToken, Task<ISynthesizerLease>> openSynthesizer,
        Func<CancellationToken, Task<ITranscriberLease>> openTranscriber, string? voice)
    {
        _openSynthesizer = openSynthesizer;
        _openTranscriber = openTranscriber;
        _speech = new SpeechRequest { Text = "", Voice = voice };
    }

    /// <summary>Opens both leases. <paramref name="voice"/> is the synthesizer's voice, null for its default.</summary>
    public static async Task<VoiceLeaseSpeech> OpenAsync(Func<CancellationToken, Task<ISynthesizerLease>> openSynthesizer,
        Func<CancellationToken, Task<ITranscriberLease>> openTranscriber, string? voice, CancellationToken cancel)
    {
        ArgumentNullException.ThrowIfNull(openSynthesizer);
        ArgumentNullException.ThrowIfNull(openTranscriber);
        VoiceLeaseSpeech speech = new(openSynthesizer, openTranscriber, voice);
        try
        {
            await speech.OpenLeasesAsync(cancel).ConfigureAwait(false);
            return speech;
        }
        catch
        {
            speech.Dispose();
            throw;
        }
    }

    // Read by the session thread too; a reopened lease holds the same model, so the rate never changes.
    public int SynthesisSampleRate => _sampleRate;

    private ISynthesizerLease Synthesizer => _tts ?? throw new ObjectDisposedException(nameof(VoiceLeaseSpeech));

    private ITranscriberLease Transcriber => _stt ?? throw new ObjectDisposedException(nameof(VoiceLeaseSpeech));

    public string Transcribe(float[] audio) => Transcriber.Transcribe(audio, VoiceAudioFrontend.SampleRate, _recognition);

    public float[] Synthesize(string text) => Synthesizer.Synthesize(text, _speech);

    public void Reopen()
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        Logs.Warning("[Voice] The engine released the speech models; reopening them.");
        // Bounded so a stuck open cannot hold the GPU thread, and with it the model set's shutdown, forever. The GPU
        // thread has no synchronization context and holds no gate here, so waiting on the open is safe.
        using CancellationTokenSource timeout = new(ReopenTimeout);
        OpenLeasesAsync(timeout.Token).GetAwaiter().GetResult();
    }

    private async Task OpenLeasesAsync(CancellationToken cancel)
    {
        ISynthesizerLease tts = await _openSynthesizer(cancel).ConfigureAwait(false);
        ITranscriberLease stt;
        try
        {
            stt = await _openTranscriber(cancel).ConfigureAwait(false);
        }
        catch
        {
            tts.Dispose();
            throw;
        }
        if (_sampleRate != 0 && tts.SampleRate != _sampleRate)
        {
            // Sessions size their resamplers from the first rate; a reopened model must be the same model.
            DisposeReplaced(tts);
            DisposeReplaced(stt);
            throw new InvalidOperationException($"The reopened synthesizer runs at {tts.SampleRate} Hz, not {_sampleRate} Hz.");
        }
        _sampleRate = tts.SampleRate;
        ISynthesizerLease? oldTts = Interlocked.Exchange(ref _tts, tts);
        ITranscriberLease? oldStt = Interlocked.Exchange(ref _stt, stt);
        // The new pair is already in place; a revoked lease that fails to close must not fail the reopen.
        DisposeReplaced(oldTts);
        DisposeReplaced(oldStt);
    }

    private static void DisposeReplaced(IDisposable? lease)
    {
        try
        {
            lease?.Dispose();
        }
        catch (Exception ex)
        {
            Logs.Warning($"[Voice] Closing a replaced speech lease failed: {ex.Message}");
        }
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }
        ISynthesizerLease? tts = Interlocked.Exchange(ref _tts, null);
        ITranscriberLease? stt = Interlocked.Exchange(ref _stt, null);
        try
        {
            tts?.Dispose();
        }
        finally
        {
            stt?.Dispose();
        }
    }
}
