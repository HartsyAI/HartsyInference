using HartsyInference.Core.Logging;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.Voice.Audio;

namespace HartsyInference.Voice;

/// <summary>The engine's runner leases as the model set's speech models: 16 kHz utterances to the transcriber with no
/// language token, sentences to the synthesizer in the configured voice.</summary>
/// <remarks>Any engine release (dispose, free memory, backend switch) revokes the leases, and the next call throws
/// <see cref="ObjectDisposedException"/>; <see cref="Reopen"/> opens a fresh pair through the same service calls. Every
/// member runs on the model set's GPU thread, so the lease fields need no synchronization.</remarks>
internal sealed class VoiceLeaseSpeech : IVoiceSpeech
{
    private static readonly AudioClip NoClip = new() { Data = [] };

    private readonly Func<CancellationToken, Task<ISynthesizerLease>> _openSynthesizer;
    private readonly Func<CancellationToken, Task<ITranscriberLease>> _openTranscriber;
    private readonly SpeechRequest _speech;
    // An empty language omits the language token, which English-only Whisper checkpoints have no slot for.
    private readonly AudioRequest _recognition = new() { Audio = NoClip, Language = "" };
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
        // The GPU thread has no synchronization context and holds no gate here, so waiting on the open is safe.
        OpenLeasesAsync(CancellationToken.None).GetAwaiter().GetResult();
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
        _sampleRate = tts.SampleRate;
        ISynthesizerLease? oldTts = Interlocked.Exchange(ref _tts, tts);
        ITranscriberLease? oldStt = Interlocked.Exchange(ref _stt, stt);
        oldTts?.Dispose();
        oldStt?.Dispose();
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
