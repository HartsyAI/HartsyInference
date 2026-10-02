namespace HartsyInference.Voice;

/// <summary>The speech models a <see cref="VoiceModelSet"/> runs: recognition of one utterance and synthesis of one
/// sentence. Called only on the model set's GPU thread, one call at a time, and disposed there.</summary>
internal interface IVoiceSpeech : IDisposable
{
    /// <summary>Rate of the audio <see cref="Synthesize"/> returns.</summary>
    int SynthesisSampleRate { get; }

    /// <summary>The words in <paramref name="audio"/> (16 kHz, ±1).</summary>
    string Transcribe(float[] audio);

    /// <summary>Mono audio (±1) for one sentence of speakable text; <paramref name="cancel"/> stops a synthesizer that
    /// checks it between its stages with <see cref="OperationCanceledException"/>.</summary>
    float[] Synthesize(string text, CancellationToken cancel);

    /// <summary>Loads the models again after their owner released them (a call threw <see cref="ObjectDisposedException"/>).
    /// Called on the GPU thread without the device gate held, because loading takes the gate itself; throws when the
    /// models cannot be reloaded.</summary>
    void Reopen();
}
