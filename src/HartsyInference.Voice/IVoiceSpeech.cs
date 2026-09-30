namespace HartsyInference.Voice;

/// <summary>The speech models a <see cref="VoiceModelSet"/> runs: recognition of one utterance and synthesis of one
/// sentence. Called only on the model set's GPU thread, one call at a time, and disposed there.</summary>
internal interface IVoiceSpeech : IDisposable
{
    /// <summary>Rate of the audio <see cref="Synthesize"/> returns.</summary>
    int SynthesisSampleRate { get; }

    /// <summary>The words in <paramref name="audio"/> (16 kHz, ±1).</summary>
    string Transcribe(float[] audio);

    /// <summary>Mono audio (±1) for one sentence of speakable text.</summary>
    float[] Synthesize(string text);
}
