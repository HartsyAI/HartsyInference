namespace SharpInference.Core.Pipelines;

/// <summary>Interface for audio pipelines (Whisper STT, Kokoro TTS, voice conversion).</summary>
public interface IAudioPipeline : IDisposable
{
    /// <summary>Name of the model loaded in this pipeline.</summary>
    string ModelName { get; }
}
