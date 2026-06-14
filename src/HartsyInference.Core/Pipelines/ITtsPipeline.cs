namespace HartsyInference.Core.Pipelines;

/// <summary>Text-to-speech pipeline interface.</summary>
public interface ITtsPipeline : IAudioPipeline
{
    /// <summary>Synthesizes speech from text, streaming audio chunks as they're generated.</summary>
    IAsyncEnumerable<ReadOnlyMemory<byte>> SynthesizeAsync(string text, string voice, CancellationToken cancellationToken = default);
}
