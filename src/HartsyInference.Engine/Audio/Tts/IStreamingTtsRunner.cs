using HartsyInference.Audio.Streaming;
using HartsyInference.Core.Backends;

namespace HartsyInference.Engine.Audio;

/// <summary>A <see cref="ITtsRunner"/> that can additionally emit audio incrementally as it's generated, instead
/// of only returning the complete buffer at the end. Implemented only by models with a genuine per-frame/per-token
/// decode loop (Kyutai today, more to follow). <see cref="Services.SpeechService.SynthesizeStreamAsync"/> falls
/// back to wrapping the plain <see cref="ITtsRunner.Synthesize"/> result in a single chunk for every model that
/// doesn't implement this, so callers have exactly one code path regardless of which model is selected.</summary>
internal interface IStreamingTtsRunner : ITtsRunner
{
    /// <summary>Synthesizes the job, yielding audio chunks as they become available rather than waiting for the
    /// whole utterance to finish.</summary>
    IAsyncEnumerable<AudioChunk> SynthesizeStream(IBackend backend, TtsJob job, CancellationToken cancel);
}
