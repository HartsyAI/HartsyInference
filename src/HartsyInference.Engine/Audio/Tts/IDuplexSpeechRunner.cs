using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;

namespace HartsyInference.Engine.Audio;

/// <summary>A loaded full-duplex speech-to-speech model. It is an <see cref="ITtsRunner"/> so the speech runner cache, eviction, leases and
/// footprint serve it unchanged, but <see cref="ITtsRunner.Synthesize"/> refuses: it has no text to read, only a user to answer.</summary>
internal interface IDuplexSpeechRunner : ITtsRunner
{
    /// <summary>Opens a conversation on the loaded model, which serves one at a time. <paramref name="referenceMono24k"/> is the decoded
    /// voice reference, or null. The returned session is unguarded: the caller adds the lease's lock and the device gate.</summary>
    IDuplexSpeechSession Open(DuplexSpeechRequest request, float[]? referenceMono24k);
}
