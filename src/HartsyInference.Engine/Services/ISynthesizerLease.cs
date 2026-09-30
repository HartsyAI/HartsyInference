using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Services;

/// <summary>A text-to-speech model held resident on its engine for direct, synchronous synthesis, opened by
/// <see cref="ISpeechService.OpenSynthesizerAsync"/>: the per-sentence path for a caller that owns a device thread, and
/// the residency pin for a host that keeps a model warm between service calls.</summary>
/// <remarks>
/// <list type="bullet">
/// <item>Use a lease from one thread at a time.</item>
/// <item><see cref="Synthesize"/> takes neither the engine's audio generation lock nor the device gate. The caller
/// serializes device use by holding <see cref="DeviceGate"/> for the engine's backend
/// (<see cref="InferenceEngine.ComputeBackend"/>) around each call, which also keeps it clear of service calls on that
/// device.</item>
/// <item>The lease always runs on the engine's own backend. A host that keeps an LLM on another card builds the engine
/// on the audio card and sets <see cref="TextRequest.Device"/> on every text generation request.</item>
/// <item>The runner is pinned: memory-pressure eviction never unloads it while the lease is open, and service calls
/// for the same model run on it. A pin survives memory pressure, not the device: any engine release (Dispose,
/// FreeMemory, SetBackend, SetPlacement) waits for a call in flight, then unloads the runner and revokes the lease.
/// The wait is bounded by the engine's 120 s release budget; a call still running past it has its runner disposed
/// underneath it.</item>
/// <item>A revoked lease reports it only as <see cref="ObjectDisposedException"/> on its next call; nothing signals it
/// sooner. To re-open, dispose the revoked lease, which is a no-op, and open a new one. After FreeMemory, SetBackend or
/// SetPlacement the engine stays usable, and the new lease reloads the model on the engine's current backend. After
/// Dispose, open it on a new engine.</item>
/// <item>Dispose releases the pin. It is idempotent and waits for a call in flight.</item>
/// </list></remarks>
public interface ISynthesizerLease : IDisposable
{
    /// <summary>Output sample rate in Hz.</summary>
    int SampleRate { get; }

    /// <summary>Synthesizes <paramref name="text"/> to mono PCM at <see cref="SampleRate"/> on the calling thread.
    /// <paramref name="options"/> supplies the voice, reference and knobs exactly as
    /// <see cref="ISpeechService.SynthesizeAsync"/> reads them; its <see cref="SpeechRequest.Text"/> is not read. For a
    /// model whose voice selects its weights (Piper), a voice other than the one the lease opened is refused.</summary>
    float[] Synthesize(string text, SpeechRequest options);
}
