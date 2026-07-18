using HartsyInference.Engine;

namespace HartsyInference.Engine.Dispatch;

/// <summary>A loaded model ready to generate. Owns the model's own state (weights, tokenizer, pipeline) but not the
/// shared <c>IBackend</c>, mirroring the engine's pipeline-ownership convention. Disposed to free the model.</summary>
public interface IModalityRunner : IDisposable
{
    /// <summary>The modality this runner serves.</summary>
    Modality Modality { get; }

    /// <summary>The model id or path this runner was loaded from (used as a session cache key).</summary>
    string ModelId { get; }
}
