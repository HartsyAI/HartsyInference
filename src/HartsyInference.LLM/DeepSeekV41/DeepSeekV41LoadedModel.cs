using HartsyInference.Core.Engram;

namespace HartsyInference.LLM.DeepSeekV41;

/// <summary>A host reference model together with the open checkpoint and row stores its weights are read from.</summary>
/// <remarks>Dense weights were copied to F32 at load, but routed experts and Engram rows are read through the checkpoint on demand, so this must stay alive while the model is used.
/// The model and its shared expert cache are not thread-safe: run one sequence at a time.</remarks>
public sealed class DeepSeekV41LoadedModel : IDisposable
{
    private readonly DeepSeekV41Checkpoint? _ownedCheckpoint;
    private readonly List<EngramTableStore> _stores;

    private readonly DeepSeekV41HostModel _model;
    private readonly DeepSeekV41Checkpoint _checkpoint;
    private bool _disposed;

    /// <summary>The longest sequence this model was loaded for; it sized the rope tables.</summary>
    public int MaxTokens { get; }

    /// <summary>The model.</summary>
    /// <exception cref="ObjectDisposedException">The loaded model was disposed, so its checkpoint and row stores are gone.</exception>
    public DeepSeekV41HostModel Model => _disposed ? throw new ObjectDisposedException(nameof(DeepSeekV41LoadedModel)) : _model;

    /// <summary>The checkpoint it was loaded from.</summary>
    /// <exception cref="ObjectDisposedException">The loaded model was disposed.</exception>
    public DeepSeekV41Checkpoint Checkpoint => _disposed ? throw new ObjectDisposedException(nameof(DeepSeekV41LoadedModel)) : _checkpoint;

    internal DeepSeekV41LoadedModel(DeepSeekV41HostModel model, DeepSeekV41Checkpoint checkpoint, bool ownsCheckpoint, List<EngramTableStore> stores, int maxTokens)
    {
        MaxTokens = maxTokens;
        _model = model;
        _checkpoint = checkpoint;
        _ownedCheckpoint = ownsCheckpoint ? checkpoint : null;
        _stores = stores;
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        foreach (EngramTableStore store in _stores) store.Dispose();
        _stores.Clear();
        _ownedCheckpoint?.Dispose();
    }
}
