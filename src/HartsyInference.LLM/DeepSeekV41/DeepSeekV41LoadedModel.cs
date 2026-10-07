using HartsyInference.Core.Engram;

namespace HartsyInference.LLM.DeepSeekV41;

/// <summary>A host reference model together with the open checkpoint and row stores its weights are read from.</summary>
/// <remarks>Dense weights were copied to F32 at load, but routed experts and Engram rows are read through the checkpoint on demand, so this must stay alive while the model is used.</remarks>
public sealed class DeepSeekV41LoadedModel : IDisposable
{
    private readonly DeepSeekV41Checkpoint? _ownedCheckpoint;
    private readonly List<EngramTableStore> _stores;

    /// <summary>The model.</summary>
    public DeepSeekV41HostModel Model { get; }

    /// <summary>The checkpoint it was loaded from.</summary>
    public DeepSeekV41Checkpoint Checkpoint { get; }

    internal DeepSeekV41LoadedModel(DeepSeekV41HostModel model, DeepSeekV41Checkpoint checkpoint, bool ownsCheckpoint, List<EngramTableStore> stores)
    {
        Model = model;
        Checkpoint = checkpoint;
        _ownedCheckpoint = ownsCheckpoint ? checkpoint : null;
        _stores = stores;
    }

    /// <inheritdoc />
    public void Dispose()
    {
        foreach (EngramTableStore store in _stores) store.Dispose();
        _stores.Clear();
        _ownedCheckpoint?.Dispose();
    }
}
