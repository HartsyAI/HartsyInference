using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.LLM.Kimi;

/// <summary>Kimi K3.1 tensors read from a safetensors checkpoint directory through its <c>model.safetensors.index.json</c>. Each tensor is a
/// borrowed view of its shard's mapping (the view keeps the mapping alive), in the dtype the checkpoint stores it; the loader converts to F32.</summary>
public sealed class KimiK31CheckpointWeights : IKimiK31WeightStore
{
    private ShardedSafeTensorSet? _set;

    private KimiK31CheckpointWeights(ShardedSafeTensorSet set) => _set = set;

    /// <summary>Opens the checkpoint's shards.</summary>
    /// <exception cref="HartsyInference.Core.HartsyInferenceException">The index is missing or a shard file is absent.</exception>
    public static KimiK31CheckpointWeights Open(string directory) => new(ShardedSafeTensorSet.OpenIndex(directory));

    /// <inheritdoc/>
    public bool Has(string name) => Set.Inventory.ContainsKey(name);

    /// <inheritdoc/>
    public Tensor Read(string name) => Set.GetTensor(name);

    private ShardedSafeTensorSet Set => _set ?? throw new ObjectDisposedException(nameof(KimiK31CheckpointWeights));

    /// <inheritdoc/>
    public void Dispose()
    {
        ShardedSafeTensorSet? set = Interlocked.Exchange(ref _set, null);
        set?.Dispose();
    }
}
