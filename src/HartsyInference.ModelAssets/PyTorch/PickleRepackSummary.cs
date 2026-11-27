namespace HartsyInference.ModelAssets.PyTorch;

/// <summary>What <see cref="StreamingPickleRepack.Repack"/> wrote.</summary>
/// <param name="TensorCount">Tensors written.</param>
/// <param name="TensorBytes">Bytes of tensor data written.</param>
/// <param name="Streamed">False when the checkpoint was a legacy (non-ZIP) pickle that had to be read whole.</param>
/// <param name="StorageRestarts">Times a tensor-by-tensor read had to reopen a storage entry from its start, which for one huge entry is a full pass each.</param>
public readonly record struct PickleRepackSummary(int TensorCount, long TensorBytes, bool Streamed, int StorageRestarts);
