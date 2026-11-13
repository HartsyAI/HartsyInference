namespace HartsyInference.LLM.Kimi;

/// <summary>A checkpoint's tensor source that its owner disposes, releasing the shard mappings the tensors are read from. It reads row and
/// column ranges, so a sharded rank loads only its share, and whole tensors, so the vision tower loads from the same store.</summary>
internal interface IKimiK31WeightStore : IKimiK31RankWeightSource, IDisposable
{
}
