namespace HartsyInference.LLM.Kimi;

/// <summary>A checkpoint's tensor source that its owner disposes, releasing the shard mappings the tensors are read from.</summary>
internal interface IKimiK31WeightStore : IKimiK31WeightSource, IDisposable
{
}
