namespace HartsyInference.LLM.DeepSeekV41;

/// <summary>What the config alone says about one layer: its compression ratio and which shared-state roles it plays.</summary>
/// <param name="Layer">Position in the config's layer list; backbone layers first, then the draft layers.</param>
/// <param name="IsDraft">True for a draft layer, whose weights live under <c>mtp.{Layer - num_hidden_layers}</c>.</param>
/// <param name="CompressRatio">The config's <c>compress_ratios</c> entry for this layer; 0 means no compressed-KV path.</param>
/// <param name="IsKvSource">Whether the layer publishes compressed KV that later layers reuse.</param>
/// <param name="IsIndexSource">Whether the layer computes sparse-attention indices that later layers reuse.</param>
/// <param name="EngramSlot">Position in <c>engram_layer_ids</c> when the layer carries an Engram table, else null.</param>
public sealed record DeepSeekV41LayerPlan(int Layer, bool IsDraft, int CompressRatio, bool IsKvSource, bool IsIndexSource, int? EngramSlot)
{
    /// <summary>True when the layer has an Engram table.</summary>
    public bool HasEngram => EngramSlot is not null;
}
