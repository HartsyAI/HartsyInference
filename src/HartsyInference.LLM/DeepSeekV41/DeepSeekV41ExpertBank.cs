using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.LLM.DeepSeekV41;

/// <summary>Borrowed access to one layer's routed experts; every tensor is a view of the checkpoint's shards and dies with it.</summary>
public sealed class DeepSeekV41ExpertBank
{
    private readonly DeepSeekV41Checkpoint _checkpoint;

    internal DeepSeekV41ExpertBank(DeepSeekV41Checkpoint checkpoint, int layer, int expertCount)
    {
        _checkpoint = checkpoint;
        Layer = layer;
        ExpertCount = expertCount;
    }

    /// <summary>Layer number, backbone first then the draft layers.</summary>
    public int Layer { get; }

    /// <summary>Routed experts in this layer.</summary>
    public int ExpertCount { get; }

    /// <summary>The canonical key of an expert's projection weight.</summary>
    public string WeightKey(int expert, DeepSeekV41ExpertProjection projection)
    {
        if ((uint)expert >= (uint)ExpertCount)
            throw new ArgumentOutOfRangeException(nameof(expert), expert, $"Layer {Layer} has {ExpertCount} routed experts.");
        return DeepSeekV41Checkpoint.ExpertWeightKey(_checkpoint.Config, Layer, expert, projection);
    }

    /// <summary>Where the projection's bytes live.</summary>
    public TensorLocation Location(int expert, DeepSeekV41ExpertProjection projection) =>
        _checkpoint.GetLocation(WeightKey(expert, projection));

    /// <summary>The projection's packed weight as a borrowed view.</summary>
    public Tensor Weight(int expert, DeepSeekV41ExpertProjection projection) =>
        _checkpoint.GetWeight(WeightKey(expert, projection));

    /// <summary>The projection's weight with its bound quantization recipe, or null when the weight is stored unquantized.</summary>
    public QuantWeightInfo? Quant(int expert, DeepSeekV41ExpertProjection projection) =>
        _checkpoint.GetQuant(WeightKey(expert, projection));
}
