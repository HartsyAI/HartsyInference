namespace HartsyInference.LLM.Models.Shared;

/// <summary>RoPE of one attention layer type: the base and the fraction of the head dimension that rotates.</summary>
public readonly record struct LayerRopeParameters(float Theta, float PartialRotaryFactor)
{
    /// <summary>Leading dimensions that rotate (all of them for a factor of 1).</summary>
    public int RotaryDim(int headDim) => (int)(PartialRotaryFactor * headDim);

    /// <summary>Rotated frequency pairs of a layout whose unrotated pairs keep zero frequency in a full-width table.</summary>
    public int RotatedPairs(int headDim) => RotaryDim(headDim) / 2;
}
