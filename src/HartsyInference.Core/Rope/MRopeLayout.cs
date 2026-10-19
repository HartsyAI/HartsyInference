namespace HartsyInference.Core.Rope;

/// <summary>How multimodal RoPE assigns a head's rotary frequency pairs to the temporal, height and width position axes.</summary>
public enum MRopeLayout
{
    /// <summary>Contiguous sections <c>[T…][H…][W…]</c> (Qwen2-VL, Qwen2.5-VL).</summary>
    Sectioned,

    /// <summary>Interleaved: pair <c>j</c> reads axis <c>j % 3</c> while <c>j / 3</c> is inside that axis's section, else the
    /// temporal axis (transformers' <c>apply_interleaved_mrope</c>; Qwen3-VL, Qwen3.5).</summary>
    Interleaved,
}
