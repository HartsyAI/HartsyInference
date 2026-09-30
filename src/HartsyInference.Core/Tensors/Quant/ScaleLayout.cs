namespace HartsyInference.Core.Tensors.Quant;

/// <summary>How a recipe's scale tensor is arranged in memory.</summary>
public enum ScaleLayout
{
    /// <summary>Plain row-major <c>[ceil(rows/blockRows), ceil(cols/blockCols)]</c>, as safetensors checkpoints store them.</summary>
    RowMajorBlocks,

    /// <summary>NVIDIA's blocked layout, padded to 128 rows and 4 block columns and swizzled in 128-row tiles.</summary>
    Swizzled128,
}
