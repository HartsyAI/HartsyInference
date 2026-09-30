namespace HartsyInference.ModelAssets.Quant;

/// <summary>Which producer's companion-tensor naming a checkpoint follows.</summary>
public enum QuantFlavor
{
    /// <summary>DeepSeek's own release: <c>X.scale</c> (or V3's <c>X.weight_scale_inv</c>).</summary>
    Official,

    /// <summary>NVIDIA ModelOpt NVFP4: <c>X.weight_scale</c>, <c>X.weight_scale_2</c>, <c>X.input_scale</c>.</summary>
    NvidiaNvfp4,

    /// <summary>AMD Quark: <c>X.weight_scale</c> as raw U8 E8M0 bytes.</summary>
    AmdQuark,

    /// <summary>EXL3: <c>X.trellis</c>, <c>X.suh</c>, <c>X.svh</c>, <c>X.mcg</c>.</summary>
    Exl3,

    /// <summary>MLX affine 4-bit: U32 <c>X.weight</c> with <c>X.scales</c> and <c>X.biases</c>, group size 64.</summary>
    Mlx,
}
