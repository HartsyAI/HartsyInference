namespace HartsyInference.Cuda;

/// <summary>Which microscaling block-scale convention an FP4 operand pair uses. The two differ in block size and
/// scale element type, and cuBLASLt needs to be told which — the scale pointers alone do not say.</summary>
public enum Fp4BlockScaleFormat
{
    /// <summary>NVFP4: one E4M3 scale per 16 elements. What ComfyUI's <c>nvfp4</c> checkpoints ship.</summary>
    Nvfp4,

    /// <summary>MXFP4: one exponent-only UE8M0 scale per 32 elements. The OCP microscaling format, used by GPT-OSS.</summary>
    Mxfp4,
}
