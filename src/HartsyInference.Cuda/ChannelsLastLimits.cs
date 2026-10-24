namespace HartsyInference.Cuda;

/// <summary>Launch limits of the channels-last layout copies (channels_last.ptx). The tiled transpose puts its first
/// dimension on gridDim.y, which CUDA caps at 65,535 blocks, so a larger dimension is rejected at launch with
/// CUDA_ERROR_INVALID_VALUE. At 1024x896 one 3-D output frame is 917,504 elements, so any conv with 3 or more output
/// frames exceeds the cap.</summary>
internal static class ChannelsLastLimits
{
    internal const long MaxGridDimYZ = 65_535;
    internal const long TileDim = 32;

    /// <summary>True when <see cref="CudaKernels.LaunchTransposeTiled"/> accepts a [batch, d1, d2] transpose.</summary>
    internal static bool TransposeFits(long batch, long d1, long d2) =>
        batch is >= 1 and <= MaxGridDimYZ
        && d1 is >= 1 and <= int.MaxValue - TileDim
        && d2 is >= 1 and <= int.MaxValue - TileDim
        && (d1 + TileDim - 1) / TileDim <= MaxGridDimYZ;

    /// <summary>True when every layout copy of a channels-last conv fits: the input [xBatch, c, xSpatial], the weight
    /// [k, c, kernelSpatial] and the output [outBatch, outSpatial, k] transposes.</summary>
    internal static bool ConvFits(long xBatch, long c, long xSpatial, long k, long kernelSpatial, long outBatch, long outSpatial) =>
        TransposeFits(xBatch, c, xSpatial)
        && TransposeFits(k, c, kernelSpatial)
        && TransposeFits(outBatch, outSpatial, k);
}
