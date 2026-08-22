using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelAssets.BlockScale;

/// <summary>Index math for NVIDIA's "blocked" block-scale layout (the cuBLAS d-block-scaling-factors layout that ComfyUI's <c>comfy.float.to_blocked</c> produces). Both MXFP8 and NVFP4 ComfyUI checkpoints store their per-block scale tensors in this swizzled form rather than a plain <c>[rows, ceil(cols/group)]</c> matrix, so a portable (non-cuBLAS) dequant must invert the swizzle to find the scale for a given logical <c>(row, blockColumn)</c>.
///
/// <para><b>Forward layout</b> (reconstructed from <c>to_blocked</c>): a logical scale matrix <c>[H, W]</c> is zero-padded to <c>[128·ceil(H/128), 4·ceil(W/4)]</c>, then the (row, col) element lands at flat position <c>((g·32 + b)·16 + a·4 + d)</c> in the padded row-major buffer, where <c>rb = row/128</c>, <c>r128 = row%128</c>, <c>a = r128/32</c>, <c>b = r128%32</c>, <c>cb = col/4</c>, <c>d = col%4</c>, and <c>g = rb·ncb + cb</c> with <c>ncb = paddedCols/4</c>. The mapping is a pure permutation; <see cref="SwizzledIndex"/> returns that flat position so the caller can read the scale byte/element directly. Verified byte-exact against a swizzle round-trip of <c>comfy.float.to_blocked</c>.</para></summary>
public static class BlockScaleSwizzle
{
    /// <summary>Maps a logical <c>(row, blockColumn)</c> to the flat element index inside the swizzled scale buffer (row-major over its stored <c>[paddedRows, paddedCols]</c> shape).</summary>
    /// <param name="row">Logical output-row index (the weight's out dimension).</param>
    /// <param name="blockColumn">Logical block-column index (input position / group size).</param>
    /// <param name="paddedCols">Stored scale tensor's last-dim length (already a multiple of 4).</param>
    /// <remarks>Forwards to <see cref="Nvfp4ResidentCodec.SwizzledScaleIndex"/> so the permutation has exactly one definition. It lives in Core because the resident nvfp4 path needs it there and Core cannot reference ModelAssets; this type stays as the name every existing caller already knows.</remarks>
    public static long SwizzledIndex(long row, long blockColumn, long paddedCols)
        => Nvfp4ResidentCodec.SwizzledScaleIndex(row, blockColumn, paddedCols);
}
