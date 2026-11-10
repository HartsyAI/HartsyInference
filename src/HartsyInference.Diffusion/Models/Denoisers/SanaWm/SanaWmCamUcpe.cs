using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.SanaWm;

/// <summary>UCPE token transform on one batch item's token-major <c>[N, H·D]</c> tensor: each token's 4×4 matrix mixes every group
/// of four channels in the first <c>D/2</c> dims of each head, then RoPE rotates the last <c>D/2</c> dims. The matrix is applied as
/// <c>y = M·x</c> per group, as upstream <c>_apply_ray_projmat</c> does. One <see cref="IBackend.UcpeTransform"/>.</summary>
internal static class SanaWmCamUcpe
{
    /// <summary>Returns a new tensor shaped like <paramref name="input"/>; the input is left untouched.</summary>
    /// <param name="matrices">F32 <c>[N, 4, 4]</c> row-major per-token matrices.</param>
    /// <param name="cos">F32 <c>[N, D/2]</c> interleaved duplicated cosines.</param>
    /// <param name="sin">F32 <c>[N, D/2]</c> interleaved duplicated sines; negated for the inverse rotation.</param>
    internal static Tensor Apply(IBackend backend, Tensor input, Tensor matrices, Tensor cos, Tensor sin, int tokens, int heads,
        int headDim)
    {
        Tensor output = new(input.Shape, DType.F32);
        try
        {
            backend.UcpeTransform(output, input, matrices, cos, sin, tokens, heads, headDim);
            return output;
        }
        catch
        {
            output.Dispose();
            throw;
        }
    }
}
